--!native
-- Stores keys or values in the Q8 KV cache: int8 values with one f32 scale per KV head and position.
--
-- A layer's cache is [kv head][position][headDim] int8 and [kv head][position] f32 scales. The scale of a row is
-- amax / 127 and a value is round(x / scale), so the largest value of a row keeps full resolution and a row of zeros
-- has scale 0. The row of a head is not padded.
--
-- Keys may be stored with a centre vector per head subtracted. A constant vector subtracted from every key changes
-- every score of a query by the same amount, which the softmax does not see, so attention reads the stored keys as they
-- are. It matters where the keys have a large offset in some channels (Qwen2.5's key biases): without it the largest
-- value of a row sets the step for every channel and the rest are rounded away.

-- Two layouts hold the int8 rows. `i8` is `headDim` bytes a row, read by the scalar `attention_q8`. `v3` is for the
-- vector read: three values to an f32 lane as `c + b / 256 + a / 65536` where `c`, `b` and `a` are the values plus 128
-- (offset binary), which f32 holds exactly, and three lanes to a vector of 12 bytes. Vector `v` of a row holds the
-- values of dimensions `9v + 3p + i` in byte `p` (0 is `c`) of lane `i`, and a row is padded to whole vectors with the
-- value 0. `attention_q8_unrolled` reads it, see `tools/gen/attention_q8.luau`.

--- Tokens whose mean is the centre of a sequence's keys.
const CENTRE_TOKENS = 32

--- The packed value of a lane of zeros, 128 in each byte.
const ZERO_LANE = 128 + 128 / 256 + 128 / 65536

--- Vectors a `v3` row of `headDim` values has.
const function vectors(headDim: number): number
    return ((headDim + 2) // 3 + 2) // 3
end

--- Bytes of a row of `headDim` values in `layout`, "i8" or "v3".
const function rowBytes(headDim: number, layout: string): number
    return if layout == "v3" then vectors(headDim) * 12 else headDim
end

--- The offset binary byte of dimension `dim` of the row at float `s` of `src`, less the centre `c` at float `cs`, for a
--- row whose values are scaled by `inv`. Dimensions past the head are 0.
const function byteAt(src: buffer, s: number, c: buffer?, cs: number, dim: number, headDim: number, inv: number): number
    if dim >= headDim then
        return 128
    end
    local x = buffer.readf32(src, s + dim * 4)
    if c ~= nil then
        x -= buffer.readf32(c :: buffer, cs + dim * 4)
    end
    return math.round(x * inv) + 128
end

--- Sets `out` (`heads` times `headDim` f32) to the mean of the first `CENTRE_TOKENS` of the `tokens` rows, laid out as
--- for `store`, for each head whose rows get a clearly smaller largest value from it (under 90% of the mean largest
--- value), and to zero for the others, since a mean of few rows only adds noise to keys that have no offset. The first
--- rows of a sequence set the centre once, for all its later keys.
const function centre(
    src: buffer,
    srcStride: number,
    headStride: number,
    heads: number,
    headDim: number,
    tokens: number,
    out: buffer
)
    const n = math.min(tokens, CENTRE_TOKENS)
    for h = 0, heads - 1 do
        for i = 0, headDim - 1 do
            local sum = 0
            for t = 0, n - 1 do
                sum += buffer.readf32(src, (t * srcStride + h * headStride + i) * 4)
            end
            buffer.writef32(out, (h * headDim + i) * 4, sum / n)
        end
        local plain, centred = 0, 0
        for t = 0, n - 1 do
            const s = (t * srcStride + h * headStride) * 4
            local a, b = 0, 0
            for i = 0, headDim - 1 do
                const v = buffer.readf32(src, s + i * 4)
                a = math.max(a, math.abs(v))
                b = math.max(b, math.abs(v - buffer.readf32(out, (h * headDim + i) * 4)))
            end
            plain += a
            centred += b
        end
        if centred >= 0.9 * plain then
            buffer.fill(out, h * headDim * 4, 0, headDim * 4)
        end
    end
end

--- Stores `tokens` rows of `heads` heads of `headDim` f32 values, row `t` at position `pos0 + t` of every head's cache.
--- Head `h` of token `t` starts at float `t * srcStride + h * headStride` of `src`. `context` is the positions a head's
--- cache holds. With a `centre` (see `centre`) its values are subtracted before the rows are rounded.
const function store(
    src: buffer,
    srcStride: number,
    headStride: number,
    heads: number,
    headDim: number,
    tokens: number,
    dst: buffer,
    scales: buffer,
    pos0: number,
    context: number,
    centre: buffer?,
    layout: string?
)
    const c = centre
    const packed = layout == "v3"
    const vecs = vectors(headDim)
    for t = 0, tokens - 1 do
        for h = 0, heads - 1 do
            const s = (t * srcStride + h * headStride) * 4
            const row = h * context + pos0 + t
            const cs = h * headDim * 4
            local amax = 0
            if c == nil then
                for i = 0, headDim - 1 do
                    const v = math.abs(buffer.readf32(src, s + i * 4))
                    if v > amax then
                        amax = v
                    end
                end
            else
                for i = 0, headDim - 1 do
                    const v = math.abs(buffer.readf32(src, s + i * 4) - buffer.readf32(c, cs + i * 4))
                    if v > amax then
                        amax = v
                    end
                end
            end
            const d = row * (if packed then vecs * 12 else headDim)
            if amax == 0 then
                buffer.writef32(scales, row * 4, 0)
                if packed then
                    for lane = 0, vecs * 3 - 1 do
                        buffer.writef32(dst, d + lane * 4, ZERO_LANE)
                    end
                else
                    for i = 0, headDim - 1 do
                        buffer.writei8(dst, d + i, 0)
                    end
                end
            else
                const inv = 127 / amax
                buffer.writef32(scales, row * 4, amax / 127)
                if packed then
                    for v = 0, vecs - 1 do
                        for i = 0, 2 do
                            const base = 9 * v + i
                            const cb = byteAt(src, s, c, cs, base, headDim, inv)
                            const bb = byteAt(src, s, c, cs, base + 3, headDim, inv)
                            const ab = byteAt(src, s, c, cs, base + 6, headDim, inv)
                            buffer.writef32(dst, d + v * 12 + i * 4, cb + bb / 256 + ab / 65536)
                        end
                    end
                elseif c == nil then
                    for i = 0, headDim - 1 do
                        buffer.writei8(dst, d + i, math.round(buffer.readf32(src, s + i * 4) * inv))
                    end
                else
                    for i = 0, headDim - 1 do
                        buffer.writei8(
                            dst,
                            d + i,
                            math.round((buffer.readf32(src, s + i * 4) - buffer.readf32(c, cs + i * 4)) * inv)
                        )
                    end
                end
            end
        end
    end
end

return {
    store = store,
    centre = centre,
    vectors = vectors,
    rowBytes = rowBytes,
}
