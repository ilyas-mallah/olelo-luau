--!native
-- Prepares the activations of one stage for the kernel that reads them. f32 kernels read the f32 rows as they are. The
-- int8 kernels read the rows quantized with one scale per token. The Q4 kernel reads them quantized like llama.cpp
-- quantizes activations for Q8_0, int8 with one scale per group of 32. The packed Q8 kernel reads the same values as f64
-- with, per group, 128 times their sum, which cancels its offset binary weights (docs/kernels.md). The vector Q8 kernel
-- reads the splat form, see `prepare_vector`. One preparation serves every product that reads the same rows.

const formats = require("@src/tensor/formats")
const prepareVector = require("./prepare_vector")
const kernelsTypes = require("@src/kernels/types")
const types = require("./types")

type Scratch = types.Scratch

const GROUP = formats.GROUP

--- A word the activations round through to f32.
const rounder = buffer.create(8)

--- Bytes of a scratch for `tokens` rows of `maxK` values.
const function bytes(tokens: number, maxK: number): number
    return tokens * maxK + tokens * (maxK // GROUP) * (prepareVector.X_GROUP + 12)
end

--- A scratch for up to `tokens` rows of `maxK` values.
const function new(tokens: number, maxK: number): Scratch
    assert(maxK % GROUP == 0, "prepared rows are whole groups")
    return {
        tokens = tokens,
        maxK = maxK,
        q = buffer.create(tokens * maxK),
        x = buffer.create(tokens * (maxK // GROUP) * prepareVector.X_GROUP),
        xs = buffer.create(tokens * (maxK // GROUP) * 12),
    }
end

--- Quantizes `tokens` rows of `cols` f32 values (row stride `srcStride` floats) into `kpad` wide int8 rows with a scale
--- per token, the values past `cols` zero. Rounds as `quant.quantizeRows` does, so the two agree exactly.
const function quantizeTokens(s: Scratch, src: buffer, srcStride: number, cols: number, kpad: number, tokens: number)
    for t = 0, tokens - 1 do
        const rowAt = t * srcStride * 4
        local amax = 0
        for j = 0, cols - 1 do
            amax = math.max(amax, math.abs(buffer.readf32(src, rowAt + j * 4)))
        end
        buffer.writef32(s.xs, t * 4, if amax == 0 then 1 else amax / 127)
        -- The kernel multiplies by the f32 scale, so the values are rounded against that same scale.
        const stored = buffer.readf32(s.xs, t * 4)
        for j = 0, kpad - 1 do
            local r = 0
            if j < cols then
                const v = buffer.readf32(src, rowAt + j * 4) / stored
                r = math.clamp(if v >= 0 then math.floor(v + 0.5) else -math.floor(-v + 0.5), -127, 127)
            end
            buffer.writei8(s.q, t * kpad + j, r)
        end
    end
end

--- Quantizes the same rows per group of 32 as `quant.quantizeGroups` does, so the two agree exactly. With `packed` it
--- writes the packed form, otherwise int8 values and the group scales.
const function quantizeGroups(
    s: Scratch,
    src: buffer,
    srcStride: number,
    cols: number,
    kpad: number,
    tokens: number,
    packed: boolean
)
    const groups = kpad // GROUP
    for t = 0, tokens - 1 do
        const rowAt = t * srcStride * 4
        for g = 0, groups - 1 do
            const first = g * GROUP
            const used = math.max(math.min(cols - first, GROUP), 0)
            const at = rowAt + first * 4
            local amax = 0
            for j = 0, used - 1 do
                amax = math.max(amax, math.abs(buffer.readf32(src, at + j * 4)))
            end
            const d = formats.roundF32(amax / 127)
            const id = if d ~= 0 then formats.roundF32(1 / d) else 0
            local sum = 0
            -- The product goes through a scratch word to round to f32, and math.round rounds half away from zero like
            -- the reference. Adding 0 turns a rounded -0 into +0, so the bytes equal the reference conversion.
            if packed then
                const dst = (t * kpad + first) * 8
                for j = 0, used - 1 do
                    buffer.writef32(rounder, 0, buffer.readf32(src, at + j * 4) * id)
                    const r = math.round(buffer.readf32(rounder, 0)) + 0
                    buffer.writef64(s.x, dst + j * 8, r)
                    sum += r
                end
                for j = used, GROUP - 1 do
                    buffer.writef64(s.x, dst + j * 8, 0)
                end
                buffer.writef32(s.xs, (t * groups + g) * 8, sum * 128)
                buffer.writef32(s.xs, (t * groups + g) * 8 + 4, formats.roundF16(d))
            else
                const dst = t * kpad + first
                for j = 0, used - 1 do
                    buffer.writef32(rounder, 0, buffer.readf32(src, at + j * 4) * id)
                    buffer.writei8(s.q, dst + j, math.round(buffer.readf32(rounder, 0)) + 0)
                end
                for j = used, GROUP - 1 do
                    buffer.writei8(s.q, dst + j, 0)
                end
                buffer.writef32(s.xs, (t * groups + g) * 4, formats.roundF16(d))
            end
        end
    end
end

--- The activations `entry` reads, as the buffers to pass to the kernel: `src` itself for the f32 kernels, which read
--- rows of `kpad` floats, otherwise the scratch after preparing `tokens` rows of `cols` values from `src`.
const function run(
    entry: kernelsTypes.Entry,
    s: Scratch?,
    src: buffer,
    srcStride: number,
    cols: number,
    kpad: number,
    tokens: number
): (buffer, buffer?)
    if entry.activation == "f32" then
        assert(srcStride == kpad, "f32 kernels read activation rows of the tensor's stored length")
        return src, nil
    end
    assert(s ~= nil, "a kernel that reads another form needs the preparation scratch")
    assert(tokens <= s.tokens and kpad <= s.maxK, "the prepared rows do not fit the scratch")
    assert(cols <= kpad and cols <= srcStride, "more columns than the rows hold")
    if entry.activation == "packed" then
        quantizeGroups(s, src, srcStride, cols, kpad, tokens, true)
        return s.x, s.xs
    elseif entry.activation == "vsplat" then
        prepareVector.splat(s.x, s.xs, s.q, src, srcStride, cols, kpad, tokens)
        return s.x, s.xs
    elseif entry.activation == "int8g" then
        quantizeGroups(s, src, srcStride, cols, kpad, tokens, false)
        return s.q, s.xs
    end
    quantizeTokens(s, src, srcStride, cols, kpad, tokens)
    return s.q, s.xs
end

return {
    bytes = bytes,
    new = new,
    run = run,
}
