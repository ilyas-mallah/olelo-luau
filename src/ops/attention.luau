--!native
-- Causal attention over the KV cache with scalar reads. `attention_vector` does the same with the vector read.
--
-- The cache of a layer holds K and V as [kv head][position][hdPad] f32, where `hdPad` is the head size rounded up to
-- a multiple of 3 and the padding is zero. The queries come as [token][head][hdPad] in `q`.

const types = require("./types")

type Shape = types.Shape

--- Attention of `tokens` new queries at positions `pos0 + t`. The cache already holds their keys and values.
--- Results go to `out` as [token][head * headDim], rows `outStride` floats apart. `scores` is scratch for `context`
--- f32 values and `acc` scratch for `hdPad` numbers.
const function attend(
    s: Shape,
    q: buffer,
    tokens: number,
    keys: buffer,
    values: buffer,
    pos0: number,
    out: buffer,
    outStride: number,
    scores: buffer,
    acc: { number }
)
    const hd, hdPad, context = s.headDim, s.hdPad, s.context
    const group, headBase, kvBase = s.group, s.headBase, s.kvBase
    for t = 0, tokens - 1 do
        const last = pos0 + t
        local first = 0
        if s.window ~= nil and last - (s.window :: number) + 1 > 0 then
            first = last - (s.window :: number) + 1
        end
        for h = 0, s.heads - 1 do
            const g = (headBase + h) // group - kvBase
            const qb = ((t * s.heads) + h) * hdPad * 4
            const cacheBase = g * context * hdPad * 4
            local maxScore = -math.huge
            for j = first, last do
                const kb = cacheBase + j * hdPad * 4
                local dot = 0
                for d = 0, hd - 1 do
                    dot += buffer.readf32(q, qb + d * 4) * buffer.readf32(keys, kb + d * 4)
                end
                dot *= s.scale
                buffer.writef32(scores, (j - first) * 4, dot)
                if dot > maxScore then
                    maxScore = dot
                end
            end
            local total = 0
            for j = 0, last - first do
                const e = math.exp(buffer.readf32(scores, j * 4) - maxScore)
                buffer.writef32(scores, j * 4, e)
                total += e
            end
            for d = 0, hd - 1 do
                acc[d + 1] = 0
            end
            for j = 0, last - first do
                const p = buffer.readf32(scores, j * 4) / total
                const vb = cacheBase + (first + j) * hdPad * 4
                for d = 0, hd - 1 do
                    acc[d + 1] += p * buffer.readf32(values, vb + d * 4)
                end
            end
            const ob = (t * outStride + h * hd) * 4
            for d = 0, hd - 1 do
                buffer.writef32(out, ob + d * 4, acc[d + 1])
            end
        end
    end
end

return {
    attend = attend,
}
