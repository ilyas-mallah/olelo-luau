--!native
-- Causal attention over the Q8 KV cache, see `kvquant` for its layout and `attention` for the arguments. The keys and
-- values are int8 with a scale per head and position, which `Shape.keyScales` and `Shape.valueScales` hold for the
-- layer. It reads scalars, so it needs no vector flag.

const types = require("./types")

--- Same arguments and results as `attention.attend` over the layer's Q8 cache, the `acc` scratch is unused.
const function attend(
    s: types.Shape,
    q: buffer,
    tokens: number,
    keys: buffer,
    values: buffer,
    pos0: number,
    out: buffer,
    outStride: number,
    scores: buffer,
    _acc: { number }
)
    const keyScales, valueScales, window = s.keyScales, s.valueScales, s.window
    assert(keyScales ~= nil and valueScales ~= nil, "the Q8 attention reads the scales")
    const hd, hdPad, context = s.headDim, s.hdPad, s.context
    const group, headBase, kvBase = s.group, s.headBase, s.kvBase
    const scale = s.scale
    const hd4 = hd // 4 * 4
    const hd8 = hd // 8 * 8
    for t = 0, tokens - 1 do
        const last = pos0 + t
        local first = 0
        if window ~= nil and last - window + 1 > 0 then
            first = last - window + 1
        end
        const count = last - first
        for h = 0, s.heads - 1 do
            const g = (headBase + h) // group - kvBase
            const qb = ((t * s.heads) + h) * hdPad * 4
            const rowBase = g * context
            local maxScore = -math.huge
            for j = 0, count do
                const row = rowBase + first + j
                const kb = row * hd
                local a, b = 0, 0
                for d = 0, hd4 - 1, 4 do
                    const qo = qb + d * 4
                    a += buffer.readf32(q, qo) * buffer.readi8(keys, kb + d) + buffer.readf32(q, qo + 4) * buffer.readi8(
                        keys,
                        kb + d + 1
                    )
                    b += buffer.readf32(q, qo + 8) * buffer.readi8(keys, kb + d + 2) + buffer.readf32(q, qo + 12) * buffer.readi8(
                        keys,
                        kb + d + 3
                    )
                end
                for d = hd4, hd - 1 do
                    a += buffer.readf32(q, qb + d * 4) * buffer.readi8(keys, kb + d)
                end
                const dot = (a + b) * buffer.readf32(keyScales, row * 4) * scale
                buffer.writef32(scores, j * 4, dot)
                if dot > maxScore then
                    maxScore = dot
                end
            end
            local total = 0
            for j = 0, count do
                const e = math.exp(buffer.readf32(scores, j * 4) - maxScore)
                buffer.writef32(scores, j * 4, e)
                total += e
            end
            -- The weight of position j is its probability times its value scale.
            const inv = 1 / total
            const vRow = rowBase + first
            for j = 0, count do
                buffer.writef32(
                    scores,
                    j * 4,
                    buffer.readf32(scores, j * 4) * inv * buffer.readf32(valueScales, (vRow + j) * 4)
                )
            end
            const ob = (t * outStride + h * hd) * 4
            const vBase = vRow * hd
            for d = 0, hd8 - 1, 8 do
                local v0, v1, v2, v3, v4, v5, v6, v7 = 0, 0, 0, 0, 0, 0, 0, 0
                for j = 0, count do
                    const w = buffer.readf32(scores, j * 4)
                    const vb = vBase + j * hd + d
                    v0 += w * buffer.readi8(values, vb)
                    v1 += w * buffer.readi8(values, vb + 1)
                    v2 += w * buffer.readi8(values, vb + 2)
                    v3 += w * buffer.readi8(values, vb + 3)
                    v4 += w * buffer.readi8(values, vb + 4)
                    v5 += w * buffer.readi8(values, vb + 5)
                    v6 += w * buffer.readi8(values, vb + 6)
                    v7 += w * buffer.readi8(values, vb + 7)
                end
                const o = ob + d * 4
                buffer.writef32(out, o, v0)
                buffer.writef32(out, o + 4, v1)
                buffer.writef32(out, o + 8, v2)
                buffer.writef32(out, o + 12, v3)
                buffer.writef32(out, o + 16, v4)
                buffer.writef32(out, o + 20, v5)
                buffer.writef32(out, o + 24, v6)
                buffer.writef32(out, o + 28, v7)
            end
            for d = hd8, hd - 1 do
                local v = 0
                for j = 0, count do
                    v += buffer.readf32(scores, j * 4) * buffer.readi8(values, vBase + j * hd + d)
                end
                buffer.writef32(out, ob + d * 4, v)
            end
        end
    end
end

return {
    attend = attend,
}
