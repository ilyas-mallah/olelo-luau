--!nonstrict
--!native
-- Causal attention like `attention`, with `buffer.readvectorf32x3` for the dot products and the value sum. Strict
-- typing is off because the type checker does not know that function. Only select it when the probe finds the fast
-- call, see `attention` for the cache layout.

const types = require("./types")

--- Same arguments and results as `attention.attend`, the `acc` scratch is unused.
--- The scores use two running sums, and the value sum keeps four lanes of the head in locals while it runs over the
--- positions, so no vector goes through a table.
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
    const hd, hdPad, context = s.headDim, s.hdPad, s.context
    const group, headBase, kvBase = s.group, s.headBase, s.kvBase
    const scale = s.scale
    const lanes = hdPad // 3
    const pairs = lanes // 2
    const odd = lanes % 2 == 1
    const quads = lanes // 4
    const rowBytes = hdPad * 4
    --- Writes the three values of lane `i`, leaving out the head padding.
    const function store(v: vector, ob: number, i: number)
        const d = i * 3
        if d + 2 < hd then
            buffer.writef32(out, ob + d * 4, v.x)
            buffer.writef32(out, ob + (d + 1) * 4, v.y)
            buffer.writef32(out, ob + (d + 2) * 4, v.z)
        else
            if d < hd then
                buffer.writef32(out, ob + d * 4, v.x)
            end
            if d + 1 < hd then
                buffer.writef32(out, ob + (d + 1) * 4, v.y)
            end
        end
    end
    for t = 0, tokens - 1 do
        const last = pos0 + t
        local first = 0
        if s.window ~= nil and last - s.window + 1 > 0 then
            first = last - s.window + 1
        end
        const count = last - first
        for h = 0, s.heads - 1 do
            const g = (headBase + h) // group - kvBase
            const qb = ((t * s.heads) + h) * rowBytes
            const cacheBase = g * context * rowBytes
            local maxScore = -math.huge
            for j = 0, count do
                const kb = cacheBase + (first + j) * rowBytes
                local a, b = vector.zero, vector.zero
                for i = 0, pairs - 1 do
                    const o = i * 24
                    a = a + buffer.readvectorf32x3(q, qb + o) * buffer.readvectorf32x3(keys, kb + o)
                    b = b + buffer.readvectorf32x3(q, qb + o + 12) * buffer.readvectorf32x3(keys, kb + o + 12)
                end
                if odd then
                    const o = pairs * 24
                    a = a + buffer.readvectorf32x3(q, qb + o) * buffer.readvectorf32x3(keys, kb + o)
                end
                const sum = a + b
                const dot = (sum.x + sum.y + sum.z) * scale
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
            const inv = 1 / total
            const ob = (t * outStride + h * hd) * 4
            const vBase = cacheBase + first * rowBytes
            for qd = 0, quads - 1 do
                const o = qd * 48
                local v0, v1, v2, v3 = vector.zero, vector.zero, vector.zero, vector.zero
                for j = 0, count do
                    const p = buffer.readf32(scores, j * 4)
                    const vb = vBase + j * rowBytes + o
                    v0 = v0 + buffer.readvectorf32x3(values, vb) * p
                    v1 = v1 + buffer.readvectorf32x3(values, vb + 12) * p
                    v2 = v2 + buffer.readvectorf32x3(values, vb + 24) * p
                    v3 = v3 + buffer.readvectorf32x3(values, vb + 36) * p
                end
                store(v0 * inv, ob, qd * 4)
                store(v1 * inv, ob, qd * 4 + 1)
                store(v2 * inv, ob, qd * 4 + 2)
                store(v3 * inv, ob, qd * 4 + 3)
            end
            for i = quads * 4, lanes - 1 do
                local v0 = vector.zero
                for j = 0, count do
                    v0 = v0
                        + buffer.readvectorf32x3(values, vBase + j * rowBytes + i * 12)
                            * buffer.readf32(scores, j * 4)
                end
                store(v0 * inv, ob, i)
            end
        end
    end
end

return {
    attend = attend,
}
