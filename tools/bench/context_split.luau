--!nonstrict
--!native
-- The attention of one decode token split into three timed parts, for tools/bench/context.luau. A replica of the loops of
-- `src/ops/attention_vector.luau`, so the sum of its parts can be checked against the whole op. Not strict, since the type
-- checker does not know `buffer.readvectorf32x3`, and not behind a function value, since that would lose its fast call.

--- Seconds of the score loop, the softmax and the value loop of one token at position `at` over every layer, per pass.
--- `shape` and `ws` are the session's attention shape and workspace, `keys` and `values` its f32 cache per layer.
const function time(
    clock: () -> number,
    shape: any,
    ws: any,
    keys: { buffer },
    values: { buffer },
    layers: number,
    at: number,
    passes: number
): (number, number, number)
    const hdPad, context = shape.hdPad, shape.context
    const group, headBase, kvBase = shape.group, shape.headBase, shape.kvBase
    const lanes = hdPad // 3
    local sSum, eSum, vSum = 0, 0, 0
    local sums: { vector } = {}
    for _ = 1, passes do
        for l = 1, layers do
            const k, v = keys[l], values[l]
            for h = 0, shape.heads - 1 do
                const g = (headBase + h) // group - kvBase
                const qb = h * hdPad * 4
                const cacheBase = g * context * hdPad * 4
                local maxScore = -math.huge
                const t0 = clock()
                for j = 0, at do
                    const kb = cacheBase + j * hdPad * 4
                    local sum = vector.zero
                    for d = 0, hdPad - 1, 3 do
                        sum = sum + buffer.readvectorf32x3(ws.qr, qb + d * 4) * buffer.readvectorf32x3(k, kb + d * 4)
                    end
                    const dot = (sum.x + sum.y + sum.z) * shape.scale
                    buffer.writef32(ws.scores, j * 4, dot)
                    if dot > maxScore then
                        maxScore = dot
                    end
                end
                const t1 = clock()
                local total = 0
                for j = 0, at do
                    const x = math.exp(buffer.readf32(ws.scores, j * 4) - maxScore)
                    buffer.writef32(ws.scores, j * 4, x)
                    total += x
                end
                const t2 = clock()
                for i = 1, lanes do
                    sums[i] = vector.zero
                end
                for j = 0, at do
                    const p = buffer.readf32(ws.scores, j * 4) / total
                    const vb = cacheBase + j * hdPad * 4
                    for i = 1, lanes do
                        sums[i] = sums[i] + buffer.readvectorf32x3(v, vb + (i - 1) * 12) * p
                    end
                end
                const t3 = clock()
                sSum += t1 - t0
                eSum += t2 - t1
                vSum += t3 - t2
            end
        end
    end
    return sSum / passes, eSum / passes, vSum / passes
end

return {
    time = time,
}
