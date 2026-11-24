-- Both attention implementations against a plain computation in doubles, with grouped heads, a window and padded heads.
const attention = require("@src/ops/attention")
const attentionUnrolled = require("@src/ops/generated")
const attentionVector = require("@src/ops/attention_vector")
const data = require("@tests/data")
const harness = require("@tests/harness")
const opsTypes = require("@src/ops/types")

const HEADS, KV_HEADS, HEAD_DIM, HD_PAD, CONTEXT = 6, 2, 10, 12, 20

const function fill(b: buffer, rng: data.Rng, count: number, stride: number, used: number)
    for i = 0, count - 1 do
        for d = 0, used - 1 do
            buffer.writef32(b, (i * stride + d) * 4, data.nextFloat(rng, 1))
        end
    end
end

--- The result computed in doubles straight from the definition.
const function expected(
    q: buffer,
    keys: buffer,
    values: buffer,
    token: number,
    pos: number,
    head: number,
    window: number?,
    d: number
): number
    const g = head // (HEADS // KV_HEADS)
    local first = 0
    if window ~= nil and pos - window + 1 > 0 then
        first = pos - window + 1
    end
    const scores: { number } = {}
    local maxScore = -math.huge
    for j = first, pos do
        local dot = 0
        for e = 0, HEAD_DIM - 1 do
            dot += buffer.readf32(q, ((token * HEADS + head) * HD_PAD + e) * 4) * buffer.readf32(
                keys,
                ((g * CONTEXT + j) * HD_PAD + e) * 4
            )
        end
        scores[j - first + 1] = dot / math.sqrt(HEAD_DIM)
        maxScore = math.max(maxScore, scores[j - first + 1])
    end
    local total = 0
    for k, s in scores do
        scores[k] = math.exp(s - maxScore)
        total += scores[k]
    end
    local out = 0
    for j = first, pos do
        out += scores[j - first + 1] / total * buffer.readf32(values, ((g * CONTEXT + j) * HD_PAD + d) * 4)
    end
    return out
end

--- Runs `attend` for the query heads `headBase` to `headBase + heads - 1` only, on the cache of the KV heads they read,
--- and compares with the whole definition.
const function check(
    h: harness.Harness,
    attend: typeof(attention.attend),
    window: number?,
    tolerance: number,
    headBase: number?,
    heads: number?
)
    const rng = data.newRng(31)
    const q = buffer.create(3 * HEADS * HD_PAD * 4)
    const keys = buffer.create(KV_HEADS * CONTEXT * HD_PAD * 4)
    const values = buffer.create(KV_HEADS * CONTEXT * HD_PAD * 4)
    fill(q, rng, 3 * HEADS, HD_PAD, HEAD_DIM)
    fill(keys, rng, KV_HEADS * CONTEXT, HD_PAD, HEAD_DIM)
    fill(values, rng, KV_HEADS * CONTEXT, HD_PAD, HEAD_DIM)
    const group = HEADS // KV_HEADS
    const first, count = headBase or 0, heads or HEADS
    const kvFirst = first // group
    const kvCount = (first + count - 1) // group - kvFirst + 1
    const cacheBytes = CONTEXT * HD_PAD * 4
    -- The share's own queries and cache, copied out of the whole.
    const qs = buffer.create(3 * count * HD_PAD * 4)
    for t = 0, 2 do
        buffer.copy(qs, t * count * HD_PAD * 4, q, (t * HEADS + first) * HD_PAD * 4, count * HD_PAD * 4)
    end
    const ks, vs = buffer.create(kvCount * cacheBytes), buffer.create(kvCount * cacheBytes)
    buffer.copy(ks, 0, keys, kvFirst * cacheBytes, kvCount * cacheBytes)
    buffer.copy(vs, 0, values, kvFirst * cacheBytes, kvCount * cacheBytes)
    const shape: opsTypes.Shape = {
        heads = count,
        kvHeads = kvCount,
        group = group,
        headBase = first,
        kvBase = kvFirst,
        headDim = HEAD_DIM,
        hdPad = HD_PAD,
        context = CONTEXT,
        scale = 1 / math.sqrt(HEAD_DIM),
        window = window,
    }
    const stride = count * HEAD_DIM + 4
    const out = buffer.create(3 * stride * 4)
    attend(shape, qs, 3, ks, vs, 9, out, stride, buffer.create(CONTEXT * 4), table.create(HD_PAD, 0))
    for t = 0, 2 do
        for head = 0, count - 1 do
            for d = 0, HEAD_DIM - 1 do
                const got = buffer.readf32(out, (t * stride + head * HEAD_DIM + d) * 4)
                h.near(got, expected(q, keys, values, t, 9 + t, first + head, window, d), tolerance)
            end
        end
    end
end

return function(h: harness.Harness)
    h.case("scalar attention matches the definition", function()
        check(h, attention.attend, nil, 1e-6)
    end)

    h.case("scalar attention with a window", function()
        check(h, attention.attend, 4, 1e-6)
    end)

    h.case("scalar attention over the heads of a share, also when a group of heads is cut", function()
        check(h, attention.attend, nil, 1e-6, 2, 3)
        check(h, attention.attend, 4, 1e-6, 4, 2)
        check(h, attention.attend, nil, 1e-6, 0, 1)
    end)

    if (buffer :: any).readvectorf32x3 == nil then
        h.skip("vector attention", "buffer.readvectorf32x3 does not exist in this mode")
        return
    end

    h.case("vector attention matches the definition", function()
        check(h, attentionVector.attend, nil, 1e-5)
    end)

    h.case("vector attention with a window", function()
        check(h, attentionVector.attend, 4, 1e-5)
    end)

    h.case("vector attention over the heads of a share", function()
        check(h, attentionVector.attend, nil, 1e-5, 2, 3)
        check(h, attentionVector.attend, 4, 1e-5, 4, 2)
    end)

    const unrolled = attentionUnrolled.byLanes[HD_PAD // 3]
    h.case(
        "unrolled attention matches the definition, with windows that leave 1 to 3 positions past a group of 4",
        function()
            for _, window in { 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12 } do
                check(h, unrolled, window, 1e-5)
            end
            check(h, unrolled, nil, 1e-5)
        end
    )

    h.case("unrolled attention over the heads of a share", function()
        check(h, unrolled, nil, 1e-5, 2, 3)
        check(h, unrolled, 4, 1e-5, 4, 2)
    end)
end
