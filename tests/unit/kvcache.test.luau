-- The Q8 KV cache: the quantizer, attention over it, and a model that runs on it against the f32 cache.
const attention = require("@src/ops/attention")
const attentionQ8 = require("@src/ops/attention_q8")
const attentionQ8Gen = require("@tools/gen/attention_q8")
const attentionUnrolled = require("@src/ops/generated")
const config = require("@src/model/config")
const data = require("@tests/data")
const engineModule = require("@src/api/engine")
const graph = require("@src/model/graph")
const harness = require("@tests/harness")
const kvquant = require("@src/ops/kvquant")
const lute = require("@backends/lute")
const session = require("@src/model/session")
const apiTypes = require("@src/api/types")
const opsTypes = require("@src/ops/types")

const KV_HEADS, HEAD_DIM, HD_PAD, CONTEXT = 2, 10, 12, 20

--- Random rows of `headDim` values (`HEAD_DIM` when nil), one per head and position, laid out as the f32 cache is.
const function randomCache(rng: data.Rng, headDim: number?): buffer
    const dim = headDim or HEAD_DIM
    const pad = config.headPad(dim)
    const b = buffer.create(KV_HEADS * CONTEXT * pad * 4)
    for i = 0, KV_HEADS * CONTEXT - 1 do
        for d = 0, dim - 1 do
            buffer.writef32(b, (i * pad + d) * 4, data.nextFloat(rng, 1 + (i % 5)))
        end
    end
    return b
end

return function(h: harness.Harness)
    h.case("the quantizer keeps every value within half a step and the largest exactly", function()
        const rng = data.newRng(5)
        const src = buffer.create(3 * 2 * 8 * 4)
        for i = 0, 3 * 2 * 8 - 1 do
            buffer.writef32(src, i * 4, data.nextFloat(rng, 3))
        end
        -- Row 1 of head 1 is all zeros.
        for i = 0, 7 do
            buffer.writef32(src, ((1 * 2 + 1) * 8 + i) * 4, 0)
        end
        const context = 6
        const dst = buffer.create(2 * context * 8)
        const scales = buffer.create(2 * context * 4)
        -- Three tokens of two heads of 8 values go to positions 2 to 4.
        kvquant.store(src, 16, 8, 2, 8, 3, dst, scales, 2, context)
        for t = 0, 2 do
            for head = 0, 1 do
                const row = head * context + 2 + t
                const d = buffer.readf32(scales, row * 4)
                local amax = 0
                for i = 0, 7 do
                    amax = math.max(amax, math.abs(buffer.readf32(src, ((t * 2 + head) * 8 + i) * 4)))
                end
                h.near(d, amax / 127, 1e-9, "the scale is amax / 127")
                for i = 0, 7 do
                    const x = buffer.readf32(src, ((t * 2 + head) * 8 + i) * 4)
                    const q = buffer.readi8(dst, row * 8 + i)
                    h.truthy(math.abs(q * d - x) <= d / 2 + 1e-7, "the error is within half a step")
                    h.truthy(q >= -127 and q <= 127)
                end
                if amax > 0 then
                    local top = 0
                    for i = 0, 7 do
                        top = math.max(top, math.abs(buffer.readi8(dst, row * 8 + i)))
                    end
                    h.equal(top, 127, "the largest value uses the full range")
                else
                    h.equal(d, 0)
                end
            end
        end
        -- Positions outside the stored ones stay untouched.
        h.equal(buffer.readf32(scales, 0), 0)
        h.equal(buffer.readf32(scales, (context + 5) * 4), 0)
    end)

    --- The int8 value of dimension `d` of row `j` of a stored head, taken out of the layout.
    const function storedValue(rows: buffer, j: number, d: number, layout: string, headDim: number): number
        if layout ~= "v3" then
            return buffer.readi8(rows, j * headDim + d)
        end
        const rowB = kvquant.rowBytes(headDim, layout)
        const v, rest = d // 9, d % 9
        const lane = buffer.readf32(rows, j * rowB + v * 12 + (rest % 3) * 4)
        const c = math.floor(lane)
        const x = (lane - c) * 256
        const b = math.floor(x)
        const bytes = { c, b, (x - b) * 256 }
        return bytes[rest // 3 + 1] - 128
    end

    --- Attention over a Q8 cache against scalar f32 attention over the values that cache holds, for the query heads
    --- `headBase` to `headBase + heads - 1`.
    const function checkAttention(window: number?, headBase: number, heads: number, layout: string, size: number?)
        const HEAD_DIM = size or 10
        const HD_PAD = config.headPad(HEAD_DIM)
        const rowB = kvquant.rowBytes(HEAD_DIM, layout)
        const attend = if layout == "v3" then attentionUnrolled.q8ByLanes[HD_PAD // 3] else attentionQ8.attend
        const rng = data.newRng(77)
        const HEADS = 6
        const group = HEADS // KV_HEADS
        const keys, values = randomCache(rng, HEAD_DIM), randomCache(rng, HEAD_DIM)
        const q = buffer.create(3 * HEADS * HD_PAD * 4)
        for i = 0, 3 * HEADS - 1 do
            for d = 0, HEAD_DIM - 1 do
                buffer.writef32(q, (i * HD_PAD + d) * 4, data.nextFloat(rng, 1))
            end
        end
        -- The quantized cache and the f32 cache of the values it holds.
        const kq, kScales = buffer.create(KV_HEADS * CONTEXT * rowB), buffer.create(KV_HEADS * CONTEXT * 4)
        const vq, vScales = buffer.create(KV_HEADS * CONTEXT * rowB), buffer.create(KV_HEADS * CONTEXT * 4)
        const kDeq, vDeq = buffer.create(buffer.len(keys)), buffer.create(buffer.len(values))
        for head = 0, KV_HEADS - 1 do
            const src = head * CONTEXT * HD_PAD * 4
            for _, pair in { { keys, kq, kScales, kDeq }, { values, vq, vScales, vDeq } } do
                const f, qb, sb, deq = pair[1], pair[2], pair[3], pair[4]
                -- A view of this head's rows, so one call stores its positions.
                const view = buffer.create(CONTEXT * HD_PAD * 4)
                buffer.copy(view, 0, f, src, CONTEXT * HD_PAD * 4)
                const qh, sh = buffer.create(CONTEXT * rowB), buffer.create(CONTEXT * 4)
                kvquant.store(view, HD_PAD, HD_PAD, 1, HEAD_DIM, CONTEXT, qh, sh, 0, CONTEXT, nil, layout)
                buffer.copy(qb, head * CONTEXT * rowB, qh, 0, CONTEXT * rowB)
                buffer.copy(sb, head * CONTEXT * 4, sh, 0, CONTEXT * 4)
                for j = 0, CONTEXT - 1 do
                    for d = 0, HEAD_DIM - 1 do
                        buffer.writef32(
                            deq,
                            src + (j * HD_PAD + d) * 4,
                            storedValue(qh, j, d, layout, HEAD_DIM) * buffer.readf32(sh, j * 4)
                        )
                    end
                end
            end
        end
        -- The share's queries and cache of the KV heads its query heads read.
        const kvFirst = headBase // group
        const kvCount = (headBase + heads - 1) // group - kvFirst + 1
        const qs = buffer.create(3 * heads * HD_PAD * 4)
        for t = 0, 2 do
            buffer.copy(qs, t * heads * HD_PAD * 4, q, (t * HEADS + headBase) * HD_PAD * 4, heads * HD_PAD * 4)
        end
        const function slice(b: buffer, bytes: number): buffer
            const out = buffer.create(kvCount * bytes)
            buffer.copy(out, 0, b, kvFirst * bytes, kvCount * bytes)
            return out
        end
        const shape: opsTypes.Shape = {
            heads = heads,
            kvHeads = kvCount,
            group = group,
            headBase = headBase,
            kvBase = kvFirst,
            headDim = HEAD_DIM,
            hdPad = HD_PAD,
            context = CONTEXT,
            scale = 1 / math.sqrt(HEAD_DIM),
            window = window,
            keyScales = slice(kScales, CONTEXT * 4),
            valueScales = slice(vScales, CONTEXT * 4),
        }
        const stride = heads * HEAD_DIM + 4
        const got, want = buffer.create(3 * stride * 4), buffer.create(3 * stride * 4)
        attend(
            shape,
            qs,
            3,
            slice(kq, CONTEXT * rowB),
            slice(vq, CONTEXT * rowB),
            9,
            got,
            stride,
            buffer.create(CONTEXT * 4),
            {}
        )
        attention.attend(
            shape,
            qs,
            3,
            slice(kDeq, CONTEXT * HD_PAD * 4),
            slice(vDeq, CONTEXT * HD_PAD * 4),
            9,
            want,
            stride,
            buffer.create(CONTEXT * 4),
            table.create(HD_PAD, 0)
        )
        for t = 0, 2 do
            for i = 0, heads * HEAD_DIM - 1 do
                h.near(buffer.readf32(got, (t * stride + i) * 4), buffer.readf32(want, (t * stride + i) * 4), 1e-5)
            end
        end
    end

    h.case("the vector layout holds the integers of the int8 layout, with and without a centre", function()
        const rng = data.newRng(19)
        const heads, dim, tokens, context = 2, HEAD_DIM, 5, 9
        const src = buffer.create(tokens * heads * dim * 4)
        for i = 0, tokens * heads * dim - 1 do
            buffer.writef32(src, i * 4, data.nextFloat(rng, 4))
        end
        -- One row of zeros, whose scale is 0 and whose bytes are the value 0.
        for i = 0, dim - 1 do
            buffer.writef32(src, (1 * heads * dim + dim + i) * 4, 0)
        end
        const centre = buffer.create(heads * dim * 4)
        for i = 0, heads * dim - 1 do
            buffer.writef32(centre, i * 4, data.nextFloat(rng, 1))
        end
        for _, c in { nil, centre } do
            const flat, flatScales = buffer.create(heads * context * dim), buffer.create(heads * context * 4)
            const rowB = kvquant.rowBytes(dim, "v3")
            const packed, packedScales = buffer.create(heads * context * rowB), buffer.create(heads * context * 4)
            kvquant.store(src, heads * dim, dim, heads, dim, tokens, flat, flatScales, 2, context, c, "i8")
            kvquant.store(src, heads * dim, dim, heads, dim, tokens, packed, packedScales, 2, context, c, "v3")
            for head = 0, heads - 1 do
                for j = 2, 2 + tokens - 1 do
                    h.equal(
                        buffer.readf32(packedScales, (head * context + j) * 4),
                        buffer.readf32(flatScales, (head * context + j) * 4)
                    )
                    for d = 0, dim - 1 do
                        h.equal(
                            storedValue(packed, head * context + j, d, "v3", dim),
                            buffer.readi8(flat, (head * context + j) * dim + d)
                        )
                    end
                end
            end
        end
    end)

    h.case("attention over a Q8 cache equals attention over the values it holds", function()
        checkAttention(nil, 0, 6, "i8")
    end)

    h.case("Q8 attention with a window and over the heads of a share", function()
        checkAttention(4, 0, 6, "i8")
        checkAttention(nil, 2, 3, "i8")
        checkAttention(4, 4, 2, "i8")
    end)

    if (buffer :: any).readvectorf32x3 ~= nil then
        h.case("vector layout attention equals attention over the values the cache holds", function()
            checkAttention(nil, 0, 6, "v3")
        end)

        h.case("vector layout attention with windows that leave 1 to 3 positions past a group of 4", function()
            for _, window in { 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12 } do
                checkAttention(window, 0, 6, "v3")
            end
        end)

        h.case(
            "vector layout attention of every generated head size equals attention over the stored values",
            function()
                for _, lanes in attentionQ8Gen.LANES do
                    -- `lanes * 3 - 1` leaves a lane with two values, and `lanes * 3` fills the last lane.
                    for _, headDim in { lanes * 3 - 1, lanes * 3 } do
                        h.truthy(attentionUnrolled.q8ByLanes[config.headPad(headDim) // 3] ~= nil, `lanes {lanes}`)
                        checkAttention(nil, 0, 6, "v3", headDim)
                        checkAttention(5, 2, 3, "v3", headDim)
                    end
                end
            end
        )

        h.case("vector layout attention over the heads of a share", function()
            checkAttention(nil, 2, 3, "v3")
            checkAttention(4, 4, 2, "v3")
        end)
    end

    h.case("centring the keys keeps attention and rescues a cache whose keys have a large offset", function()
        const rng = data.newRng(31)
        const HEADS = 2
        const keys, values = randomCache(rng), randomCache(rng)
        -- Channels 0 to 2 of every key sit far from zero, as the key biases of Qwen2.5 put them.
        for j = 0, CONTEXT - 1 do
            for d = 0, 2 do
                buffer.writef32(keys, (j * HD_PAD + d) * 4, buffer.readf32(keys, (j * HD_PAD + d) * 4) + 40)
            end
        end
        const q = buffer.create(HEADS * HD_PAD * 4)
        for i = 0, HEADS - 1 do
            for d = 0, HEAD_DIM - 1 do
                buffer.writef32(q, (i * HD_PAD + d) * 4, data.nextFloat(rng, 1))
            end
        end
        const shape: opsTypes.Shape = {
            heads = HEADS,
            kvHeads = 1,
            group = 2,
            headBase = 0,
            kvBase = 0,
            headDim = HEAD_DIM,
            hdPad = HD_PAD,
            context = CONTEXT,
            scale = 1 / math.sqrt(HEAD_DIM),
            window = nil,
            keyScales = buffer.create(CONTEXT * 4),
            valueScales = buffer.create(CONTEXT * 4),
        }
        const stride = HEADS * HEAD_DIM + 4
        const want = buffer.create(stride * 4)
        attention.attend(
            shape,
            q,
            1,
            keys,
            values,
            CONTEXT - 1,
            want,
            stride,
            buffer.create(CONTEXT * 4),
            table.create(HD_PAD, 0)
        )
        const vq = buffer.create(CONTEXT * HEAD_DIM)
        kvquant.store(values, HD_PAD, HD_PAD, 1, HEAD_DIM, CONTEXT, vq, shape.valueScales :: buffer, 0, CONTEXT, nil)
        const function errorWith(centred: boolean): number
            const kq = buffer.create(CONTEXT * HEAD_DIM)
            local centre: buffer? = nil
            if centred then
                centre = buffer.create(HEAD_DIM * 4)
                kvquant.centre(keys, HD_PAD, HD_PAD, 1, HEAD_DIM, CONTEXT, centre :: buffer)
            end
            kvquant.store(keys, HD_PAD, HD_PAD, 1, HEAD_DIM, CONTEXT, kq, shape.keyScales :: buffer, 0, CONTEXT, centre)
            const got = buffer.create(stride * 4)
            attentionQ8.attend(shape, q, 1, kq, vq, CONTEXT - 1, got, stride, buffer.create(CONTEXT * 4), {})
            local sum, size = 0, 0
            for i = 0, HEADS * HEAD_DIM - 1 do
                const x = buffer.readf32(want, i * 4)
                sum += (x - buffer.readf32(got, i * 4)) ^ 2
                size += x * x
            end
            return math.sqrt(sum / size)
        end
        const plain, centred = errorWith(false), errorWith(true)
        h.truthy(centred < 0.01, `centred keys give {centred} relative error`)
        h.truthy(centred * 3 < plain, `{centred} centred against {plain} plain`)
    end)

    h.case(
        "the centre is the mean of the first 32 rows of a head with an offset, and zero for a head without",
        function()
            const rng = data.newRng(8)
            const tokens, heads, dim = 40, 2, 4
            const src = buffer.create(tokens * heads * dim * 4)
            -- Head 0 has an offset the centre removes, head 1 has none and gets no centre.
            for i = 0, tokens * heads * dim - 1 do
                const head = (i // dim) % heads
                buffer.writef32(src, i * 4, data.nextFloat(rng, 5) + if head == 0 then 30 else 0)
            end
            const out = buffer.create(heads * dim * 4)
            kvquant.centre(src, heads * dim, dim, heads, dim, tokens, out)
            for head = 0, heads - 1 do
                for d = 0, dim - 1 do
                    local sum = 0
                    for t = 0, 31 do
                        sum += buffer.readf32(src, ((t * heads + head) * dim + d) * 4)
                    end
                    h.near(buffer.readf32(out, (head * dim + d) * 4), if head == 0 then sum / 32 else 0, 1e-6)
                end
            end
            kvquant.centre(src, heads * dim, dim, heads, dim, 3, out)
            local sum = 0
            for t = 0, 2 do
                sum += buffer.readf32(src, (t * heads * dim) * 4)
            end
            h.near(buffer.readf32(out, 0), sum / 3, 1e-6, "a short call averages what it has")
        end
    )

    const MODEL = "tests/fixtures/tiny-llama/model.olelo"
    const function open(kv: string?): apiTypes.Engine
        return engineModule.new({
            host = lute.host(),
            source = lute.fileSource(MODEL),
            context = 128,
            chunk = 8,
            kv = kv,
        })
    end

    --- The logits of the last token of `ids` run in passes of 8, copied.
    const function logitsOf(e: apiTypes.Engine, ids: { number }): buffer
        const sess = e:session()
        local at = 1
        local out: buffer? = nil
        while at <= #ids do
            const count = math.min(8, #ids - at + 1)
            out = graph.forward(sess, ids, at, count, at + count > #ids, nil, nil)
            at += count
        end
        const vocab = e.core.config.vocab
        const copy = buffer.create(vocab * 4)
        buffer.copy(copy, 0, out :: buffer, 0, vocab * 4)
        return copy
    end

    h.case("a model on the Q8 cache gives the f32 cache's logits within the rounding", function()
        const f32, q8 = open(), open("q8")
        const ids = f32:tokenize("the quick brown fox jumps over the lazy dog, what is a hash table? hello there")
        const a, b = logitsOf(f32, ids), logitsOf(q8, ids)
        const vocab = f32.core.config.vocab
        local sum, size = 0, 0
        for i = 0, vocab - 1 do
            const x = buffer.readf32(a, i * 4)
            sum += (x - buffer.readf32(b, i * 4)) ^ 2
            size += x * x
        end
        const relative = math.sqrt(sum / size)
        h.truthy(relative > 0, "the Q8 cache rounds something")
        h.truthy(relative < 0.02, `the logits differ by {relative} relative to their RMS`)
    end)

    h.case("rewinding a session on the Q8 cache and running on gives the logits of a straight run", function()
        const e = open("q8")
        const ids = e:tokenize("the quick brown fox jumps over the lazy dog, what is a hash table? hello there")
        const straight = logitsOf(e, ids)
        const sess = e:session()
        graph.forward(sess, ids, 1, 8, false, nil, nil)
        graph.forward(sess, ids, 9, 5, false, nil, nil)
        sess.length = 8
        local at = 9
        local out: buffer? = nil
        while at <= #ids do
            const count = math.min(8, #ids - at + 1)
            out = graph.forward(sess, ids, at, count, at + count > #ids, nil, nil)
            at += count
        end
        for i = 0, e.core.config.vocab - 1 do
            h.equal(buffer.readf32(out :: buffer, i * 4), buffer.readf32(straight, i * 4))
        end
    end)

    h.case("a snapshot of a Q8 session carries its centres, and a session restored from it continues alike", function()
        const e = open("q8")
        const ids = e:tokenize("the quick brown fox jumps over the lazy dog, what is a hash table? hello there")
        const straight = logitsOf(e, ids)
        const first = e:session()
        graph.forward(first, ids, 1, 8, false, nil, nil)
        graph.forward(first, ids, 9, 4, false, nil, nil)
        const snap = session.snapshot(first, 8)
        const second = e:session()
        session.restore(second, snap)
        for l = 1, e.core.config.layers do
            h.equal(
                buffer.tostring((second.cache.keyCentres :: { buffer })[l]),
                buffer.tostring((first.cache.keyCentres :: { buffer })[l])
            )
        end
        local at, out = 9, nil :: buffer?
        while at <= #ids do
            const count = math.min(8, #ids - at + 1)
            out = graph.forward(second, ids, at, count, at + count > #ids, nil, nil)
            at += count
        end
        for i = 0, e.core.config.vocab - 1 do
            h.equal(buffer.readf32(out :: buffer, i * 4), buffer.readf32(straight, i * 4))
        end
        h.truthy(session.snapshotBytes(snap) > 0)
    end)

    h.case("the memory estimate counts a Q8 cache at the row size of its layout", function()
        const q8Engine = open("q8")
        const f32, q8 = open().core.config, q8Engine.core.config
        h.equal(f32.layers, q8.layers)
        const a, b = open():memory().cache, q8Engine:memory().cache
        h.truthy(b < a, "the Q8 cache is smaller")
        const cfg = f32
        const row = kvquant.rowBytes(cfg.headDim, (q8Engine.core.model :: any).kvLayout)
        h.equal(b, cfg.layers * (2 * cfg.kvHeads * 128 * (row + 4) + cfg.kvHeads * cfg.headDim * 4))
    end)

    h.case("an engine refuses an unknown cache format", function()
        h.fails(function()
            open("q4")
        end)
    end)
end
