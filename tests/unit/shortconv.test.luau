-- The gated short convolution of LFM2: the op against its definition, and a model with such layers keeps its history in
-- the cache, so passes of any size, a rewind and a snapshot all give the logits of one straight run.
const container = require("@src/container")
const apiTypes = require("@src/api/types")
const engineModule = require("@src/api/engine")
const graph = require("@src/model/graph")
const harness = require("@tests/harness")
const lute = require("@backends/lute")
const modelTypes = require("@src/model/types")
const plan = require("@src/model/plan")
const session = require("@src/model/session")
const shortconv = require("@src/ops/shortconv")

const function floats(values: { number }): buffer
    const b = buffer.create(#values * 4)
    for i, v in values do
        buffer.writef32(b, (i - 1) * 4, v)
    end
    return b
end

const MODEL = "tests/fixtures/tiny-lfm2/model.olelo"

return function(h: harness.Harness)
    h.case("the op writes B * x to the history and C times the filtered history", function()
        -- Two channels, three taps, two tokens at positions 0 and 1, rows of B, C and x with a stride of 8.
        const bcx = floats({ 1, 2, 10, 20, 3, 4, 0, 0, 5, 6, 30, 40, 7, 8, 0, 0 })
        const taps = floats({ 1, 2, 3, 0.5, 0.25, 1 })
        const history = buffer.create(4 * 2 * 4)
        const out = buffer.create(2 * 4 * 4)
        shortconv.run(bcx, 8, 2, taps, 3, history, 0, 2, out, 4)
        -- bx: position 0 is (1 * 3, 2 * 4), position 1 is (5 * 7, 6 * 8).
        h.equal(buffer.readf32(history, 0), 3)
        h.equal(buffer.readf32(history, 4), 8)
        h.equal(buffer.readf32(history, 8), 35)
        h.equal(buffer.readf32(history, 12), 48)
        -- Position 0 sees only its own value through the last tap, position 1 also the one before.
        h.near(buffer.readf32(out, 0), 10 * (3 * 3), 1e-5)
        h.near(buffer.readf32(out, 4), 20 * (1 * 8), 1e-5)
        h.near(buffer.readf32(out, 16), 30 * (2 * 3 + 3 * 35), 1e-5)
        h.near(buffer.readf32(out, 20), 40 * (0.25 * 8 + 1 * 48), 1e-5)
    end)

    h.case("a later pass reads the positions the history already holds", function()
        const taps = floats({ 1, 2, 3 })
        -- One channel: rows of B, C and x with a stride of 4, four tokens.
        const bcx = floats({ 1, 1, 2, 0, 1, 1, 3, 0, 1, 1, 4, 0, 1, 1, 5, 0 })
        const whole, split = buffer.create(16), buffer.create(16)
        const all, early, late = buffer.create(16), buffer.create(12), buffer.create(4)
        shortconv.run(bcx, 4, 1, taps, 3, whole, 0, 4, all, 1)
        shortconv.run(bcx, 4, 1, taps, 3, split, 0, 3, early, 1)
        const rest = buffer.create(16)
        buffer.copy(rest, 0, bcx, 48, 16)
        shortconv.run(rest, 4, 1, taps, 3, split, 3, 1, late, 1)
        for i = 0, 2 do
            h.equal(buffer.readf32(early, i * 4), buffer.readf32(all, i * 4))
        end
        h.equal(buffer.readf32(late, 0), buffer.readf32(all, 12))
        h.equal(buffer.tostring(split), buffer.tostring(whole))
    end)

    const function open(): apiTypes.Engine
        return engineModule.new({ host = lute.host(), source = lute.fileSource(MODEL), context = 128, chunk = 8 })
    end

    --- The logits of the last token of `ids` run in passes of `chunk`, copied.
    const function run(sess: modelTypes.Session, ids: { number }, from: number, chunk: number): buffer
        local at = from
        local out: buffer? = nil
        while at <= #ids do
            const count = math.min(chunk, #ids - at + 1)
            out = graph.forward(sess, ids, at, count, at + count > #ids, nil, nil)
            at += count
        end
        const copy = buffer.create(buffer.len(out :: buffer))
        buffer.copy(copy, 0, out :: buffer)
        return copy
    end

    const function same(a: buffer, b: buffer, vocab: number)
        for i = 0, vocab - 1 do
            h.equal(buffer.readf32(a, i * 4), buffer.readf32(b, i * 4))
        end
    end

    const TEXT = "the quick brown fox jumps over the lazy dog, what is a hash table? hello there"

    h.case("passes of one token and of eight give the same logits as the largest pass", function()
        const e = open()
        const ids = e:tokenize(TEXT)
        const vocab = e.core.config.vocab
        const straight = run(e:session(), ids, 1, 8)
        same(run(e:session(), ids, 1, 1), straight, vocab)
        same(run(e:session(), ids, 1, 5), straight, vocab)
    end)

    h.case("a rewound session continues like a straight run", function()
        const e = open()
        const ids = e:tokenize(TEXT)
        const straight = run(e:session(), ids, 1, 8)
        const sess = e:session()
        graph.forward(sess, ids, 1, 8, false, nil, nil)
        graph.forward(sess, ids, 9, 5, false, nil, nil)
        sess.length = 8
        same(run(sess, ids, 9, 8), straight, e.core.config.vocab)
    end)

    h.case("a session restored from a snapshot continues like a straight run", function()
        const e = open()
        const ids = e:tokenize(TEXT)
        const straight = run(e:session(), ids, 1, 8)
        const first = e:session()
        graph.forward(first, ids, 1, 8, false, nil, nil)
        graph.forward(first, ids, 9, 4, false, nil, nil)
        const snap = session.snapshot(first, 8)
        const second = e:session()
        session.restore(second, snap)
        same(run(second, ids, 9, 8), straight, e.core.config.vocab)
        h.truthy(session.snapshotBytes(snap) > 0)
    end)

    h.case(
        "the memory estimate counts a history for each convolution layer and a cache for each attention layer",
        function()
            const e = open()
            const cfg = e.core.config
            const kinds = cfg.layerTypes :: { string }
            local conv, attention = 0, 0
            for _, kind in kinds do
                if kind == "conv" then
                    conv += 1
                else
                    attention += 1
                end
            end
            h.truthy(conv > 0 and attention > 0, "the fixture has both kinds of layer")
            const bytes = session.cacheBytes(cfg, 128, "f32")
            h.equal(bytes, conv * 128 * cfg.hidden * 4 + attention * session.layerCacheBytes(cfg, 128, "f32"))
            const sess = e:session()
            local held = 0
            for l = 1, cfg.layers do
                held += buffer.len(sess.cache.keys[l]) + buffer.len(sess.cache.values[l])
                held += buffer.len((sess.cache.conv :: { buffer })[l])
            end
            h.equal(held, bytes)
            h.equal(plan.estimate(container.open(lute.fileSource(MODEL)), 128, 8, "f32").cache, bytes)
        end
    )

    h.case("workers are refused for a model with convolution layers", function()
        h.fails(function()
            engineModule.new({
                host = lute.host(),
                source = lute.fileSource(MODEL),
                context = 64,
                transports = { {} :: any },
            })
        end)
    end)
end
