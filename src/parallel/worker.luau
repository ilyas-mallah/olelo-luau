--!native
-- One worker of tensor parallel inference. It holds a share of the model's weights and the KV cache of its heads for
-- every sequence it has been asked about, and answers requests with buffers it owns. It knows nothing about the
-- conversation, the tokens or the embedding table. Protocol in docs/parallel.md.

const config = require("@src/model/config")
const container = require("@src/container")
const graph = require("@src/model/graph")
const host = require("@src/host")
const model = require("@src/model")
const session = require("@src/model/session")
const split = require("./split")
const kernelsTypes = require("@src/kernels/types")
const modelTypes = require("@src/model/types")
const types = require("./types")

type Stats = types.Stats
type Worker = types.Worker

--- The bytes the weights of a model take.
const function weightBytes(m: modelTypes.Model): number
    local total = 0
    const function add(t: { parts: { any } })
        for _, part in t.parts do
            total += buffer.len(part.buf) + (if part.scales ~= nil then buffer.len(part.scales) else 0)
        end
    end
    for _, layer in m.layers do
        const tensors = { layer.gate, layer.up, layer.down }
        for _, t in { layer.q, layer.k, layer.v, layer.o } do
            table.insert(tensors, t)
        end
        for _, t in tensors do
            add(t)
        end
    end
    add(m.head)
    return total
end

--- Loads share `index` of `workers` from the container and starts answering. `context` and `chunk` are the positions a
--- sequence may use and the tokens of a pass, as in `Options` of the engine.
const function new(
    source: host.WeightSource,
    clock: () -> number,
    index: number,
    workers: number,
    context: number,
    chunk: number,
    caps: kernelsTypes.Caps,
    kv: string?
): Worker
    const c = container.open(source)
    const cfg: config.Config = c.header.config
    -- The format of a quantized layer tells whether the head dimension has to divide into groups.
    const quantized = c.tensors["layers.0.q"].format == "q8" or c.tensors["layers.0.q"].format == "q4"
    const share = split.split(cfg, workers, quantized)[index]
    const m = model.load(c, caps, context, share, kv)
    const sequences: { [number]: modelTypes.Session } = {}
    const saved: { [number]: modelTypes.Snapshot } = {}
    const hs = config.pad(cfg.hidden)
    const stats: Stats = {
        attnSeconds = 0,
        attnCalls = 0,
        mlpSeconds = 0,
        mlpCalls = 0,
        logitsSeconds = 0,
        logitsCalls = 0,
        bytesIn = 0,
        bytesOut = 0,
        weightBytes = weightBytes(m),
        vectorRead = caps.vectorRead,
    }

    const function open(seq: number): modelTypes.Session
        local s = sequences[seq]
        if s == nil then
            s = session.new(m, chunk)
            sequences[seq] = s
        end
        return s :: modelTypes.Session
    end

    --- Copies `count` rows of `buf` into the workspace's normalized input.
    const function receive(s: modelTypes.Session, xn: buffer, count: number)
        const bytes = count * hs * 4
        assert(buffer.len(xn) == bytes, "the request does not hold the rows it announces")
        assert(count >= 1 and count <= s.ws.tokens, "a pass takes between 1 and the workspace tokens")
        buffer.copy(s.ws.xn, 0, xn, 0, bytes)
        stats.bytesIn += bytes
    end

    --- A copy of the first `count` rows of `ws.o`.
    const function reply(s: modelTypes.Session, count: number): buffer
        const out = buffer.create(count * hs * 4)
        buffer.copy(out, 0, s.ws.o, 0, count * hs * 4)
        stats.bytesOut += count * hs * 4
        return out
    end

    return {
        share = share,
        model = m,
        weightBytes = stats.weightBytes,
        attn = function(_self: Worker, seq: number, layer: number, pos0: number, count: number, xn: buffer): buffer
            const t0 = clock()
            const s = open(seq)
            if layer == 1 and pos0 ~= s.length then
                error(`the cache of sequence {seq} holds {s.length} positions and the request starts at {pos0}`, 0)
            end
            assert(pos0 + count <= s.context, `the context of {s.context} positions is full`)
            receive(s, xn, count)
            graph.attnBlock(s, layer, pos0, count, nil)
            if layer == cfg.layers then
                s.length = pos0 + count
            end
            const out = reply(s, count)
            stats.attnSeconds += clock() - t0
            stats.attnCalls += 1
            return out
        end,
        mlp = function(_self: Worker, seq: number, layer: number, count: number, xn: buffer): buffer
            const t0 = clock()
            const s = open(seq)
            receive(s, xn, count)
            graph.mlpBlock(s, layer, count, nil)
            const out = reply(s, count)
            stats.mlpSeconds += clock() - t0
            stats.mlpCalls += 1
            return out
        end,
        logits = function(_self: Worker, seq: number, xn: buffer): buffer
            const t0 = clock()
            const s = open(seq)
            assert(buffer.len(xn) == hs * 4, "the request does not hold one normalized row")
            buffer.copy(s.ws.lastNorm, 0, xn, 0, hs * 4)
            stats.bytesIn += hs * 4
            const logits = graph.headProduct(s, nil)
            const out = buffer.create(share.vocabCount * 4)
            buffer.copy(out, 0, logits, 0, share.vocabCount * 4)
            stats.bytesOut += share.vocabCount * 4
            stats.logitsSeconds += clock() - t0
            stats.logitsCalls += 1
            return out
        end,
        trim = function(_self: Worker, seq: number, length: number)
            const s = open(seq)
            if length < s.length then
                s.length = length
            end
        end,
        free = function(_self: Worker, seq: number)
            sequences[seq] = nil
        end,
        save = function(_self: Worker, seq: number, length: number, key: number)
            saved[key] = session.snapshot(open(seq), length)
        end,
        restore = function(_self: Worker, seq: number, key: number)
            const snap = saved[key]
            if snap == nil then
                error(`nothing is saved under {key}`, 0)
            end
            const s = open(seq)
            assert(s.length == 0, "a snapshot goes into an empty sequence")
            session.restore(s, snap)
        end,
        drop = function(_self: Worker, key: number)
            saved[key] = nil
        end,
        length = function(_self: Worker, key: number): number
            const snap = saved[key]
            if snap == nil then
                error(`nothing is saved under {key}`, 0)
            end
            return snap.length
        end,
        stats = function(_self: Worker): Stats
            return table.clone(stats)
        end,
    }
end

return {
    new = new,
}
