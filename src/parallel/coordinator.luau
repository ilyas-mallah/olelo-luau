--!native
-- The main worker of tensor parallel inference. It owns the embedding table, the residual stream, the norms and the
-- token list. Each pass it sends the normalized rows of a block to every worker, adds the partial sums they return and
-- continues, two exchanges per layer and one for the logits. Protocol in docs/parallel.md.

const arch = require("@src/arch")
const budget = require("@src/runtime/budget")
const config = require("@src/model/config")
const container = require("@src/container")
const embed = require("@src/ops/embed")
const host = require("@src/host")
const norm = require("@src/ops/norm")
const split = require("./split")
const containerTypes = require("@src/container/types")
const modelTypes = require("@src/model/types")
const runtimeTypes = require("@src/runtime/types")
const types = require("./types")

export type Options = {
    --- Positions a sequence may use.
    context: number,
    --- Tokens per pass.
    chunk: number,
    budget: runtimeTypes.Budget?,
}

--- The norm weights of one layer, which the main worker applies.
type LayerNorms = {
    attn: buffer,
    mlp: buffer,
    postAttn: buffer?,
    postMlp: buffer?,
}

--- A sequence as the main worker holds it: its residual stream and the buffers a pass works in.
type Sequence = {
    length: number,
    context: number,
    id: number,
    h: buffer,
    xn: buffer,
    o: buffer,
    lastIn: buffer,
    lastNorm: buffer,
    logits: buffer,
}

--- The counters of the main worker and the workers' own.
export type Stats = {
    --- Seconds the main worker waited for workers, and the exchanges it made.
    waitSeconds: number,
    exchanges: number,
    --- Seconds of a pass outside the exchanges.
    ownSeconds: number,
    workers: { any },
}

export type Coordinator = modelTypes.Runner & {
    workers: number,
    shares: { types.Share },
    stats: (self: Coordinator) -> Stats,
}

--- The main worker's state of a sequence it handed out.
const function own(handle: modelTypes.Sequence): Sequence
    return handle :: Sequence
end

--- Starts a main worker over `transports`, workers that hold a share each of the model in `c`, loaded with the same
--- context and chunk.
const function new(c: containerTypes.Container, transports: { host.Transport }, options: Options): Coordinator
    const cfg: config.Config = c.header.config
    const plugin = arch.get(c.header.arch)
    const P = #transports
    const first = c.tensors["layers.0.q"].format
    const shares = split.split(cfg, P, first == "q8" or first == "q4")
    const hs = config.pad(cfg.hidden)
    const tokens = options.chunk
    const embedding = container.loadTensor(c, "embed")
    const finalNorm = container.loadVector(c, "final_norm")
    const layers: { LayerNorms } = {}
    for i = 0, cfg.layers - 1 do
        table.insert(layers, {
            attn = container.loadVector(c, `layers.{i}.attn_norm`),
            mlp = container.loadVector(c, `layers.{i}.mlp_norm`),
            postAttn = if plugin.postNorms then container.loadVector(c, `layers.{i}.post_attn_norm`) else nil,
            postMlp = if plugin.postNorms then container.loadVector(c, `layers.{i}.post_mlp_norm`) else nil,
        })
    end
    const plusOne = plugin.normPlusOne
    const scale = plugin.embedScale(cfg.hidden)
    const join = transports[1].join
    local nextId = 1
    local waitSeconds, exchanges = 0, 0
    -- Rows of the requests, one buffer per row count so a request carries no more than its rows.
    const sends: { [number]: buffer } = {}
    const b = options.budget
    const timer = if b ~= nil then b.host.clock else os.clock

    --- Sends one request to every worker and waits for all replies.
    const function fan(op: string, ...: any): { any }
        const t0 = timer()
        const futures = table.create<<host.Future>>(P)
        for i = 1, P do
            futures[i] = transports[i].call(op, ...)
        end
        const replies = join(futures)
        waitSeconds += timer() - t0
        exchanges += 1
        return replies
    end

    --- The `count` rows of `seq.xn` in a buffer of their own.
    const function rows(seq: Sequence, count: number): buffer
        const cached = sends[count]
        const out = if cached ~= nil then cached else buffer.create(count * hs * 4)
        sends[count] = out
        buffer.copy(out, 0, seq.xn, 0, count * hs * 4)
        return out
    end

    --- `seq.o` as the sum of the workers' partial results.
    const function sum(seq: Sequence, parts: { any }, count: number)
        buffer.copy(seq.o, 0, parts[1] :: buffer, 0, count * hs * 4)
        for i = 2, P do
            norm.add(seq.o, hs, parts[i] :: buffer, hs, cfg.hidden, count)
        end
    end

    const self: Coordinator = {
        yields = true,
        vocab = cfg.vocab,
        workers = P,
        shares = shares,
        open = function(_self: modelTypes.Runner): modelTypes.Sequence
            const id = nextId
            nextId += 1
            return {
                length = 0,
                context = options.context,
                id = id,
                h = buffer.create(tokens * hs * 4),
                xn = buffer.create(tokens * hs * 4),
                o = buffer.create(tokens * hs * 4),
                lastIn = buffer.create(hs * 4),
                lastNorm = buffer.create(hs * 4),
                logits = buffer.create(config.roundUp(cfg.vocab, 8) * 4),
            }
        end,
        forward = function(
            _self: modelTypes.Runner,
            handle: modelTypes.Sequence,
            ids: { number },
            firstId: number,
            count: number,
            wantLogits: boolean,
            hook: modelTypes.Hook?
        ): buffer?
            const seq = own(handle)
            assert(count >= 1 and count <= tokens, "a forward pass takes between 1 and the chunk tokens")
            const pos0 = seq.length
            assert(pos0 + count <= seq.context, `the context of {seq.context} positions is full`)
            for t = 0, count - 1 do
                embed.row(embedding, ids[firstId + t], seq.h, t * hs * 4)
            end
            if scale ~= 1 then
                for t = 0, count - 1 do
                    for i = 0, cfg.hidden - 1 do
                        const at = (t * hs + i) * 4
                        buffer.writef32(seq.h, at, buffer.readf32(seq.h, at) * scale)
                    end
                end
            end
            for l = 1, cfg.layers do
                const w = layers[l]
                norm.rms(seq.h, hs, w.attn, seq.xn, hs, cfg.hidden, count, cfg.rmsEps, plusOne)
                sum(seq, fan("attn", seq.id, l, pos0, count, rows(seq, count)), count)
                const postAttn = w.postAttn
                if postAttn ~= nil then
                    norm.rms(seq.o, hs, postAttn, seq.o, hs, cfg.hidden, count, cfg.rmsEps, plusOne)
                end
                norm.add(seq.h, hs, seq.o, hs, cfg.hidden, count)

                norm.rms(seq.h, hs, w.mlp, seq.xn, hs, cfg.hidden, count, cfg.rmsEps, plusOne)
                sum(seq, fan("mlp", seq.id, l, count, rows(seq, count)), count)
                const postMlp = w.postMlp
                if postMlp ~= nil then
                    norm.rms(seq.o, hs, postMlp, seq.o, hs, cfg.hidden, count, cfg.rmsEps, plusOne)
                end
                norm.add(seq.h, hs, seq.o, hs, cfg.hidden, count)
                if hook ~= nil then
                    hook(l, seq.h, hs, count)
                end
                if b ~= nil then
                    budget.check(b :: runtimeTypes.Budget)
                end
            end
            seq.length = pos0 + count
            if not wantLogits then
                return nil
            end
            -- The last token's row goes through the final norm, then each worker returns its rows of the logits.
            buffer.copy(seq.lastIn, 0, seq.h, (count - 1) * hs * 4, hs * 4)
            norm.rms(seq.lastIn, hs, finalNorm, seq.lastNorm, hs, cfg.hidden, 1, cfg.rmsEps, plusOne)
            const parts = fan("logits", seq.id, seq.lastNorm)
            for i = 1, P do
                const share = shares[i]
                buffer.copy(seq.logits, share.vocabStart * 4, parts[i] :: buffer, 0, share.vocabCount * 4)
            end
            return seq.logits
        end,
        rewind = function(_self: modelTypes.Runner, handle: modelTypes.Sequence, length: number)
            const seq = own(handle)
            if length < seq.length then
                seq.length = length
                fan("trim", seq.id, length)
            end
        end,
        save = function(_self: modelTypes.Runner, handle: modelTypes.Sequence, length: number, key: number)
            assert(length >= 1 and length <= own(handle).length, "a snapshot holds positions the sequence has")
            fan("save", own(handle).id, length, key)
        end,
        restore = function(_self: modelTypes.Runner, handle: modelTypes.Sequence, key: number): number
            const seq = own(handle)
            assert(seq.length == 0, "a snapshot goes into an empty sequence")
            fan("restore", seq.id, key)
            -- Every worker holds the same number of positions under a key, the first one says how many.
            const length = fan("length", key)[1] :: number
            seq.length = length
            return length
        end,
        drop = function(_self: modelTypes.Runner, key: number)
            fan("drop", key)
        end,
        close = function(_self: modelTypes.Runner, handle: modelTypes.Sequence)
            fan("free", own(handle).id)
        end,
        shutdown = function(_self: modelTypes.Runner)
            for i = 1, P do
                transports[i].close()
            end
        end,
        stats = function(_self: Coordinator): Stats
            const futures = table.create<<host.Future>>(P)
            for i = 1, P do
                futures[i] = transports[i].call("stats")
            end
            return { waitSeconds = waitSeconds, exchanges = exchanges, ownSeconds = 0, workers = join(futures) }
        end,
    }
    return self
end

return {
    new = new,
}
