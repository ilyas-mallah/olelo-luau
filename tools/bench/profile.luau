-- Where a pass spends its time: the forward pass of `model/graph.luau` run with a timer around each op, on the same
-- kernels and buffers.
--
--   lute run tools/bench/profile.luau <model.olelo> [steps=32] [context=119] [batch=0]
--
-- Prefills `context` tokens in chunks of 32, then decodes `steps` greedy tokens, and prints one `PROF` line per op
-- with milliseconds per pass for the prefill chunk and for the decode pass, and the sampler. With `batch` k it also
-- times k token passes (the verification pass of speculative decoding) against k decode passes. The parts add up to
-- the pass time, `tools/bench/generate.luau` has the unprofiled rate to compare.
const embed = require("@src/ops/embed")
const engine = require("@src/api/engine")
const lute = require("@backends/lute")
const mlp = require("@src/ops/mlp")
const model = require("@src/model")
const norm = require("@src/ops/norm")
const prepare = require("@src/ops/prepare")
const process = require("@std/process")
const linear = require("@src/ops/linear")
const registry = require("@src/kernels/registry")
const rope = require("@src/ops/rope")
const sampler = require("@src/sampling/sampler")
const session = require("@src/model/session")
const config = require("@src/model/config")
const kernelsTypes = require("@src/kernels/types")
const modelTypes = require("@src/model/types")
const opsTypes = require("@src/ops/types")
const tensorTypes = require("@src/tensor/types")

const path = process.args[2]
if path == nil then
    print("usage: lute run tools/bench/profile.luau <model.olelo> [steps] [context] [batch]")
    process.exit(2)
end
const steps = tonumber(process.args[3]) or 32
const contextTokens = tonumber(process.args[4]) or 119
const batch = tonumber(process.args[5]) or 0

const clock = os.clock
type Times = { [string]: number }
local times: Times = {}
local last = 0
--- Adds the time since the last lap to `name`.
const function lap(name: string)
    const now = clock()
    times[name] = (times[name] or 0) + (now - last)
    last = clock()
end

const function prep(
    s: modelTypes.Session,
    e: kernelsTypes.Entry,
    src: buffer,
    stride: number,
    cols: number,
    t: tensorTypes.Tensor,
    tokens: number
): (buffer, buffer?)
    return prepare.run(e, s.ws.prepared :: opsTypes.Scratch, src, stride, cols, t.kpad, tokens)
end

--- `graph.forward` with a lap after each op.
const function forward(
    s: modelTypes.Session,
    ids: { number },
    first: number,
    count: number,
    wantLogits: boolean
): buffer?
    const m = s.model
    const cfg = m.config
    const dims = m.dims
    const ws = s.ws
    const hs, qs, ks, is = ws.hs, ws.qs, ws.ks, ws.is
    const hdPad = m.hdPad
    const plusOne = m.arch.normPlusOne
    const pos0 = s.length
    const entry = if count == 1 then m.decode else m.prefill
    const processed = config.roundUp(count, entry.S)
    last = clock()
    for t = 0, count - 1 do
        embed.row(m.embed :: tensorTypes.Tensor, ids[first + t], ws.h, t * hs * 4)
    end
    lap("embed")
    for l = 1, cfg.layers do
        const layer = m.layers[l]
        const wq, wk, wv, wo = layer.q, layer.k, layer.v, layer.o
        if wq == nil or wk == nil or wv == nil or wo == nil then
            error("the profile covers models without short convolution layers", 0)
        end
        norm.rms(ws.h, hs, layer.attnNorm, ws.xn, hs, cfg.hidden, count, cfg.rmsEps, plusOne)
        lap("norm")
        const xa, xas = prep(s, entry, ws.xn, hs, cfg.hidden, wq, processed)
        lap("prep")
        linear.apply(entry, wq, xa, xas, ws.q, qs, processed, nil)
        lap("q")
        linear.apply(entry, wk, xa, xas, ws.k, ks, processed, nil)
        lap("k")
        linear.apply(entry, wv, xa, xas, ws.v, ks, processed, nil)
        lap("v")
        if layer.qBias ~= nil then
            norm.addBias(ws.q, qs, layer.qBias :: buffer, dims.heads * cfg.headDim, count)
            norm.addBias(ws.k, ks, layer.kBias :: buffer, dims.kvHeads * cfg.headDim, count)
            norm.addBias(ws.v, ks, layer.vBias :: buffer, dims.kvHeads * cfg.headDim, count)
            lap("bias")
        end
        if layer.qNorm ~= nil then
            norm.rmsHeads(ws.q, qs, layer.qNorm :: buffer, dims.heads, cfg.headDim, count, cfg.rmsEps, plusOne)
            norm.rmsHeads(ws.k, ks, layer.kNorm :: buffer, dims.kvHeads, cfg.headDim, count, cfg.rmsEps, plusOne)
            lap("qknorm")
        end
        const window = m.arch.window(l, cfg)
        const ropeTable = if window ~= nil and m.ropeLocal ~= nil then m.ropeLocal :: opsTypes.Table else m.rope
        rope.apply(
            ropeTable,
            ws.q,
            qs,
            dims.heads,
            cfg.headDim,
            count,
            pos0,
            ws.qr,
            0,
            dims.heads * hdPad * 4,
            hdPad * 4
        )
        const stride = s.context * hdPad * 4
        rope.apply(
            ropeTable,
            ws.k,
            ks,
            dims.kvHeads,
            cfg.headDim,
            count,
            pos0,
            s.cache.keys[l],
            pos0 * hdPad * 4,
            hdPad * 4,
            stride
        )
        rope.copy(ws.v, ks, dims.kvHeads, cfg.headDim, count, s.cache.values[l], pos0 * hdPad * 4, hdPad * 4, stride)
        lap("rope")
        const shape = ws.shape
        shape.window = window
        m.attend(shape, ws.qr, count, s.cache.keys[l], s.cache.values[l], pos0, ws.attn, qs, ws.scores, ws.acc)
        lap("attention")
        const oa, oas = prep(s, entry, ws.attn, qs, wo.cols, wo, processed)
        lap("prep")
        linear.apply(entry, wo, oa, oas, ws.o, hs, processed, nil)
        lap("o")
        norm.add(ws.h, hs, ws.o, hs, cfg.hidden, count)
        lap("add")

        norm.rms(ws.h, hs, layer.mlpNorm, ws.xn, hs, cfg.hidden, count, cfg.rmsEps, plusOne)
        lap("norm")
        const ga, gas = prep(s, entry, ws.xn, hs, cfg.hidden, layer.gate, processed)
        lap("prep")
        linear.apply(entry, layer.gate, ga, gas, ws.gate, is, processed, nil)
        lap("gate")
        linear.apply(entry, layer.up, ga, gas, ws.up, is, processed, nil)
        lap("up")
        mlp.gated(m.arch.activation, ws.gate, is, ws.up, is, ws.act, is, dims.inter, count)
        lap("act")
        const da, das = prep(s, entry, ws.act, is, dims.inter, layer.down, processed)
        lap("prep")
        linear.apply(entry, layer.down, da, das, ws.o, hs, processed, nil)
        lap("down")
        norm.add(ws.h, hs, ws.o, hs, cfg.hidden, count)
        lap("add")
    end
    s.length += count
    if not wantLogits then
        return nil
    end
    buffer.copy(ws.lastIn, 0, ws.h, (count - 1) * hs * 4, hs * 4)
    norm.rms(ws.lastIn, hs, m.finalNorm, ws.lastNorm, hs, cfg.hidden, 1, cfg.rmsEps, plusOne)
    lap("norm")
    const ha, has = prep(s, m.headKernel, ws.lastNorm, hs, cfg.hidden, m.head, 1)
    lap("prep")
    linear.apply(m.headKernel, m.head, ha, has, ws.logits, m.head.rowsPadded, 1, nil)
    lap("head")
    return ws.logits
end

const ORDER = {
    "embed",
    "norm",
    "prep",
    "q",
    "k",
    "v",
    "bias",
    "qknorm",
    "rope",
    "attention",
    "o",
    "add",
    "gate",
    "up",
    "act",
    "down",
    "head",
    "sample",
}

--- Prints the lines of one kind of pass, `passes` passes in `times`, `total` seconds of wall time.
const function report(kind: string, passes: number, tokensPerPass: number, total: number)
    local sum = 0
    for _, name in ORDER do
        sum += times[name] or 0
    end
    for _, name in ORDER do
        const t = times[name]
        if t ~= nil then
            print(
                string.format(
                    "PROF %s op=%s ms_per_pass=%.3f share=%.1f%%",
                    kind,
                    name,
                    t / passes * 1000,
                    t / sum * 100
                )
            )
        end
    end
    print(
        string.format(
            "PROF %s total ms_per_pass=%.3f tokens_per_pass=%d parts_ms=%.3f",
            kind,
            total / passes * 1000,
            tokensPerPass,
            sum / passes * 1000
        )
    )
end

const e = engine.new({ host = lute.host(), source = lute.fileSource(path), context = 1024 })
const m = e.core.model :: modelTypes.Model
const text = "Summarize this paragraph in one sentence: Luau is a fast, small, safe, gradually typed embeddable "
    .. "scripting language derived from Lua. It is used by Roblox to build games and by many other projects as a "
    .. "scripting layer. Its virtual machine can compile hot functions to native code, which makes numeric loops "
    .. "several times faster than the interpreter, and its buffer type gives direct access to raw memory without the "
    .. "overhead of tables. "
const base = e:tokenize(text)
const ids: { number } = {}
for i = 1, contextTokens do
    ids[i] = base[(i - 1) % #base + 1]
end

const CHUNK = 32
const s = session.new(m, CHUNK)
const smp = sampler.new(1)
const vocab = m.config.vocab

-- Prefill, profiled per chunk of 32 tokens, the last chunk keeps its logits for the first decode token.
local logits: buffer? = nil
local prefillChunks, prefillTotal = 0, 0
local at = 1
table.clear(times)
local fullChunks = 0
while at <= #ids do
    const count = math.min(CHUNK, #ids - at + 1)
    const t0 = clock()
    logits = forward(s, ids, at, count, at + count > #ids)
    if count == CHUNK then
        fullChunks += 1
        prefillTotal += clock() - t0
    end
    at += count
end
prefillChunks = fullChunks
if prefillChunks > 0 then
    report("prefill32", prefillChunks, CHUNK, prefillTotal)
end

-- Decode.
table.clear(times)
const history = table.clone(ids)
local decodeTotal = 0
for _ = 1, steps do
    const t0 = clock()
    last = t0
    const id = sampler.sample(smp, logits :: buffer, vocab, history, {})
    lap("sample")
    table.insert(history, id)
    logits = forward(s, { id }, 1, 1, true)
    decodeTotal += clock() - t0
end
report("decode", steps, 1, decodeTotal)
print(string.format("PROF decode position=%d", s.length))

if batch > 0 then
    -- The verification pass of speculative decoding: k tokens in one pass with the logits of every position, against
    -- one decode pass (the pass of a single token with its logits), from the same position, no sampling.
    const reps = 6
    const start = s.length
    const hs = s.ws.hs
    const headPrefill = registry.select(m.head.format, e.caps, true)
    const normed = buffer.create(CHUNK * hs * 4)
    const logitsAll = buffer.create(CHUNK * m.head.rowsPadded * 4)
    --- The logits of the last `count` rows of the residual stream, the way a verification pass needs them.
    const function headAll(count: number)
        norm.rms(s.ws.h, hs, m.finalNorm, normed, hs, m.config.hidden, count, m.config.rmsEps, m.arch.normPlusOne)
        lap("norm")
        const processed = config.roundUp(count, headPrefill.S)
        const ha, has = prep(s, headPrefill, normed, hs, m.config.hidden, m.head, processed)
        lap("prep")
        linear.apply(headPrefill, m.head, ha, has, logitsAll, m.head.rowsPadded, processed, nil)
        lap("head")
    end
    for _, k in { 1, 2, 3, 4, 6, 8, 12, 16 } do
        if k <= math.max(batch, 1) then
            table.clear(times)
            local total = 0
            for _ = 1, reps do
                s.length = start
                const t0 = clock()
                if k == 1 then
                    forward(s, history, #history, 1, true)
                else
                    forward(s, history, #history - k + 1, k, false)
                    headAll(k)
                end
                total += clock() - t0
            end
            print(
                string.format(
                    "BATCH k=%d ms_per_pass=%.3f ms_per_token=%.3f vs_decode=%.2f",
                    k,
                    total / reps * 1000,
                    total / reps / k * 1000,
                    total / reps / k / (decodeTotal / steps)
                )
            )
        end
    end
end
