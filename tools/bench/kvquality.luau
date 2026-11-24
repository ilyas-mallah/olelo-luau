--!native
-- The quality loss of a quantized KV cache on a real model, against the f32 cache.
--
--   lute run tools/bench/kvquality.luau <model.olelo> <variant> [steps=48] [prompts=all]
--
-- Variants: `q8` is the engine's Q8 KV cache (a scale per head and position). The `fq*` variants keep the f32 cache and
-- round its rows in place after every token, which simulates a format without implementing it: `fq<bits>h` rounds
-- symmetric to a scale per head and position, `fq<bits>g32` per group of 32 values, `fq4m` is 4 bit with a minimum and
-- a maximum per group of 32. A trailing `K` or `V` rounds only the keys or only the values. Each prompt is run twice over: the candidate follows the reference's own greedy tokens
-- (teacher forcing) for the logits error and the top 1 agreement, and runs free for the greedy continuation.
-- Prints a `KVQ` line per prompt and a `KVQ total` line.
const engineApi = require("@src/api/engine")
const graph = require("@src/model/graph")
const lute = require("@backends/lute")
const process = require("@std/process")
const session = require("@src/model/session")
const templates = require("@src/tokenizer/templates")
const tokenizer = require("@src/tokenizer")
const modelTypes = require("@src/model/types")

const path = process.args[2]
const variant = process.args[3]
if path == nil or variant == nil then
    print("usage: lute run tools/bench/kvquality.luau <model.olelo> <variant> [steps] [prompts]")
    process.exit(2)
end
const steps = tonumber(process.args[4]) or 48
const only = process.args[5]

const PARAGRAPH = "Luau is a fast, small, safe, gradually typed embeddable scripting language derived from Lua. It is "
    .. "used by Roblox to build games and by many other projects as a scripting layer. Its virtual machine can compile "
    .. "hot functions to native code, which makes numeric loops several times faster than the interpreter, and its "
    .. "buffer type gives direct access to raw memory without the overhead of tables. "

const PROMPTS: { { name: string, system: string?, text: string } } = {
    { name = "greeting", text = "Hi, who are you?" },
    { name = "hash table", text = "Explain what a hash table is in two sentences." },
    { name = "haiku", system = "You are a poet who answers in haiku.", text = "Write about the sea." },
    { name = "long3", text = "Summarize this text in one sentence: " .. string.rep(PARAGRAPH, 3) },
    { name = "long8", text = "Summarize this text in one sentence: " .. string.rep(PARAGRAPH, 8) },
}

const CONTEXT = 1280
const reference = engineApi.new({ host = lute.host(), source = lute.fileSource(path), context = CONTEXT })
const real = variant == "q8"
const candidate = if real
    then engineApi.new({ host = lute.host(), source = lute.fileSource(path), context = CONTEXT, kv = "q8" })
    else reference
const vocab = reference.core.config.vocab
const cfg = reference.core.config

--- Rounds `count` values at `at` in `b` (f32) in place: symmetric with `levels` steps each side, or min max with
--- `levels` steps in all when `asymmetric`.
const function roundRun(b: buffer, at: number, count: number, levels: number, asymmetric: boolean)
    if asymmetric then
        local lo, hi = math.huge, -math.huge
        for i = 0, count - 1 do
            const v = buffer.readf32(b, at + i * 4)
            lo = math.min(lo, v)
            hi = math.max(hi, v)
        end
        const range = hi - lo
        if range == 0 then
            return
        end
        const step = range / levels
        for i = 0, count - 1 do
            buffer.writef32(b, at + i * 4, lo + math.round((buffer.readf32(b, at + i * 4) - lo) / step) * step)
        end
    else
        local amax = 0
        for i = 0, count - 1 do
            amax = math.max(amax, math.abs(buffer.readf32(b, at + i * 4)))
        end
        if amax == 0 then
            return
        end
        const step = amax / levels
        for i = 0, count - 1 do
            buffer.writef32(b, at + i * 4, math.round(buffer.readf32(b, at + i * 4) / step) * step)
        end
    end
end

const fakeBits, fakeGroup, fakeAsymmetric, fakeCentred, fakeOnly = (
    function(): (number, number, boolean, boolean, number)
        if real then
            return 0, 0, false, false, 0
        end
        const bits, rest = string.match(variant, "^fq(%d+)(%w+)$")
        if bits == nil then
            error(`unknown variant {variant}`, 0)
        end
        local kind = rest :: string
        local only = 0
        if string.sub(kind, -1) == "K" or string.sub(kind, -1) == "V" then
            only = if string.sub(kind, -1) == "K" then 1 else 2
            kind = string.sub(kind, 1, -2)
        end
        const b = tonumber(bits) :: number
        const centred = string.sub(kind, -1) == "c"
        const shape = if centred then string.sub(kind, 1, -2) else kind
        return b, if shape == "h" then cfg.headDim else 32, shape == "m", centred, only
    end
)()

--- The mean of each channel over the first 32 positions of the reference's cache, per layer and KV head: keys then
--- values. A constant vector subtracted from every key leaves the softmax unchanged, and one subtracted from every value
--- comes back as a constant after the weighted sum, so a cache may store the rows without it.
local centres: { { { { number } } } } = { {}, {} }

const function computeCentres(s: modelTypes.Session)
    const n = math.min(32, s.length)
    const hdPad = s.model.hdPad
    for kind = 1, 2 do
        for l = 1, cfg.layers do
            centres[kind][l] = {}
            const cache = if kind == 1 then s.cache.keys[l] else s.cache.values[l]
            for g = 0, s.model.dims.kvHeads - 1 do
                const mean: { number } = {}
                for d = 0, cfg.headDim - 1 do
                    local sum = 0
                    for p = 0, n - 1 do
                        sum += buffer.readf32(cache, ((g * s.context + p) * hdPad + d) * 4)
                    end
                    mean[d + 1] = sum / n
                end
                centres[kind][l][g + 1] = mean
            end
        end
    end
end

--- Rounds the cache rows of positions `pos0` to `pos0 + count - 1` of the f32 session in the fake variants.
const function fake(s: modelTypes.Session, pos0: number, count: number)
    const hdPad = s.model.hdPad
    const levels = if fakeAsymmetric then 2 ^ fakeBits - 1 else 2 ^ (fakeBits - 1) - 1
    for l = 1, cfg.layers do
        for kind, cache in { s.cache.keys[l], s.cache.values[l] } do
            for g = 0, if fakeOnly ~= 0 and fakeOnly ~= kind then -1 else s.model.dims.kvHeads - 1 do
                for p = pos0, pos0 + count - 1 do
                    const row = ((g * s.context + p) * hdPad) * 4
                    const centre = if fakeCentred then centres[kind][l][g + 1] else nil
                    if centre ~= nil then
                        for d = 0, cfg.headDim - 1 do
                            buffer.writef32(cache, row + d * 4, buffer.readf32(cache, row + d * 4) - centre[d + 1])
                        end
                    end
                    for start = 0, cfg.headDim - 1, fakeGroup do
                        roundRun(
                            cache,
                            row + start * 4,
                            math.min(fakeGroup, cfg.headDim - start),
                            levels,
                            fakeAsymmetric
                        )
                    end
                    if centre ~= nil then
                        for d = 0, cfg.headDim - 1 do
                            buffer.writef32(cache, row + d * 4, buffer.readf32(cache, row + d * 4) + centre[d + 1])
                        end
                    end
                end
            end
        end
    end
end

const function argmax(b: buffer): number
    local best, id = -math.huge, 0
    for i = 0, vocab - 1 do
        const v = buffer.readf32(b, i * 4)
        if v > best then
            best, id = v, i
        end
    end
    return id
end

--- RMS of the difference of two logit rows over the RMS of the first.
const function relative(a: buffer, b: buffer): number
    local d, n = 0, 0
    for i = 0, vocab - 1 do
        const x = buffer.readf32(a, i * 4)
        const y = buffer.readf32(b, i * 4)
        d += (x - y) ^ 2
        n += x * x
    end
    return math.sqrt(d / n)
end

--- A pass over the prompt in chunks of `chunk`, returning the last logits.
const function prefill(s: modelTypes.Session, ids: { number }, chunk: number, quantize: boolean): buffer
    local at = 1
    local out: buffer? = nil
    while at <= #ids do
        const count = math.min(chunk, #ids - at + 1)
        const before = s.length
        out = graph.forward(s, ids, at, count, at + count > #ids, nil, nil)
        if quantize then
            fake(s, before, count)
        end
        at += count
    end
    return out :: buffer
end

--- One decoded token on `s`.
const function step(s: modelTypes.Session, id: number, quantize: boolean): buffer
    const before = s.length
    const out = graph.forward(s, { id }, 1, 1, true, nil, nil) :: buffer
    if quantize then
        fake(s, before, 1)
    end
    return out
end

local totalSteps, totalAgree, totalFree, errSum, errMax, promptsRun = 0, 0, 0, 0, 0, 0
for _, p in PROMPTS do
    if only ~= nil and only ~= "all" and not string.find(only, p.name, 1, true) then
        continue
    end
    const spec = reference.core.tokenizer.chat
    const text = templates.opening(spec, p.system, p.text) .. templates.generationPrompt(spec)
    const ids = tokenizer.encode(reference.core.tokenizer, text)
    local chunk = if real then 32 else 1
    const quantize = not real
    const refSession = reference:session(32)
    const teacher = if real then candidate:session(32) else reference:session(32)
    const free = if real then candidate:session(32) else reference:session(32)
    local refLogits = prefill(refSession, ids, 32, false)
    if fakeCentred then
        computeCentres(refSession)
    end
    local teacherLogits = prefill(teacher, ids, chunk, quantize)
    local freeLogits = prefill(free, ids, chunk, quantize)
    local errs, agree, freeAgree = 0, 0, 0
    local worst = 0
    local diverged = false
    local firstDivergence = steps
    const freeIds: { number } = {}
    for i = 1, steps do
        const e = relative(refLogits, teacherLogits)
        errs += e
        worst = math.max(worst, e)
        const refId = argmax(refLogits)
        if argmax(teacherLogits) == refId then
            agree += 1
        end
        const freeId = argmax(freeLogits)
        if freeId == refId and not diverged then
            freeAgree += 1
        elseif not diverged then
            diverged = true
            firstDivergence = i - 1
        end
        table.insert(freeIds, freeId)
        if i < steps then
            refLogits = step(refSession, refId, false)
            teacherLogits = step(teacher, refId, quantize)
            freeLogits = step(free, freeId, quantize)
        end
    end
    totalSteps += steps
    totalAgree += agree
    totalFree += firstDivergence
    errSum += errs
    errMax = math.max(errMax, worst)
    promptsRun += 1
    print(
        string.format(
            "KVQ variant=%s prompt=%s tokens=%d rel_rms_mean=%.5f rel_rms_max=%.5f top1_agree=%d/%d free_equal_prefix=%d/%d",
            variant,
            p.name,
            #ids,
            errs / steps,
            worst,
            agree,
            steps,
            firstDivergence,
            steps
        )
    )
end
print(
    string.format(
        "KVQ total variant=%s prompts=%d rel_rms_mean=%.5f rel_rms_max=%.5f top1_agree=%.4f free_prefix_fraction=%.4f",
        variant,
        promptsRun,
        errSum / totalSteps,
        errMax,
        totalAgree / totalSteps,
        totalFree / totalSteps
    )
)
