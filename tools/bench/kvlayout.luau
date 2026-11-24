--!native
-- Decode and prefill speed against context length for the f32 cache and the Q8 cache in both of its layouts, in one
-- process, so a slow stretch hits all of them alike.
--
--   lute run tools/bench/kvlayout.luau <model.olelo> [lengths=128,512,1024,2048] [decode=16] [repeats=5]
--
-- Opens the model once per variant (`f32`, `q8_i8` the scalar attention over int8 rows, `q8_v3` the vector attention
-- over the packed layout) and fills one session per variant with the same tokens, chunk by chunk, taking the variants in
-- a rotating order. At each length it times `decode` single token passes per variant, `repeats` times, alternating the
-- variants within a repeat, and the attention op alone. Prints `DEC` lines (median over the repeats) and `PRE` lines
-- (tokens per second of the chunks since the previous length) per variant and length.
const engineApi = require("@src/api/engine")
const graph = require("@src/model/graph")
const lute = require("@backends/lute")
const process = require("@std/process")
const modelTypes = require("@src/model/types")

const path = process.args[2]
if path == nil then
    print("usage: lute run tools/bench/kvlayout.luau <model.olelo> [lengths] [decode] [repeats]")
    process.exit(2)
end
const lengths: { number } = {}
for piece in string.gmatch(process.args[3] or "128,512,1024,2048", "[^,]+") do
    table.insert(lengths, tonumber(piece) :: number)
end
const decodeSteps = tonumber(process.args[4]) or 16
const repeats = tonumber(process.args[5]) or 5
const CHUNK = 32
const maxLen = lengths[#lengths]

type Variant = {
    name: string,
    session: modelTypes.Session,
    clock: () -> number,
    attend: (at: number, passes: number) -> number,
    decode: (at: number) -> number,
}

const SPECS = {
    { name = "f32", kv = "f32", layout = nil },
    { name = "q8_i8", kv = "q8", layout = "i8" },
    { name = "q8_v3", kv = "q8", layout = "v3" },
}

const ids: { number } = {}
for i = 1, maxLen + decodeSteps + CHUNK do
    ids[i] = (i * 7919 + 13) % 3000 + 100
end

const variants: { Variant } = {}
for _, spec in SPECS do
    const e = engineApi.new({
        host = lute.host(),
        source = lute.fileSource(path),
        context = maxLen + decodeSteps + CHUNK,
        chunk = CHUNK,
        kv = spec.kv,
        kvLayout = spec.layout,
    })
    const m = e.core.model
    if m == nil then
        error("the engine holds no model", 0)
    end
    const cfg = e.core.config
    const s = e:session(CHUNK)
    const ws = s.ws
    const shape = ws.shape
    shape.window = nil
    const clock = e.core.host.clock
    table.insert(variants, {
        name = spec.name,
        session = s,
        clock = clock,
        attend = function(at: number, passes: number): number
            const t0 = clock()
            for _ = 1, passes do
                for l = 1, cfg.layers do
                    m.attend(shape, ws.qr, 1, s.cache.keys[l], s.cache.values[l], at, ws.attn, ws.qs, ws.scores, ws.acc)
                end
            end
            return (clock() - t0) / passes
        end,
        decode = function(at: number): number
            const t0 = clock()
            for _ = 1, decodeSteps do
                graph.forward(s, ids, at + 1, 1, true, nil, nil)
                s.length = at
            end
            return (clock() - t0) / decodeSteps
        end,
    })
end
const n = #variants

const function median(values: { number }): number
    const copy = table.clone(values)
    table.sort(copy)
    return copy[(#copy + 1) // 2]
end

--- Times `measure` for every variant `repeats` times, the order rotating from repeat to repeat, and prints the medians.
const function report(label: string, at: number, measure: (v: Variant) -> number)
    const samples: { { number } } = {}
    for i = 1, n do
        samples[i] = {}
    end
    for r = 0, repeats - 1 do
        for k = 0, n - 1 do
            const i = (k + r) % n + 1
            table.insert(samples[i], measure(variants[i]))
        end
    end
    for i, v in variants do
        const seconds = median(samples[i])
        if label == "DEC" then
            print(
                string.format(
                    "DEC n=%d variant=%s decode_tps=%.2f decode_ms=%.2f",
                    at,
                    v.name,
                    1 / seconds,
                    seconds * 1000
                )
            )
        else
            print(string.format("ATT n=%d variant=%s attn_ms=%.2f", at, v.name, seconds * 1000))
        end
    end
end

for _, v in variants do
    v.session.length = 0
end
report("DEC", 0, function(v)
    v.session.length = 0
    return v.decode(0)
end)

const targets: { [number]: boolean } = {}
for _, len in lengths do
    targets[len] = true
end
const windowSeconds: { number } = table.create(n, 0)
local windowTokens = 0
local at = 1
local rotation = 0
while at <= maxLen do
    for k = 0, n - 1 do
        const i = (k + rotation) % n + 1
        const v = variants[i]
        const t0 = v.clock()
        graph.forward(v.session, ids, at, CHUNK, false, nil, nil)
        windowSeconds[i] += v.clock() - t0
    end
    rotation += 1
    windowTokens += CHUNK
    at += CHUNK
    if targets[variants[1].session.length] then
        const len = variants[1].session.length
        for i, v in variants do
            print(string.format("PRE n=%d variant=%s window_tps=%.2f", len, v.name, windowTokens / windowSeconds[i]))
            windowSeconds[i] = 0
        end
        windowTokens = 0
        report("DEC", len, function(v)
            return v.decode(len)
        end)
        report("ATT", len, function(v)
            return v.attend(len, 4)
        end)
    end
end
