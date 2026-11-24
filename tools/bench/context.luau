--!native
-- Decode and prefill speed against context length, and where the time of a long context goes.
--
--   lute run tools/bench/context.luau <model.olelo> [lengths=128,512,1024,2048] [decode=16] [repeats=3] [kv=f32]
--
-- Fills one session with a fixed token sequence, chunk by chunk, and records the time of every prefill chunk. At each
-- length it times `decode` single token passes, rewinds, and runs the attention op alone over every layer, once whole
-- and once split into its score loop, softmax and value loop. Prints `DEC`, `PRE` and `ATT` lines with medians over the
-- repeats. With `kv=q8` the split parts are 0, since the split replicates the f32 loops. Run it pinned to CPUs 14 and
-- 15, see docs/benchmarks.md.
const engineApi = require("@src/api/engine")
const graph = require("@src/model/graph")
const lute = require("@backends/lute")
const process = require("@std/process")
const split = require("./context_split")

const path = process.args[2]
if path == nil then
    print("usage: lute run tools/bench/context.luau <model.olelo> [lengths] [decode] [repeats] [kv]")
    process.exit(2)
end
const lengths: { number } = {}
for piece in string.gmatch(process.args[3] or "128,512,1024,2048", "[^,]+") do
    table.insert(lengths, tonumber(piece) :: number)
end
const decodeSteps = tonumber(process.args[4]) or 16
const repeats = tonumber(process.args[5]) or 3
const kv = process.args[6] or "f32"
const CHUNK = 32
const maxLen = lengths[#lengths]

const e = engineApi.new({
    host = lute.host(),
    source = lute.fileSource(path),
    context = maxLen + decodeSteps + CHUNK,
    chunk = CHUNK,
    kv = kv,
})
const clock = e.core.host.clock
const m = e.core.model
if m == nil then
    error("the engine holds no model", 0)
end
const cfg = e.core.config
const s = e:session(CHUNK)
const ws = s.ws
const shape = ws.shape
shape.window = nil

const ids: { number } = {}
for i = 1, maxLen + decodeSteps + CHUNK do
    ids[i] = (i * 7919 + 13) % 3000 + 100
end

const function median(values: { number }): number
    const copy = table.clone(values)
    table.sort(copy)
    return copy[(#copy + 1) // 2]
end

--- Seconds for `decodeSteps` single token passes at the current length, which is restored.
const function timeDecode(): number
    const at = s.length
    const t0 = clock()
    for _ = 1, decodeSteps do
        graph.forward(s, ids, at + 1, 1, true, nil, nil)
        s.length = at
    end
    return (clock() - t0) / decodeSteps
end

--- Seconds of the attention op of one decode token over every layer at position `at`, whole.
const function timeAttend(at: number, passes: number): number
    const t0 = clock()
    for _ = 1, passes do
        for l = 1, cfg.layers do
            m.attend(shape, ws.qr, 1, s.cache.keys[l], s.cache.values[l], at, ws.attn, ws.qs, ws.scores, ws.acc)
        end
    end
    return (clock() - t0) / passes
end

--- The same work split in three timed parts: scores, softmax, values. A replica of the vector attention loops, which
--- is checked against `timeAttend` by the sum of its parts. Only for the f32 cache.
const function timeSplit(at: number, passes: number): (number, number, number)
    if kv ~= "f32" then
        return 0, 0, 0
    end
    return split.time(clock, shape, ws, s.cache.keys, s.cache.values, cfg.layers, at, passes)
end

--- Seconds of the attention op over a prefill chunk that ends at position `at`, every layer.
const function timeAttendChunk(at: number, passes: number): number
    const t0 = clock()
    for _ = 1, passes do
        for l = 1, cfg.layers do
            m.attend(
                shape,
                ws.qr,
                CHUNK,
                s.cache.keys[l],
                s.cache.values[l],
                at - CHUNK,
                ws.attn,
                ws.qs,
                ws.scores,
                ws.acc
            )
        end
    end
    return (clock() - t0) / passes
end

const decodeAt: { [number]: { number } } = {}
const attendAt: { [number]: { number } } = {}
const attendChunkAt: { [number]: { number } } = {}
const splitAt: { [number]: { { number } } } = {}
const chunkSeconds: { { number } } = {}
const targets: { [number]: boolean } = {}
for _, n in lengths do
    targets[n] = true
    decodeAt[n] = {}
    attendAt[n] = {}
    attendChunkAt[n] = {}
    splitAt[n] = { {}, {}, {} }
end
decodeAt[0] = {}
attendAt[0] = {}
splitAt[0] = { {}, {}, {} }

for r = 1, repeats do
    s.length = 0
    table.insert(decodeAt[0], timeDecode())
    table.insert(attendAt[0], timeAttend(0, 4))
    const a, b, c = timeSplit(0, 4)
    table.insert(splitAt[0][1], a)
    table.insert(splitAt[0][2], b)
    table.insert(splitAt[0][3], c)
    chunkSeconds[r] = {}
    local at = 1
    while at <= maxLen do
        const t0 = clock()
        graph.forward(s, ids, at, CHUNK, false, nil, nil)
        table.insert(chunkSeconds[r], clock() - t0)
        at += CHUNK
        if targets[s.length] then
            const n = s.length
            table.insert(decodeAt[n], timeDecode())
            table.insert(attendAt[n], timeAttend(n, 4))
            table.insert(attendChunkAt[n], timeAttendChunk(n, 2))
            const x, y, z = timeSplit(n, 4)
            table.insert(splitAt[n][1], x)
            table.insert(splitAt[n][2], y)
            table.insert(splitAt[n][3], z)
        end
    end
end

const kvBytesPerPosition = 2 * cfg.layers * cfg.kvHeads * (if kv == "q8" then cfg.headDim + 4 else m.hdPad * 4)
const base = median(decodeAt[0])
print(
    string.format(
        "DEC n=%d decode_tps=%.2f decode_ms=%.2f extra_ms=%.2f attn_ms=%.2f scores_ms=%.2f softmax_ms=%.2f values_ms=%.2f kv_mb=%.1f",
        0,
        1 / base,
        base * 1000,
        0,
        median(attendAt[0]) * 1000,
        median(splitAt[0][1]) * 1000,
        median(splitAt[0][2]) * 1000,
        median(splitAt[0][3]) * 1000,
        kvBytesPerPosition / 1048576
    )
)
for _, n in lengths do
    const t = median(decodeAt[n])
    print(
        string.format(
            "DEC n=%d decode_tps=%.2f decode_ms=%.2f extra_ms=%.2f attn_ms=%.2f scores_ms=%.2f softmax_ms=%.2f values_ms=%.2f kv_mb=%.1f",
            n,
            1 / t,
            t * 1000,
            (t - base) * 1000,
            median(attendAt[n]) * 1000,
            median(splitAt[n][1]) * 1000,
            median(splitAt[n][2]) * 1000,
            median(splitAt[n][3]) * 1000,
            kvBytesPerPosition * (n + 1) / 1048576
        )
    )
end
--- Median seconds of the prefill chunk that starts at token `first` (1 based), over the repeats.
const function chunkMedian(index: number): number
    const values: { number } = {}
    for r = 1, repeats do
        table.insert(values, chunkSeconds[r][index])
    end
    return median(values)
end
for _, n in lengths do
    -- The rate over the four chunks that end at n, and the time of the whole prompt of n tokens.
    local window, total = 0, 0
    const chunks = n // CHUNK
    for c = 1, chunks do
        const sec = chunkMedian(c)
        total += sec
        if c > chunks - 4 then
            window += sec
        end
    end
    print(
        string.format(
            "PRE n=%d window_tps=%.2f cumulative_tps=%.2f total_s=%.3f attn_chunk_ms=%.1f",
            n,
            4 * CHUNK / window,
            n / total,
            total,
            if attendChunkAt[n] and #attendChunkAt[n] > 0 then median(attendChunkAt[n]) * 1000 else 0
        )
    )
end
