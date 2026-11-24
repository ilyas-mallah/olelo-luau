-- Whether running the decode steps of several conversations in one forward pass would win.
--
--   lute run tools/bench/batch.luau <model.olelo> [contexts=128,512] [sizes=1,2,4,8,32] [repeats=5] [kv=f32]
--
-- A batched decode step would run one token of each of B conversations through the weights once, with the prefill
-- tile kernels, and attend over each conversation's own cache. This times `graph.forward` of B consecutive tokens of one
-- session at a context of n positions without logits, whose attention covers about n positions per token as B separate
-- caches would, and adds B head passes, since every conversation needs its own logits. Prints one `BAT` line per context
-- and size with the projected milliseconds per token and the ratio to a single token step, which is a win below 1.
-- Medians over the repeats, the sizes interleaved by round.
const engineApi = require("@src/api/engine")
const graph = require("@src/model/graph")
const lute = require("@backends/lute")
const process = require("@std/process")

const path = process.args[2]
if path == nil then
    print("usage: lute run tools/bench/batch.luau <model.olelo> [contexts] [sizes] [repeats] [kv]")
    process.exit(2)
end
const function numbers(text: string): { number }
    const out: { number } = {}
    for piece in string.gmatch(text, "[^,]+") do
        table.insert(out, tonumber(piece) :: number)
    end
    return out
end
const contexts = numbers(process.args[3] or "128,512")
const sizes = numbers(process.args[4] or "1,2,4,8,32")
const repeats = tonumber(process.args[5]) or 5
const kv = process.args[6] or "f32"
const CHUNK = 32
const maxContext = contexts[#contexts]

const e = engineApi.new({
    host = lute.host(),
    source = lute.fileSource(path),
    context = maxContext + CHUNK + 8,
    chunk = CHUNK,
    kv = kv,
})
const clock = e.core.host.clock
const model = e.core.model
if model == nil then
    error("the engine holds no model", 0)
end
const cfg = e.core.config
const s = e:session(CHUNK)
const ws = s.ws
ws.shape.window = nil

const ids: { number } = {}
for i = 1, maxContext + CHUNK + 8 do
    ids[i] = (i * 7919 + 13) % 3000 + 100
end

const function median(values: { number }): number
    const copy = table.clone(values)
    table.sort(copy)
    return copy[(#copy + 1) // 2]
end

--- Seconds of one pass over `count` tokens after `n` positions, with or without logits, which is rewound.
const function timePass(n: number, count: number, logits: boolean, passes: number): number
    const t0 = clock()
    for _ = 1, passes do
        graph.forward(s, ids, n + 1, count, logits, nil, nil)
        s.length = n
    end
    return (clock() - t0) / passes
end

--- Seconds of the attention op of one token at position `n` over every layer.
const function timeAttend(n: number, passes: number): number
    const t0 = clock()
    for _ = 1, passes do
        for l = 1, cfg.layers do
            model.attend(ws.shape, ws.qr, 1, s.cache.keys[l], s.cache.values[l], n, ws.attn, ws.qs, ws.scores, ws.acc)
        end
    end
    return (clock() - t0) / passes
end

--- Fills the session to `n` positions.
const function fill(n: number)
    s.length = 0
    local at = 1
    while at <= n do
        const count = math.min(CHUNK, n - at + 1)
        graph.forward(s, ids, at, count, false, nil, nil)
        at += count
    end
end

for _, n in contexts do
    fill(n)
    const passes: { [number]: { number } } = {}
    for _, b in sizes do
        passes[b] = {}
    end
    const single: { number }, noLogits: { number }, attend: { number } = {}, {}, {}
    for _ = 1, repeats do
        table.insert(single, timePass(n, 1, true, 3))
        table.insert(noLogits, timePass(n, 1, false, 3))
        table.insert(attend, timeAttend(n, 3))
        for _, b in sizes do
            table.insert(passes[b], timePass(n, b, false, 2))
        end
    end
    const one, bare, attendOne = median(single), median(noLogits), median(attend)
    const head = math.max(0, one - bare)
    for _, b in sizes do
        const pass = median(passes[b])
        -- B tokens through the weights, B heads, and the attention of B caches in place of the one the pass did over
        -- its own B positions.
        const perToken = (pass + b * head) / b
        print(
            string.format(
                "BAT n=%d b=%d pass_ms=%.2f head_ms=%.2f attend1_ms=%.2f single_step_ms=%.2f per_token_ms=%.2f ratio_to_single=%.2f",
                n,
                b,
                pass * 1000,
                head * 1000,
                attendOne * 1000,
                one * 1000,
                perToken * 1000,
                perToken / one
            )
        )
    end
end
