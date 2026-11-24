-- Times chat generation: prefill and decode rates and time to first token on a fixed prompt.
--
--   lute run tools/bench/generate.luau <model.olelo> [tokens=64] [repeats=3] [context=1024] [workers=0]
--
-- Prints one `GEN` line with the median of the repeats. With workers it divides the layers among that many worker VMs and
-- adds the main worker's wait and the workers' time to the line. Run it pinned to two CPUs without workers, see
-- docs/benchmarks.md.
const engine = require("@src/api/engine")
const lute = require("@backends/lute")
const process = require("@std/process")

const PROMPT = "Summarize this paragraph in one sentence: Luau is a fast, small, safe, gradually typed embeddable "
    .. "scripting language derived from Lua. It is used by Roblox to build games and by many other projects as a "
    .. "scripting layer. Its virtual machine can compile hot functions to native code, which makes numeric loops "
    .. "several times faster than the interpreter, and its buffer type gives direct access to raw memory without the "
    .. "overhead of tables."

const path = process.args[2]
if path == nil then
    print("usage: lute run tools/bench/generate.luau <model.olelo> [tokens] [repeats] [context]")
    process.exit(2)
end
const tokens = tonumber(process.args[3]) or 64
const repeats = tonumber(process.args[4]) or 3
const context = tonumber(process.args[5]) or 1024
const workers = tonumber(process.args[6]) or 0

const host = lute.host()
const e = engine.new({
    host = host,
    source = lute.fileSource(path),
    context = context,
    transports = if workers > 0 then lute.spawnWorkers(path, workers, { context = context, chunk = 32 }) else nil,
})
const prefill: { number } = {}
const decode: { number } = {}
const first: { number } = {}
local promptTokens = 0
const started = host.clock()
for r = 1, repeats do
    const chat = e:chat()
    for _ in chat:send(PROMPT, { maxTokens = tokens }) do
    end
    const s = chat:stats()
    promptTokens = s.promptTokens
    prefill[r] = s.prefillRate
    decode[r] = s.decodeRate
    first[r] = s.timeToFirstToken
end
const wall = host.clock() - started
const function median(values: { number }): number
    table.sort(values)
    return values[(#values + 1) // 2]
end
print(
    string.format(
        "GEN workers=%d prompt=%d new=%d kernels=%s prefill_tps=%.2f decode_tps=%.2f first_token_s=%.3f wall_s=%.3f",
        workers,
        promptTokens,
        tokens,
        if e.caps.vectorRead then "vector" else "scalar",
        median(prefill),
        median(decode),
        median(first),
        wall
    )
)
if workers > 0 then
    const stats = (e.core.runner :: any):stats()
    local attn, mlp, logits = 0, 0, 0
    for _, w in stats.workers do
        attn = math.max(attn, w.attnSeconds)
        mlp = math.max(mlp, w.mlpSeconds)
        logits = math.max(logits, w.logitsSeconds)
    end
    print(
        string.format(
            "PAR workers=%d exchanges=%d main_wait_s=%.3f worker_attn_s=%.3f worker_mlp_s=%.3f worker_logits_s=%.3f",
            workers,
            stats.exchanges,
            stats.waitSeconds,
            attn,
            mlp,
            logits
        )
    )
end
