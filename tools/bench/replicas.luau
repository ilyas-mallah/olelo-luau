-- Aggregate speed of a pool of replicas: N VMs that each hold the whole model, one conversation each, answering at once.
--
--   lute run tools/bench/replicas.luau <model.olelo> [replicas=1] [tokens=48] [kv=f32] [rounds=3] [context=512]
--
-- Every round N users submit the same 119 token prompt through the scheduler, one lane per replica, and the reply of
-- each is timed from its first token. Prints a `REP` line with the medians over the rounds: the sum of the users' decode
-- rates (the aggregate), the slowest and fastest user, and the sum of the prefill rates. Run it under
-- `/usr/bin/time -f %M` for the peak memory of the whole process, see docs/benchmarks.md.
const lute = require("@backends/lute")
const process = require("@std/process")
const scheduler = require("@src/serve/scheduler")
const task = require("@std/task")
const time = require("@std/time")
const serveTypes = require("@src/serve/types")

const PROMPT = "Summarize this paragraph in one sentence: Luau is a fast, small, safe, gradually typed embeddable "
    .. "scripting language derived from Lua. It is used by Roblox to build games and by many other projects as a "
    .. "scripting layer. Its virtual machine can compile hot functions to native code, which makes numeric loops "
    .. "several times faster than the interpreter, and its buffer type gives direct access to raw memory without the "
    .. "overhead of tables."

const path = process.args[2]
if path == nil then
    print("usage: lute run tools/bench/replicas.luau <model.olelo> [replicas] [tokens] [kv] [rounds] [context]")
    process.exit(2)
end
const count = tonumber(process.args[3]) or 1
const tokens = tonumber(process.args[4]) or 48
const kv = process.args[5] or "f32"
const rounds = tonumber(process.args[6]) or 3
const context = tonumber(process.args[7]) or 512

const function median(values: { number }): number
    const copy = table.clone(values)
    table.sort(copy)
    return copy[(#copy + 1) // 2]
end

const start = time.now()
const transports = lute.spawnReplicas(path, count, { context = context, kv = kv })
const loadSeconds = time.since(start)
const lanes: { serveTypes.LaneSpec } = {}
for i = 1, count do
    lanes[i] = { transport = transports[i], slots = 1 }
end
const sched = scheduler.new({
    lanes = lanes,
    clock = function(): number
        return time.since(start)
    end,
})

const sums: { number } = {}
const slowest: { number } = {}
const fastest: { number } = {}
const prefills: { number } = {}
const firsts: { number } = {}
local promptTokens = 0
for round = 1, rounds do
    const requests: { serveTypes.Request } = {}
    for i = 1, count do
        requests[i] = sched:submit(`r{round}u{i}`, PROMPT, { maxTokens = tokens, ignoreStop = true })
    end
    sched:run(function()
        task.wait(0.002)
    end)
    local sum, low, high, prefill, first = 0, math.huge, 0, 0, 0
    for _, req in requests do
        const r = req.reply :: serveTypes.Reply
        if r.finish ~= "length" then
            error(`a reply ended with {r.finish}: {r.error}`, 0)
        end
        const rate = (r.newTokens - 1) / (r.totalSeconds - r.firstTokenSeconds)
        sum += rate
        low = math.min(low, rate)
        high = math.max(high, rate)
        prefill += r.promptTokens / r.firstTokenSeconds
        first = math.max(first, r.firstTokenSeconds)
        promptTokens = r.promptTokens
    end
    for i = 1, count do
        sched:close(`r{round}u{i}`)
    end
    table.insert(sums, sum)
    table.insert(slowest, low)
    table.insert(fastest, high)
    table.insert(prefills, prefill)
    table.insert(firsts, first)
end
print(
    string.format(
        "REP replicas=%d kv=%s prompt=%d new=%d load_s=%.1f decode_tps_sum=%.2f decode_tps_min=%.2f decode_tps_max=%.2f prefill_tps_sum=%.2f first_token_max_s=%.2f",
        count,
        kv,
        promptTokens,
        tokens,
        loadSeconds,
        median(sums),
        median(slowest),
        median(fastest),
        median(prefills),
        median(firsts)
    )
)
