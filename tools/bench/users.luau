-- Throughput and fairness of the scheduler under more conversations than lanes.
--
--   lute run tools/bench/users.luau <model.olelo> [users=8] [lanes=4] [slots=1] [tokens=32] [rounds=3] [context=512]
--
-- `users` users submit a 119 token prompt at the same moment to `lanes` replica VMs that hold `slots` conversations each.
-- A round ends when every reply has ended. Prints one `USR` line with the medians over the rounds: the aggregate decode
-- rate (all new tokens over the seconds from submit to the last end), the first token time of the median and of the
-- slowest user, the end time of the first and the last user, and Jain's index of the users' end times (1 is equal).
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
    print("usage: lute run tools/bench/users.luau <model.olelo> [users] [lanes] [slots] [tokens] [rounds] [context]")
    process.exit(2)
end
const users = tonumber(process.args[3]) or 8
const laneCount = tonumber(process.args[4]) or 4
const slots = tonumber(process.args[5]) or 1
const tokens = tonumber(process.args[6]) or 32
const rounds = tonumber(process.args[7]) or 3
const context = tonumber(process.args[8]) or 512

const function median(values: { number }): number
    const copy = table.clone(values)
    table.sort(copy)
    return copy[(#copy + 1) // 2]
end

const start = time.now()
const transports = lute.spawnReplicas(path, laneCount, { context = context })
const lanes: { serveTypes.LaneSpec } = {}
for i = 1, laneCount do
    lanes[i] = { transport = transports[i], slots = slots }
end
const sched = scheduler.new({
    lanes = lanes,
    clock = function(): number
        return time.since(start)
    end,
})

const rates: { number } = {}
const firsts: { number } = {}
const slowFirsts: { number } = {}
const firstEnds: { number } = {}
const lastEnds: { number } = {}
const jains: { number } = {}
for round = 1, rounds do
    const t0 = time.since(start)
    const requests: { serveTypes.Request } = {}
    for i = 1, users do
        requests[i] = sched:submit(`r{round}u{i}`, `{PROMPT} ({i})`, { maxTokens = tokens, ignoreStop = true })
    end
    sched:run(function()
        task.wait(0.002)
    end)
    local total, sumEnd, sumSquare = 0, 0, 0
    const first: { number } = {}
    const ends: { number } = {}
    for _, req in requests do
        const r = req.reply :: serveTypes.Reply
        if r.finish ~= "length" then
            error(`a reply ended with {r.finish}: {r.error}`, 0)
        end
        total += r.newTokens
        table.insert(first, r.firstTokenSeconds)
        table.insert(ends, r.totalSeconds)
        sumEnd += r.totalSeconds
        sumSquare += r.totalSeconds * r.totalSeconds
    end
    const wall = math.max(table.unpack(ends))
    table.insert(rates, total / wall)
    table.insert(firsts, median(first))
    table.insert(slowFirsts, math.max(table.unpack(first)))
    table.insert(firstEnds, math.min(table.unpack(ends)))
    table.insert(lastEnds, wall)
    table.insert(jains, sumEnd * sumEnd / (users * sumSquare))
    for i = 1, users do
        sched:close(`r{round}u{i}`)
    end
    print(`round {round} {time.since(start) - t0} s`)
end
print(
    string.format(
        "USR users=%d lanes=%d slots=%d tokens=%d agg_tps=%.2f first_token_median_s=%.2f first_token_max_s=%.2f end_first_s=%.2f end_last_s=%.2f jain_end=%.3f",
        users,
        laneCount,
        slots,
        tokens,
        median(rates),
        median(firsts),
        median(slowFirsts),
        median(firstEnds),
        median(lastEnds),
        median(jains)
    )
)
