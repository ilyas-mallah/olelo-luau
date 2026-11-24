-- The Lute pool: replica VMs and groups of worker VMs behind one scheduler answer as a chat in one VM does.
const engineApi = require("@src/api/engine")
const harness = require("@tests/harness")
const layout = require("@src/serve/layout")
const lute = require("@backends/lute")
const pool = require("@backends/lute/pool")
const process = require("@std/process")
const scheduler = require("@src/serve/scheduler")
const task = require("@std/task")
const serveTypes = require("@src/serve/types")

const PATH = "tests/fixtures/wide-llama/model-q8.olelo"
const TOKENS = 8

--- What a chat in one VM answers to each of `texts`.
const function expected(texts: { string }): { string }
    const e = engineApi.new({ host = lute.host(), source = lute.fileSource(PATH), context = 192, chunk = 32 })
    const out: { string } = {}
    for i, text in texts do
        local reply = ""
        for piece in e:chat({}):send(text, { maxTokens = TOKENS }) do
            reply ..= piece
        end
        out[i] = reply
    end
    e:close()
    return out
end

--- Runs `texts` as one user each through a scheduler over the lanes of `l`, and returns the replies.
const function answers(l: serveTypes.Layout, texts: { string }): ({ string }, serveTypes.Stats)
    const p = pool.open(PATH, l, { context = 192 })
    const start = os.clock()
    const sched = scheduler.new({
        lanes = p.lanes,
        clock = function(): number
            return os.clock() - start
        end,
    })
    const requests: { serveTypes.Request } = {}
    for i, text in texts do
        requests[i] = sched:submit(`user{i}`, text, { maxTokens = TOKENS })
    end
    sched:run(function()
        task.wait(0.001)
    end)
    const out: { string } = {}
    for i, r in requests do
        const reply = r.reply :: serveTypes.Reply
        if reply.finish ~= "length" then
            error(`reply {i} ended with {reply.finish}: {reply.error}`, 0)
        end
        out[i] = reply.text
    end
    const stats = sched:stats()
    p.close()
    return out, stats
end

return function(h: harness.Harness)
    -- A lute that crashes under repeated calls between VMs (the Ref release race) must not run these.
    if process.env.OLELO_VM ~= "1" then
        h.skip("pool VMs", "OLELO_VM is not 1, which says the lute has the fix for the vm.create crash")
        return
    end
    const texts = { "what is a hash table?", "hello there", "the quick brown fox", "tell me more" }

    h.case("replica VMs behind the scheduler answer as a chat in one VM does", function()
        const want = expected(texts)
        const got, stats = answers({ lanes = 2, workers = 1, slots = 2, bytes = 0, fits = true, reason = "" }, texts)
        for i = 1, #texts do
            h.equal(got[i], want[i], `user {i}`)
        end
        h.equal(stats.finished, 4)
        h.truthy(stats.lanes[1].tokens > 0 and stats.lanes[2].tokens > 0, "both replicas worked")
    end)

    h.case("groups of worker VMs behind the scheduler answer as a chat in one VM does", function()
        const want = expected(texts)
        const got, stats = answers({ lanes = 2, workers = 2, slots = 2, bytes = 0, fits = true, reason = "" }, texts)
        for i = 1, #texts do
            h.equal(got[i], want[i], `user {i}`)
        end
        h.equal(stats.finished, 4)
        h.equal(stats.lanes[1].workers, 2)
    end)

    h.case("a layout from the rule opens a pool that serves its users", function()
        const l = layout.choose(
            { threads = 4, memoryBytes = 8 * 1073741824 },
            { weightBytes = 1000000, sessionBytes = 100000 },
            2
        )
        h.equal(l.lanes, 2)
        h.equal(l.workers, 1)
        const want = expected({ texts[1], texts[2] })
        const got = answers(l, { texts[1], texts[2] })
        h.equal(got[1], want[1])
        h.equal(got[2], want[2])
    end)
end
