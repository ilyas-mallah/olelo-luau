-- The scheduler: fairness, placement, eviction and rebuild, streaming, cancel, timeouts and limits. The timing cases run
-- on scripted units with a clock they advance, the content cases on real units of the tiny model.
const engineApi = require("@src/api/engine")
const harness = require("@tests/harness")
const lute = require("@backends/lute")
const scheduler = require("@src/serve/scheduler")
const transport = require("@src/parallel/transport")
const unit = require("@src/serve/unit")
const serveTypes = require("@src/serve/types")

const MODEL = "tests/fixtures/tiny-llama/model.olelo"

type Clock = { now: number }

--- A unit that spends `perToken` seconds of the shared clock for each token and `perPrefill` for each prefill step, and
--- answers with the text "t" per token. A reply has `maxTokens` tokens.
const function scripted(clock: Clock, perToken: number, perPrefill: number): any
    type Job = { kind: string, left: number, total: number, cancelled: boolean }
    const jobs: { [number]: Job? } = {}
    const open: { [number]: boolean } = {}
    return {
        open = function(_self: any, sid: number, _options: any, state: any)
            open[sid] = true
            if state ~= nil then
                jobs[sid] = { kind = "prefill", left = #state.tokens, total = #state.tokens, cancelled = false }
            end
        end,
        begin = function(_self: any, sid: number, _text: string, options: any)
            const n = if options ~= nil and options.maxTokens ~= nil then options.maxTokens else 4
            jobs[sid] = { kind = "decode", left = n + 1, total = n, cancelled = false }
        end,
        step = function(_self: any, sid: number, budget: number?): serveTypes.StepOut
            const job = jobs[sid] :: Job
            if job.kind == "prefill" then
                clock.now += perPrefill
                job.left -= 1
                const done = job.left <= 0
                if done then
                    jobs[sid] = nil
                end
                return { text = "", done = done, kind = "prefill", tokens = 1, seconds = perPrefill }
            end
            -- The first step of a reply prefills its prompt.
            if job.left == job.total + 1 then
                clock.now += perPrefill
                job.left -= 1
                return { text = "", done = false, kind = "prefill", tokens = 1, seconds = perPrefill }
            end
            local made = 0
            for _ = 1, budget or 1 do
                if job.left <= 0 or job.cancelled then
                    break
                end
                clock.now += perToken
                job.left -= 1
                made += 1
            end
            const done = job.left <= 0 or job.cancelled
            if done then
                jobs[sid] = nil
            end
            return {
                text = string.rep("t", made),
                done = done,
                kind = "decode",
                tokens = made,
                seconds = made * perToken,
                finish = if done then (if job.cancelled then "cancelled" else "length") else nil,
                history = if done then { 1, 2, 3, 4 } else nil,
                pending = if done then true else nil,
                promptTokens = 1,
                newTokens = job.total - job.left,
                decodeRate = 1 / perToken,
            }
        end,
        cancel = function(_self: any, sid: number)
            const job = jobs[sid]
            if job ~= nil then
                (job :: Job).cancelled = true
            end
        end,
        close = function(_self: any, sid: number)
            open[sid] = nil
            jobs[sid] = nil
        end,
        info = function(_self: any): any
            return {}
        end,
    }
end

const function lane(u: any, slots: number?): serveTypes.LaneSpec
    return { transport = transport.inProcess(u), slots = slots }
end

--- Pumps until nothing is left, at most `limit` rounds.
const function drain(s: serveTypes.Scheduler, limit: number?)
    local rounds = 0
    while s:pump() do
        rounds += 1
        if rounds > (limit or 100000) then
            error("the scheduler did not finish", 2)
        end
    end
end

const function realUnit(kv: string?): any
    const e = engineApi.new({
        host = lute.host(),
        source = lute.fileSource(MODEL),
        context = 192,
        chunk = 8,
        kv = kv,
    })
    return unit.new(e)
end

--- The reply the plain chat API gives for the same messages.
const function direct(texts: { string }, maxTokens: number): { string }
    const e = engineApi.new({ host = lute.host(), source = lute.fileSource(MODEL), context = 192, chunk = 8 })
    const chat = e:chat()
    const out: { string } = {}
    for _, text in texts do
        local reply = ""
        for piece in chat:send(text, { maxTokens = maxTokens }) do
            reply ..= piece
        end
        table.insert(out, reply)
    end
    return out
end

return function(h: harness.Harness)
    h.case("a reply through the scheduler equals the chat API's, and streams in pieces", function()
        const clock = { now = 0 }
        const s = scheduler.new({ lanes = { lane(realUnit()) }, clock = os.clock })
        const pieces: { string } = {}
        const req = s:submit("ann", "hello there", {
            maxTokens = 10,
            onText = function(p: string)
                table.insert(pieces, p)
            end,
        })
        h.equal(req.state, "waiting")
        drain(s)
        h.equal(req.state, "done")
        const reply = req.reply :: serveTypes.Reply
        h.equal(reply.finish, "length")
        h.equal(reply.newTokens, 10)
        h.equal(reply.text, direct({ "hello there" }, 10)[1])
        h.equal(table.concat(pieces), reply.text)
        h.truthy(#pieces > 1, "the text arrives in several pieces")
        h.truthy(reply.firstTokenSeconds > 0 and reply.totalSeconds >= reply.firstTokenSeconds)
        clock.now = 0
    end)

    h.case("later turns of a user continue the conversation on its cache", function()
        const s = scheduler.new({ lanes = { lane(realUnit()) }, clock = os.clock })
        const texts = { "hello there", "what is a hash table?", "thanks" }
        const expected = direct(texts, 8)
        for i, text in texts do
            const req = s:submit("ann", text, { maxTokens = 8 })
            drain(s)
            h.equal((req.reply :: serveTypes.Reply).text, expected[i], `turn {i}`)
        end
        h.equal(#s:messages("ann"), 6)
    end)

    h.case("users on one lane take turns by the seconds they use", function()
        const clock = { now = 0 }
        -- One lane, two slots: a long reply and a short one start together.
        const s = scheduler.new({
            lanes = { lane(scripted(clock, 1, 1), 2) },
            clock = function(): number
                return clock.now
            end,
        })
        local shortDone = -1
        local longTokensWhenShortDone = 0
        const long = s:submit("long", "x", { maxTokens = 100 })
        const short = s:submit("short", "x", {
            maxTokens = 5,
            onDone = function()
                shortDone = clock.now
                longTokensWhenShortDone = s:stats().users["long"].tokens
            end,
        })
        drain(s)
        h.equal((long.reply :: serveTypes.Reply).newTokens, 100)
        h.equal((short.reply :: serveTypes.Reply).newTokens, 5)
        h.truthy(shortDone > 0 and shortDone <= 20, `the short reply ended at {shortDone}, not after the long one`)
        h.truthy(
            longTokensWhenShortDone <= 8,
            `the long reply had {longTokensWhenShortDone} tokens when the short one ended`
        )
    end)

    h.case("equal users get equal shares and a weighted user gets its share", function()
        const clock = { now = 0 }
        const s = scheduler.new({
            lanes = { lane(scripted(clock, 1, 0), 3) },
            clock = function(): number
                return clock.now
            end,
        })
        s:configure("heavy", nil, nil, 2)
        for _, id in { "a", "b", "heavy" } do
            s:submit(id, "x", { maxTokens = 400 })
        end
        -- Run until 300 seconds have passed, then compare the shares.
        while clock.now < 300 and s:pump() do
        end
        const stats = s:stats()
        const a, b, heavy = stats.users["a"].tokens, stats.users["b"].tokens, stats.users["heavy"].tokens
        h.truthy(math.abs(a - b) <= 2, `a {a} and b {b}`)
        h.truthy(math.abs(heavy - 2 * a) <= 4, `heavy {heavy} against twice {a}`)
    end)

    h.case("a user who comes back later does not take the time it was away", function()
        const clock = { now = 0 }
        const s = scheduler.new({
            lanes = { lane(scripted(clock, 1, 0), 2) },
            clock = function(): number
                return clock.now
            end,
        })
        s:submit("early", "x", { maxTokens = 50 })
        drain(s)
        const late = s:submit("late", "x", { maxTokens = 50 })
        const again = s:submit("early", "x", { maxTokens = 50 })
        drain(s)
        -- Both finish within 100 seconds of each other's start: neither starves the other.
        h.truthy(
            math.abs((late.reply :: serveTypes.Reply).totalSeconds - (again.reply :: serveTypes.Reply).totalSeconds)
                <= 3
        )
    end)

    h.case("users beyond the slots wait, and a lane with an idle conversation drops it to make room", function()
        const clock = { now = 0 }
        const u = scripted(clock, 1, 1)
        const s = scheduler.new({
            lanes = { lane(u, 1) },
            clock = function(): number
                return clock.now
            end,
        })
        const first = s:submit("one", "x", { maxTokens = 3 })
        const second = s:submit("two", "x", { maxTokens = 3 })
        s:pump()
        h.equal(second.state, "waiting")
        drain(s)
        h.equal(first.state, "done")
        h.equal(second.state, "done")
        h.equal((second.reply :: serveTypes.Reply).finish, "length")
        h.truthy((second.reply :: serveTypes.Reply).waitSeconds > 0, "the second user waited for the lane")
        -- The first user returns, its cache is rebuilt from its messages.
        const back = s:submit("one", "again", { maxTokens = 3 })
        drain(s)
        h.equal((back.reply :: serveTypes.Reply).finish, "length")
        h.equal(#s:messages("one"), 4)
        h.equal(s:stats().lanes[1].resident, 1)
    end)

    h.case("a dropped conversation is rebuilt and continues as it would have", function()
        const s = scheduler.new({ lanes = { lane(realUnit(), 1) }, clock = os.clock })
        const expected = direct({ "hello there", "what is a hash table?" }, 8)
        const a = s:submit("ann", "hello there", { maxTokens = 8 })
        drain(s)
        -- Another user takes the only slot.
        s:submit("bob", "hello", { maxTokens = 4 })
        drain(s)
        const again = s:submit("ann", "what is a hash table?", { maxTokens = 8 })
        drain(s)
        h.equal((a.reply :: serveTypes.Reply).text, expected[1])
        h.equal((again.reply :: serveTypes.Reply).text, expected[2])
    end)

    h.case("new users go to the least loaded lane", function()
        const clock = { now = 0 }
        const s = scheduler.new({
            lanes = { lane(scripted(clock, 1, 0), 3), lane(scripted(clock, 1, 0), 3) },
            clock = function(): number
                return clock.now
            end,
        })
        for i = 1, 4 do
            s:submit(`u{i}`, "x", { maxTokens = 3 })
        end
        s:pump()
        const stats = s:stats()
        h.equal(stats.lanes[1].resident, 2)
        h.equal(stats.lanes[2].resident, 2)
        drain(s)
        h.equal(s:stats().finished, 4)
    end)

    h.case("a request cancelled while waiting never reaches a lane, a running one ends at its next token", function()
        const clock = { now = 0 }
        const s = scheduler.new({
            lanes = { lane(scripted(clock, 1, 0), 1) },
            clock = function(): number
                return clock.now
            end,
        })
        const running = s:submit("one", "x", { maxTokens = 50 })
        const waiting = s:submit("two", "x", { maxTokens = 50 })
        for _ = 1, 6 do
            s:pump()
        end
        waiting:cancel()
        running:cancel()
        drain(s)
        h.equal((waiting.reply :: serveTypes.Reply).finish, "cancelled")
        h.equal((waiting.reply :: serveTypes.Reply).text, "")
        h.equal((running.reply :: serveTypes.Reply).finish, "cancelled")
        h.truthy((running.reply :: serveTypes.Reply).newTokens < 50)
        -- The cancelled reply is part of the conversation, which goes on.
        const next = s:submit("one", "y", { maxTokens = 2 })
        drain(s)
        h.equal((next.reply :: serveTypes.Reply).finish, "length")
    end)

    h.case("a request that waits too long or runs too long ends with its own finish", function()
        const clock = { now = 0 }
        const s = scheduler.new({
            lanes = { lane(scripted(clock, 1, 0), 1) },
            clock = function(): number
                return clock.now
            end,
        })
        const slow = s:submit("one", "x", { maxTokens = 500, timeout = 10 })
        const impatient = s:submit("two", "x", { maxTokens = 5, queueTimeout = 4 })
        drain(s)
        h.equal((slow.reply :: serveTypes.Reply).finish, "timeout")
        h.truthy((slow.reply :: serveTypes.Reply).newTokens >= 8 and (slow.reply :: serveTypes.Reply).newTokens <= 12)
        h.equal((impatient.reply :: serveTypes.Reply).finish, "queue_timeout")
    end)

    h.case("a user may not pile up requests and the queue is bounded", function()
        const clock = { now = 0 }
        const s = scheduler.new({
            lanes = { lane(scripted(clock, 1, 0), 1) },
            clock = function(): number
                return clock.now
            end,
            maxPerUser = 2,
            maxQueued = 3,
        })
        s:submit("one", "x", { maxTokens = 5 })
        s:submit("one", "y", { maxTokens = 5 })
        const third = s:submit("one", "z", { maxTokens = 5 })
        h.equal(third.state, "done")
        h.equal((third.reply :: serveTypes.Reply).finish, "rejected")
        s:submit("a", "x", { maxTokens = 5 })
        s:submit("b", "x", { maxTokens = 5 })
        const over = s:submit("c", "x", { maxTokens = 5 })
        h.equal((over.reply :: serveTypes.Reply).finish, "rejected")
        drain(s)
        h.equal(s:stats().refused, 2)
    end)

    h.case("a unit error ends the request with the message and the user can go on", function()
        const s = scheduler.new({ lanes = { lane(realUnit()) }, clock = os.clock })
        const bad = s:submit("ann", string.rep("the quick brown fox ", 40), { maxTokens = 4 })
        drain(s)
        h.equal((bad.reply :: serveTypes.Reply).finish, "error")
        h.truthy(string.find((bad.reply :: serveTypes.Reply).error or "", "does not fit", 1, true) ~= nil)
        const good = s:submit("ann", "hello", { maxTokens = 4 })
        drain(s)
        h.equal((good.reply :: serveTypes.Reply).finish, "length")
    end)

    h.case("closing a user frees its slot for the next one", function()
        const clock = { now = 0 }
        const s = scheduler.new({
            lanes = { lane(scripted(clock, 1, 0), 1) },
            clock = function(): number
                return clock.now
            end,
        })
        const one = s:submit("one", "x", { maxTokens = 100 })
        const two = s:submit("two", "x", { maxTokens = 3 })
        for _ = 1, 4 do
            s:pump()
        end
        s:close("one")
        h.equal((one.reply :: serveTypes.Reply).finish, "closed")
        drain(s)
        h.equal((two.reply :: serveTypes.Reply).finish, "length")
        h.equal(s:stats().lanes[1].resident, 1)
    end)

    h.case("conversations on a Q8 KV cache run through the scheduler as on the chat API", function()
        const s = scheduler.new({ lanes = { lane(realUnit("q8"), 2) }, clock = os.clock })
        const e =
            engineApi.new({ host = lute.host(), source = lute.fileSource(MODEL), context = 192, chunk = 8, kv = "q8" })
        local expected = ""
        for piece in e:chat():send("hello there", { maxTokens = 8 }) do
            expected ..= piece
        end
        const a = s:submit("ann", "hello there", { maxTokens = 8 })
        const b = s:submit("bob", "hello there", { maxTokens = 8 })
        drain(s)
        h.equal((a.reply :: serveTypes.Reply).text, expected)
        h.equal((b.reply :: serveTypes.Reply).text, expected)
    end)
end
