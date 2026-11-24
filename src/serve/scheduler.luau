--!native
-- Runs the conversations of many users on a pool of lanes. A lane is a unit (`unit.luau`) behind a transport: a replica
-- (the whole model in one VM) or a group of tensor parallel workers. The scheduler keeps no thread. `pump` collects what
-- the lanes finished, enforces cancels and timeouts, places waiting users and gives each idle lane its next step, and the
-- host calls it in a loop.
--
-- Fairness: every user has a virtual time that grows with the seconds of lane time its steps took. An idle lane runs the
-- step of the resident user with the least virtual time, and a waiting user takes the least loaded lane, so a user with a
-- long prompt or a long reply cannot starve the others. A conversation stays on its lane for its cache. When a lane has
-- no free slot and a user waits, the least recently used idle conversation is dropped from its lane and is rebuilt from
-- its messages when its owner comes back.

const host = require("@src/host")
const apiTypes = require("@src/api/types")
const tokenizerTypes = require("@src/tokenizer/types")
const types = require("./types")

type LaneSpec = types.LaneSpec
type Reply = types.Reply
type Request = types.Request
type RequestOptions = types.RequestOptions
type Scheduler = types.Scheduler
type Stats = types.Stats

export type Options = {
    lanes: { LaneSpec },
    --- Seconds, monotonic.
    clock: () -> number,
    --- Tokens a decode step samples, 1 when nil. More saves calls and costs fairness.
    stepTokens: number?,
    --- Requests that may wait for a slot, 64 when nil. A request over the limit is refused.
    maxQueued: number?,
    --- Requests a user may have in the scheduler, the running one included, 2 when nil.
    maxPerUser: number?,
    --- Seconds a request may wait for a lane, and seconds from submit to the end of its reply. Nil for no limit.
    queueTimeout: number?,
    requestTimeout: number?,
    --- The system message of users that set none.
    system: string?,
}

type Req = {
    id: number,
    user: string,
    text: string,
    options: RequestOptions,
    --- "wait" (no lane yet), "open", "resume", "begin", "reply" or "done".
    phase: string,
    created: number,
    startedAt: number,
    firstAt: number,
    deadline: number?,
    queueDeadline: number?,
    pieces: { string },
    stopping: string?,
    stopSent: boolean,
    resumeState: apiTypes.State?,
    public: Request,
    replyMeta: { promptTokens: number, newTokens: number, cachedTokens: number, decodeRate: number },
}

type User = {
    id: string,
    system: string?,
    seed: number?,
    weight: number,
    messages: { tokenizerTypes.Message },
    --- The state of the conversation after its last reply, to rebuild its cache from.
    state: apiTypes.State?,
    vtime: number,
    lane: number?,
    sid: number?,
    requests: { Req },
    lastUsed: number,
    tokens: number,
    seconds: number,
}

type Flight = {
    op: string,
    user: User?,
    req: Req?,
    started: number,
}

type Lane = {
    index: number,
    transport: host.Transport,
    workers: number,
    slots: number,
    residents: { [number]: User },
    resident: number,
    nextSid: number,
    future: host.Future?,
    flight: Flight?,
    --- Conversations to drop from the unit before anything else runs.
    closes: { number },
    steps: number,
    busy: number,
    tokens: number,
}

const FINISH_CANCELLED = "cancelled"

const function new(options: Options): Scheduler
    const clock = options.clock
    const lanes = table.create<<Lane>>(#options.lanes)
    for i, spec in options.lanes do
        lanes[i] = {
            index = i,
            transport = spec.transport,
            workers = spec.workers or 1,
            slots = spec.slots or 1,
            residents = {},
            resident = 0,
            nextSid = 1,
            future = nil,
            flight = nil,
            closes = {},
            steps = 0,
            busy = 0,
            tokens = 0,
        }
    end
    const users: { [string]: User } = {}
    local nextRequest = 1
    local virtualTime = 0
    local submitted, finished, refused = 0, 0, 0
    const stepTokens = options.stepTokens or 1
    const maxQueued = options.maxQueued or 64
    const maxPerUser = options.maxPerUser or 2

    const function userOf(id: string): User
        const known = users[id]
        if known ~= nil then
            return known
        end
        const fresh: User = {
            id = id,
            system = options.system,
            seed = nil,
            weight = 1,
            messages = {},
            state = nil,
            vtime = virtualTime,
            lane = nil,
            sid = nil,
            requests = {},
            lastUsed = clock(),
            tokens = 0,
            seconds = 0,
        }
        users[id] = fresh
        return fresh
    end

    --- Ends `req` with `finish`, tells its owner and moves its user on to the next request.
    const function conclude(req: Req, finish: string, errorText: string?)
        if req.phase == "done" then
            return
        end
        const user = users[req.user]
        const now = clock()
        req.phase = "done"
        const reply: Reply = {
            id = req.id,
            user = req.user,
            text = table.concat(req.pieces),
            finish = finish,
            error = errorText,
            promptTokens = req.replyMeta.promptTokens,
            newTokens = req.replyMeta.newTokens,
            cachedTokens = req.replyMeta.cachedTokens,
            firstTokenSeconds = if req.firstAt > 0 then req.firstAt - req.created else 0,
            totalSeconds = now - req.created,
            waitSeconds = if req.startedAt > 0 then req.startedAt - req.created else now - req.created,
            decodeRate = req.replyMeta.decodeRate,
        }
        req.public.state = "done"
        req.public.reply = reply
        finished += 1
        if user ~= nil then
            const at = table.find(user.requests, req)
            if at ~= nil then
                table.remove(user.requests, at)
            end
            user.lastUsed = now
        end
        const done = req.options.onDone
        if done ~= nil then
            done(reply)
        end
    end

    --- Takes the user's conversation off its lane and, when the unit holds it, drops it there too.
    const function release(user: User, unitHolds: boolean)
        const laneIndex, sid = user.lane, user.sid
        if laneIndex ~= nil and sid ~= nil then
            const lane = lanes[laneIndex]
            lane.residents[sid] = nil
            lane.resident -= 1
            if unitHolds then
                table.insert(lane.closes, sid)
            end
        end
        user.lane = nil
        user.sid = nil
    end

    --- The head request of the user, when it has started.
    const function head(user: User): Req?
        return user.requests[1]
    end

    --- The id of the user's first request, after every real id when it has none.
    const function firstId(user: User): number
        const req = user.requests[1]
        return if req ~= nil then req.id else math.huge
    end

    --- Whether the user's conversation may leave its lane: it has no request.
    const function idle(user: User): boolean
        return #user.requests == 0
    end

    --- A lane with a free slot, the least filled one, or a lane where an idle conversation can make room.
    const function chooseLane(): Lane?
        local best: Lane? = nil
        local bestLoad = math.huge
        for _, lane in lanes do
            if lane.resident < lane.slots then
                const load = lane.resident / lane.slots
                if load < bestLoad then
                    best, bestLoad = lane, load
                end
            end
        end
        if best ~= nil then
            return best
        end
        -- Drop the least recently used idle conversation of any lane.
        local victim: User? = nil
        for _, lane in lanes do
            for _, u in lane.residents do
                if idle(u) and (victim == nil or u.lastUsed < victim.lastUsed) then
                    victim = u
                end
            end
        end
        if victim ~= nil then
            const index = victim.lane
            release(victim, true)
            if index ~= nil then
                return lanes[index]
            end
        end
        return nil
    end

    const function place(user: User, req: Req)
        if user.lane ~= nil then
            req.phase = "begin"
            return
        end
        const lane = chooseLane()
        if lane == nil then
            return
        end
        const sid = lane.nextSid
        lane.nextSid += 1
        lane.residents[sid] = user
        lane.resident += 1
        user.lane = lane.index
        user.sid = sid
        req.phase = "open"
        req.resumeState = user.state
    end

    --- The first request of the user that has no lane yet, which waits to be placed.
    const function waitingHead(user: User): Req?
        const req = user.requests[1]
        if req ~= nil and req.phase == "wait" then
            return req
        end
        return nil
    end

    --- The reply options of a request, which cross the transport, without its callbacks and limits. A field that
    --- `apiTypes.SendOptions` gains is listed here too.
    const function sendOptions(req: Req): types.ReplyOptions
        const o = req.options
        return {
            temperature = o.temperature,
            topK = o.topK,
            topP = o.topP,
            minP = o.minP,
            repetitionPenalty = o.repetitionPenalty,
            repetitionWindow = o.repetitionWindow,
            frequencyPenalty = o.frequencyPenalty,
            presencePenalty = o.presencePenalty,
            maxTokens = o.maxTokens,
            stop = o.stop,
            ignoreStop = o.ignoreStop,
        }
    end

    --- Handles the reply of the call a lane had in flight.
    const function complete(lane: Lane)
        const future, flight = lane.future, lane.flight
        if future == nil or flight == nil then
            return
        end
        lane.future = nil
        lane.flight = nil
        const now = clock()
        const dt = now - flight.started
        lane.busy += dt
        lane.steps += 1
        const user, req = flight.user, flight.req
        if user ~= nil then
            user.vtime += dt / user.weight
            user.seconds += dt
        end
        if future.failure ~= nil then
            if user ~= nil then
                release(user, true)
            end
            if req ~= nil then
                conclude(req, "error", future.failure)
            end
            return
        end
        const value = future.value
        if req == nil or user == nil then
            return
        end
        const r, u = req, user
        if r.phase == "done" then
            return
        end
        if flight.op == "open" then
            r.phase = if r.resumeState ~= nil then "resume" else "begin"
        elseif flight.op == "resume" then
            if value.done then
                r.phase = "begin"
                r.resumeState = nil
            end
        elseif flight.op == "begin" then
            r.phase = "reply"
        elseif flight.op == "step" then
            lane.tokens += value.tokens
            u.tokens += value.tokens
            if value.text ~= "" then
                if r.firstAt == 0 then
                    r.firstAt = now
                end
                table.insert(r.pieces, value.text)
                const onText = r.options.onText
                if onText ~= nil then
                    onText(value.text)
                end
            end
            if value.done then
                r.replyMeta.promptTokens = value.promptTokens or 0
                r.replyMeta.newTokens = value.newTokens or 0
                r.replyMeta.cachedTokens = value.cachedTokens or 0
                r.replyMeta.decodeRate = value.decodeRate or 0
                const text = table.concat(r.pieces)
                table.insert(u.messages, { role = "user", content = r.text })
                table.insert(u.messages, { role = "assistant", content = text })
                if value.history ~= nil then
                    u.state = { tokens = value.history, pending = value.pending == true }
                end
                local finish: string = value.finish or "stop"
                const stopping = r.stopping
                if finish == FINISH_CANCELLED and stopping ~= nil then
                    finish = stopping
                end
                conclude(r, finish, nil)
            end
        end
    end

    --- Marks requests that were cancelled or ran out of time, and ends those that no lane holds yet.
    const function enforce(now: number)
        for _, user in users do
            const req = head(user)
            if req ~= nil then
                const r = req
                const queueDeadline, deadline = r.queueDeadline, r.deadline
                if r.stopping == nil then
                    if r.phase == "wait" and queueDeadline ~= nil and now >= queueDeadline then
                        r.stopping = "queue_timeout"
                    elseif deadline ~= nil and now >= deadline then
                        r.stopping = "timeout"
                    end
                end
                const why = r.stopping
                if why ~= nil and r.phase == "wait" then
                    conclude(r, why)
                end
            end
            -- Requests behind the head wait for it, but their own limits still count.
            for k = 2, #user.requests do
                const queued = user.requests[k]
                const queueDeadline = queued.queueDeadline
                if queued.stopping == nil and queueDeadline ~= nil and now >= queueDeadline then
                    conclude(queued, "queue_timeout")
                end
            end
        end
    end

    --- Starts the next call on an idle lane. Returns "issued" when a call is in flight, "handled" when it ended a
    --- request without a call, and "idle" when the lane has nothing to do.
    const function dispatch(lane: Lane): string
        const now = clock()
        if #lane.closes > 0 then
            const sid = lane.closes[1]
            table.remove(lane.closes, 1)
            lane.future = lane.transport.call("close", sid)
            lane.flight = { op = "close", user = nil, req = nil, started = now }
            return "issued"
        end
        local best: User? = nil
        local bestReq: Req? = nil
        for _, candidate in lane.residents do
            const head = head(candidate)
            if head ~= nil and head.phase ~= "wait" and head.phase ~= "done" then
                if
                    best == nil
                    or bestReq == nil
                    or candidate.vtime < best.vtime
                    or (candidate.vtime == best.vtime and head.id < bestReq.id)
                then
                    best, bestReq = candidate, head
                end
            end
        end
        if best == nil then
            return "idle"
        end
        if bestReq == nil then
            return "idle"
        end
        const user, req = best, bestReq
        const sid = user.sid
        assert(sid ~= nil, "a resident conversation has a sid")
        virtualTime = math.max(virtualTime, user.vtime)
        if req.startedAt == 0 then
            req.startedAt = now
        end
        local op: string
        const stopping = req.stopping
        if stopping ~= nil and not req.stopSent then
            req.stopSent = true
            if req.phase == "open" then
                release(user, false)
                conclude(req, stopping)
                return "handled"
            elseif req.phase == "resume" then
                -- A half rebuilt cache is of no use. Drop it, the next request rebuilds it from the messages.
                release(user, true)
                conclude(req, stopping)
                return "handled"
            elseif req.phase == "begin" then
                conclude(req, stopping)
                return "handled"
            end
            op = "cancel"
            lane.future = lane.transport.call("cancel", sid)
        elseif req.phase == "open" then
            op = "open"
            lane.future = lane.transport.call("open", sid, { system = user.system, seed = user.seed }, req.resumeState)
        elseif req.phase == "resume" then
            op = "resume"
            lane.future = lane.transport.call("step", sid, stepTokens)
        elseif req.phase == "begin" then
            op = "begin"
            lane.future = lane.transport.call("begin", sid, req.text, sendOptions(req))
        else
            op = "step"
            lane.future = lane.transport.call("step", sid, stepTokens)
        end
        lane.flight = { op = op, user = user, req = req, started = now }
        return "issued"
    end

    const function pump(_self: Scheduler): boolean
        for _, lane in lanes do
            const future = lane.future
            if future ~= nil and future.done then
                complete(lane)
            end
        end
        enforce(clock())
        -- Place the waiting users, the one with the least virtual time first.
        const waiting: { User } = {}
        for _, user in users do
            if waitingHead(user) ~= nil then
                table.insert(waiting, user)
            end
        end
        table.sort(waiting, function(a: User, b: User): boolean
            if a.vtime ~= b.vtime then
                return a.vtime < b.vtime
            end
            return firstId(a) < firstId(b)
        end)
        for _, user in waiting do
            const req = waitingHead(user)
            if req ~= nil then
                place(user, req)
            end
        end
        -- A resident user whose conversation already sits on a lane has no request in phase "wait", it is placed by its first request.
        for _, lane in lanes do
            -- Ending a request without a call may leave work for the lane, so go on until it is busy or has none.
            while lane.future == nil and dispatch(lane) == "handled" do
            end
        end
        for _, lane in lanes do
            if lane.future ~= nil then
                return true
            end
        end
        for _, user in users do
            if #user.requests > 0 then
                return true
            end
        end
        return false
    end

    return {
        configure = function(_self: Scheduler, userId: string, system: string?, seed: number?, weight: number?)
            const user = userOf(userId)
            user.system = system
            user.seed = seed
            user.weight = weight or 1
        end,
        submit = function(_self: Scheduler, userId: string, text: string, ro: RequestOptions?): Request
            const o: RequestOptions = ro or {}
            const user = userOf(userId)
            const now = clock()
            const id = nextRequest
            nextRequest += 1
            submitted += 1
            const limit = o.timeout or options.requestTimeout
            const queueLimit = o.queueTimeout or options.queueTimeout
            const public: Request = {
                id = id,
                user = userId,
                state = "waiting",
                reply = nil,
                cancel = function(_r: Request) end,
            }
            const req: Req = {
                id = id,
                user = userId,
                text = text,
                options = o,
                phase = "wait",
                created = now,
                startedAt = 0,
                firstAt = 0,
                deadline = if limit ~= nil then now + limit else nil,
                queueDeadline = if queueLimit ~= nil then now + queueLimit else nil,
                pieces = {},
                stopping = nil,
                stopSent = false,
                resumeState = nil,
                public = public,
                replyMeta = { promptTokens = 0, newTokens = 0, cachedTokens = 0, decodeRate = 0 },
            }
            public.cancel = function(_r: Request)
                if req.phase == "done" then
                    return
                end
                if req.stopping == nil then
                    req.stopping = FINISH_CANCELLED
                end
            end
            local queued = 0
            for _, u in users do
                if waitingHead(u) ~= nil then
                    queued += 1
                end
            end
            if #user.requests >= maxPerUser or queued >= maxQueued then
                refused += 1
                req.phase = "wait"
                conclude(
                    req,
                    "rejected",
                    if #user.requests >= maxPerUser then "the user has too many requests" else "the queue is full"
                )
                return public
            end
            if #user.requests == 0 then
                user.vtime = math.max(user.vtime, virtualTime)
            end
            table.insert(user.requests, req)
            return public
        end,
        close = function(_self: Scheduler, userId: string)
            const user = users[userId]
            if user == nil then
                return
            end
            for _, req in table.clone(user.requests) do
                -- The reply of a running request is cut off, its cache goes with the conversation.
                conclude(req, "closed")
            end
            -- A call in flight for this user completes against a user that is gone, which `complete` tolerates.
            release(user, true)
            users[userId] = nil
        end,
        pump = pump,
        run = function(s: Scheduler, wait: () -> ())
            while s:pump() do
                wait()
            end
        end,
        messages = function(_self: Scheduler, userId: string): { tokenizerTypes.Message }
            const user = users[userId]
            return if user ~= nil then table.clone(user.messages) else {}
        end,
        stats = function(_self: Scheduler): Stats
            const out: Stats =
                { lanes = {}, users = {}, submitted = submitted, finished = finished, refused = refused, waiting = 0 }
            for _, lane in lanes do
                table.insert(out.lanes, {
                    workers = lane.workers,
                    slots = lane.slots,
                    resident = lane.resident,
                    steps = lane.steps,
                    busySeconds = lane.busy,
                    tokens = lane.tokens,
                })
            end
            for id, user in users do
                out.users[id] = { tokens = user.tokens, seconds = user.seconds, vtime = user.vtime }
                if waitingHead(user) ~= nil then
                    out.waiting += 1
                end
            end
            return out
        end,
    }
end

return {
    new = new,
}
