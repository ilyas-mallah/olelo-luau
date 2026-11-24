--!native
-- The main side of tensor parallel inference on Roblox: one Actor per worker, each with the worker script, and a
-- `host.Transport` to each that sends "req" messages and collects the "rep" messages. A message is the only line between
-- Actors, and its arguments are copies. The round trip is microseconds when the main script runs in the parallel phase
-- and a frame when it does not (docs/roblox.md, Rules Roblox enforces).

const host = require("@src/host")
const hostModule = require("./host")
const worker = require("./worker")

--- What `spawn` needs from the place it runs in.
export type Place = {
    --- The Actor the main script runs under, which receives the replies.
    main: any,
    --- The Folder or Service the worker Actors are parented to, where scripts run.
    parent: any,
    --- The Script, as a template, that calls `worker.bind`.
    script: any,
    --- The root module of the bundle, which the worker script finds in an ObjectValue named Root.
    root: any,
    --- `Instance.new`.
    new: (className: string) -> any,
}

--- Whether every future of `list` is done.
const function allDone(list: { host.Future }): boolean
    for _, f in list do
        if not f.done then
            return false
        end
    end
    return true
end

--- Seconds a worker Actor gets to bind its topics.
const READY_SECONDS = 30

--- Waits until every Actor of `actors` has set `ready`, then a frame, since a message sent before the worker script has
--- bound its topics is dropped.
const function awaitReady(p: hostModule.Platform, actors: { any })
    const deadline = p.clock() + READY_SECONDS
    for _, actor in actors do
        while not actor:GetAttribute(worker.READY) do
            if p.clock() > deadline then
                error(`a worker Actor did not bind its messages in {READY_SECONDS} seconds`, 0)
            end
            p.wait()
        end
    end
    p.wait()
end

--- Starts `count` worker Actors and returns a transport to each. Does not send `load`. Waits for the workers to bind, so
--- it must run in a thread that can yield, under the script of `place.main`.
const function spawn(p: hostModule.Platform, place: Place, count: number): { host.Transport }
    local nextId = 0
    const pending: { [number]: host.Future } = {}
    local waiting: { thread: thread, futures: { host.Future } }? = nil
    -- The replies are handled in the parallel phase, so a main script in that phase takes them without a frame.
    place.main:BindToMessageParallel("rep", function(id: number, ok: boolean, value: any)
        const future = pending[id]
        if future == nil then
            return
        end
        pending[id] = nil
        if ok then
            future.value = value
        else
            future.failure = tostring(value)
        end
        future.done = true
        const w = waiting
        if w ~= nil and allDone(w.futures) then
            waiting = nil
            p.spawn(w.thread)
        end
    end)
    const actors = table.create<<any>>(count)
    for i = 1, count do
        const actor = place.new("Actor")
        const root = place.new("ObjectValue")
        root.Name = "Root"
        root.Value = place.root
        root.Parent = actor
        place.script:Clone().Parent = actor
        actor.Parent = place.parent
        actors[i] = actor
    end
    awaitReady(p, actors)
    const transports = table.create<<host.Transport>>(count)
    for i = 1, count do
        const actor = actors[i]
        actor:SendMessage("hello", place.main)
        transports[i] = {
            call = function(op: string, ...: any): host.Future
                nextId += 1
                const future: host.Future = { done = false, value = nil, failure = nil }
                pending[nextId] = future
                actor:SendMessage(if worker.SERIAL_OPS[op] then worker.SERIAL_TOPIC else worker.TOPIC, nextId, op, ...)
                return future
            end,
            join = function(futures: { host.Future }): { any }
                if not allDone(futures) then
                    waiting = { thread = coroutine.running(), futures = futures }
                    coroutine.yield()
                    -- A reply handler in the parallel phase woke this thread, which continues in that phase.
                    if p.restore ~= nil then
                        p.restore()
                    end
                end
                const out: { any } = {}
                for k, f in futures do
                    const failure = f.failure
                    if failure ~= nil then
                        error(failure, 0)
                    end
                    out[k] = f.value
                end
                return out
            end,
            close = function()
                actor:Destroy()
            end,
        }
    end
    return transports
end

--- `spawn`, then loads each worker's share of the container `options.source` describes (`worker.SourceSpec`). Waits, so
--- it must run in a thread that can yield.
const function start(
    p: hostModule.Platform,
    place: Place,
    count: number,
    options: { context: number, chunk: number, vectorRead: boolean?, source: any }
): { host.Transport }
    const transports = spawn(p, place, count)
    const futures = table.create<<host.Future>>(count)
    for i = 1, count do
        futures[i] =
            transports[i].call("load", i, count, options.context, options.chunk, options.vectorRead, options.source)
    end
    transports[1].join(futures)
    return transports
end

--- Starts `count` Actors that each hold the whole model and serve conversations (`worker.bind` "replica"), and returns a
--- transport to each, for the lanes of a scheduler. Waits, so it must run in a thread that can yield.
const function startReplicas(
    p: hostModule.Platform,
    place: Place,
    count: number,
    options: {
        context: number,
        chunk: number,
        vectorRead: boolean?,
        kv: string?,
        seed: number?,
        source: any,
    }
): { host.Transport }
    const transports = spawn(p, place, count)
    const futures = table.create<<host.Future>>(count)
    for i = 1, count do
        futures[i] = transports[i].call(
            "replica",
            options.context,
            options.chunk,
            options.vectorRead,
            options.kv,
            options.seed,
            options.source
        )
    end
    transports[1].join(futures)
    return transports
end

return {
    spawn = spawn,
    start = start,
    startReplicas = startReplicas,
}
