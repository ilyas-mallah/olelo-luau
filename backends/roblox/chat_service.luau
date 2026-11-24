-- A chat service for a Roblox server: every player has a conversation, the conversations share the lanes of a scheduler
-- (`src/serve/scheduler.luau`), and a loop thread pumps it once per frame. A lane is a replica Actor or a group of worker
-- Actors (`workers.startReplicas`, `workers.start`). Unverified in Roblox Studio, so no speed here is measured
-- (docs/roblox.md).

const hostModule = require("./host")
const scheduler = require("@src/serve/scheduler")
const serveTypes = require("@src/serve/types")

--- Anything that has a `UserId`, such as a Player, or a plain id.
export type Player = any

export type Options = {
    lanes: { serveTypes.LaneSpec },
    --- The system message of players that set none.
    system: string?,
    --- Seconds the loop sleeps when no request is waiting or running, 0.1 when nil.
    idle: number?,
    --- Passed to the scheduler: `stepTokens`, `maxQueued`, `maxPerUser`, `queueTimeout`, `requestTimeout`.
    stepTokens: number?,
    maxQueued: number?,
    maxPerUser: number?,
    queueTimeout: number?,
    requestTimeout: number?,
}

export type Service = {
    scheduler: serveTypes.Scheduler,
    --- Sets a player's system message, seed and share of the lanes before the first `ask`.
    configure: (self: Service, player: Player, system: string?, seed: number?, weight: number?) -> (),
    --- Queues the reply to `text` for `player`. `onText` receives the pieces as they arrive, `onDone` the final `Reply`.
    ask: (self: Service, player: Player, text: string, options: serveTypes.RequestOptions?) -> serveTypes.Request,
    --- Ends the player's requests and forgets the conversation, for a player who left.
    leave: (self: Service, player: Player) -> (),
    --- Starts the loop thread. Does nothing when it runs.
    start: (self: Service) -> (),
    --- Ends the loop after its current frame.
    stop: (self: Service) -> (),
    --- Closes the conversations of players that leave. `players` is the Players service.
    bindPlayers: (self: Service, players: any) -> (),
    --- Answers a RemoteEvent: a client fires a text, the service fires back `("text", piece)` per piece and
    --- `("done", finish)` at the end. `defaults` are the request options of every message, such as `maxTokens`.
    bindRemote: (self: Service, remote: any, defaults: serveTypes.RequestOptions?) -> (),
}

--- The scheduler's key for a player: the user id of an instance, or the value as text.
const function keyOf(player: Player): string
    if type(player) == "table" or typeof(player) == "Instance" then
        const id = (player :: any).UserId
        if id ~= nil then
            return tostring(id)
        end
    end
    return tostring(player)
end

const function new(p: hostModule.Platform, options: Options): Service
    const sched = scheduler.new({
        lanes = options.lanes,
        clock = p.clock,
        system = options.system,
        stepTokens = options.stepTokens,
        maxQueued = options.maxQueued,
        maxPerUser = options.maxPerUser,
        queueTimeout = options.queueTimeout,
        requestTimeout = options.requestTimeout,
    })
    const idle = options.idle or 0.1
    local running = false
    local looping = false

    const service: Service = {
        scheduler = sched,
        configure = function(_self: Service, player: Player, system: string?, seed: number?, weight: number?)
            sched:configure(keyOf(player), system, seed, weight)
        end,
        ask = function(_self: Service, player: Player, text: string, o: serveTypes.RequestOptions?): serveTypes.Request
            return sched:submit(keyOf(player), text, o)
        end,
        leave = function(_self: Service, player: Player)
            sched:close(keyOf(player))
        end,
        start = function(_self: Service)
            if running then
                return
            end
            running = true
            -- One loop at a time even when stop and start follow each other within a frame.
            if looping then
                return
            end
            looping = true
            p.spawn(function()
                while running do
                    const ok, busy = pcall(sched.pump, sched)
                    if not ok then
                        p.warn(`[chat service] {tostring(busy)}`)
                    end
                    p.wait(if ok and busy then 0 else idle)
                end
                looping = false
            end)
        end,
        stop = function(_self: Service)
            running = false
        end,
        bindPlayers = function(_self: Service, players: any)
            players.PlayerRemoving:Connect(function(player: Player)
                sched:close(keyOf(player))
            end)
        end,
        bindRemote = function(self: Service, remote: any, defaults: serveTypes.RequestOptions?)
            remote.OnServerEvent:Connect(function(player: Player, text: any)
                if type(text) ~= "string" or text == "" then
                    return
                end
                const o: serveTypes.RequestOptions = if defaults then table.clone(defaults) else {}
                o.onText = function(piece: string)
                    remote:FireClient(player, "text", piece)
                end
                o.onDone = function(reply: serveTypes.Reply)
                    remote:FireClient(player, "done", reply.finish)
                end
                self:ask(player, text, o)
            end)
        end,
    }
    return service
end

return {
    new = new,
    keyOf = keyOf,
}
