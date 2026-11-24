-- The Roblox chat service over replica Actors of the model of Roblox (tests/roblox/mock.luau): players get the answers a
-- chat in one VM gives, share the Actors by turns, are forgotten when they leave, and are answered over a RemoteEvent.
-- The references run outside the model, since Lute's own calls yield to Lute's scheduler. It checks the wiring. It does not say how fast Roblox runs it (docs/roblox.md).
const chatService = require("@backends/roblox/chat_service")
const fs = require("@std/fs")
const harness = require("@tests/harness")
const scheduler = require("@src/serve/scheduler")
const worlds = require("@tests/roblox/world")
const serveTypes = require("@src/serve/types")

const PATH = "tests/fixtures/wide-llama/model-q8.olelo"
const SYSTEM = "You are a careful assistant who answers briefly."
const TOKENS = 8

--- What a chat in one VM answers to `text`, the reference.
const function expected(text: string, system: string?): string
    local out = ""
    for piece in worlds.reference(PATH):chat({ system = system }):send(text, { maxTokens = TOKENS }) do
        out ..= piece
    end
    return out
end

--- A service over `count` replica Actors, and the world it runs in.
const function open(w: worlds.World, data: string, count: number, slots: number, extra: { [string]: any }?): any
    const folder = worlds.chunkFolder(w, data)
    const workers: any = w.mock.require(w.root.roblox.workers)
    const transports = workers.startReplicas(w.platform, w.place, count, {
        context = worlds.CONTEXT,
        chunk = 32,
        source = { kind = "modules", folder = folder, size = #data, chunkBytes = worlds.CHUNK_BYTES },
    })
    const lanes: { serveTypes.LaneSpec } = {}
    for i, t in transports do
        lanes[i] = { transport = t, slots = slots }
    end
    const options: { [string]: any } = { lanes = lanes, system = SYSTEM, idle = 0 }
    for k, v in extra or {} do
        options[k] = v
    end
    return (w.mock.require(w.root.roblox.chat_service) :: any).new(w.platform, options)
end

--- Waits for the frames until every request of `list` has finished.
const function waitAll(w: worlds.World, list: { serveTypes.Request })
    for _ = 1, 20000 do
        local pending = false
        for _, r in list do
            if r.state ~= "done" then
                pending = true
            end
        end
        if not pending then
            return
        end
        w.mock.task.wait()
    end
    error("the requests did not finish", 0)
end

--- A table with the shape of a Roblox event: `Connect` keeps the handler, `fire` calls it.
const function signal(): any
    const handlers: { (...any) -> () } = {}
    return {
        Connect = function(_self: any, handler: (...any) -> ())
            table.insert(handlers, handler)
        end,
        fire = function(_self: any, ...: any)
            for _, handler in handlers do
                handler(...)
            end
        end,
    }
end

return function(h: harness.Harness)
    const data = fs.readFileToString(PATH)

    h.case("players on replica Actors get the answer of a chat in one VM", function()
        const w = worlds.world()
        const want = expected("what is a hash table?", SYSTEM)
        worlds.run(w, function()
            const service = open(w, data, 2, 1)
            service:start()
            const requests: { serveTypes.Request } = {}
            const texts: { [number]: { string } } = {}
            for _, id in { 11, 22, 33 } do
                texts[id] = {}
                const request: serveTypes.Request = service:ask({ UserId = id }, "what is a hash table?", {
                    maxTokens = TOKENS,
                    onText = function(piece: string)
                        table.insert(texts[id], piece)
                    end,
                })
                table.insert(requests, request)
            end
            waitAll(w, requests)
            for i, r in requests do
                h.equal((r.reply :: serveTypes.Reply).text, want, `player {i}`)
                h.equal(table.concat(texts[({ 11, 22, 33 })[i]]), want, `the pieces streamed to player {i}`)
            end
            const stats = service.scheduler:stats()
            h.equal(stats.finished, 3)
            h.truthy(stats.lanes[1].tokens > 0 and stats.lanes[2].tokens > 0, "both Actors worked")
            service:stop()
        end)
    end)

    h.case("a player's second message continues the conversation, and a player who leaves is forgotten", function()
        const w = worlds.world()
        const want = expected("hello", SYSTEM)
        worlds.run(w, function()
            const service = open(w, data, 1, 2)
            service:start()
            const player = { UserId = 7 }
            const first = service:ask(player, "hello", { maxTokens = TOKENS })
            waitAll(w, { first })
            const second = service:ask(player, "what is a hash table?", { maxTokens = TOKENS })
            waitAll(w, { second })
            h.equal(#service.scheduler:messages("7"), 4, "two user and two assistant messages")
            service:leave(player)
            h.equal(#service.scheduler:messages("7"), 0)
            local reply: any = nil
            const again = service:ask(player, "hello", {
                maxTokens = TOKENS,
                onDone = function(r: any)
                    reply = r
                end,
            })
            waitAll(w, { again })
            h.equal(reply.text, want, "a returning player starts a new conversation")
            service:stop()
        end)
    end)

    h.case("a player's own system message and the players service", function()
        const w = worlds.world()
        const want = expected("hello", "You answer in one word.")
        worlds.run(w, function()
            const service = open(w, data, 1, 2)
            service:start()
            const players = { PlayerRemoving = signal() }
            service:bindPlayers(players)
            const player = { UserId = 5 }
            service:configure(player, "You answer in one word.", 3)
            const r = service:ask(player, "hello", { maxTokens = TOKENS })
            waitAll(w, { r })
            h.equal((r.reply :: serveTypes.Reply).text, want)
            players.PlayerRemoving:fire(player)
            h.equal(#service.scheduler:messages("5"), 0, "PlayerRemoving closes the conversation")
            service:stop()
        end)
    end)

    h.case("a RemoteEvent carries the text in and the pieces and the end out", function()
        const w = worlds.world()
        const want = expected("hello", SYSTEM)
        worlds.run(w, function()
            const service = open(w, data, 1, 2)
            service:start()
            const sent: { { any } } = {}
            const remote = {
                OnServerEvent = signal(),
                FireClient = function(_self: any, player: any, kind: string, value: string)
                    table.insert(sent, { player.UserId, kind, value })
                end,
            }
            service:bindRemote(remote, { maxTokens = TOKENS })
            remote.OnServerEvent:fire({ UserId = 9 }, "")
            remote.OnServerEvent:fire({ UserId = 9 }, 12)
            h.equal(service.scheduler:stats().submitted, 0, "an empty or a non text message is ignored")
            remote.OnServerEvent:fire({ UserId = 9 }, "hello")
            for _ = 1, 20000 do
                if #sent > 0 and sent[#sent][2] == "done" then
                    break
                end
                w.mock.task.wait()
            end
            local text = ""
            for _, event in sent do
                h.equal(event[1], 9)
                if event[2] == "text" then
                    text ..= event[3] :: string
                end
            end
            h.equal(sent[#sent][2], "done")
            h.equal(sent[#sent][3], "length")
            h.equal(text, want)
            service:stop()
        end)
    end)

    h.case("a request over the limits ends as rejected and the loop survives a stop and a start", function()
        const w = worlds.world()
        worlds.run(w, function()
            const service = open(w, data, 1, 1, { maxPerUser = 1 })
            service:start()
            service:start()
            const player = { UserId = 3 }
            const first = service:ask(player, "hello", { maxTokens = TOKENS })
            const second = service:ask(player, "again", { maxTokens = TOKENS })
            waitAll(w, { first, second })
            h.equal((second.reply :: serveTypes.Reply).finish, "rejected")
            service:stop()
            service:start()
            const third = service:ask(player, "again", { maxTokens = TOKENS })
            waitAll(w, { third })
            h.equal((third.reply :: serveTypes.Reply).finish, "length")
            service:stop()
        end)
    end)

    h.case("a player key is the user id of an instance or the value as text", function()
        h.equal(chatService.keyOf({ UserId = 4 }), "4")
        h.equal(chatService.keyOf(12), "12")
        h.equal(chatService.keyOf("guest"), "guest")
    end)
end
