--!nocheck
-- A server Script that answers one prompt with the Roblox backend. It ran in Roblox Studio 0.748, Run mode
-- (docs/roblox.md).
--
-- Layout under ReplicatedStorage after `tools/roblox/build.luau` and, for the "modules" source,
-- `tools/roblox/chunks.luau`, mapped with Rojo:
--   olelo          the bundle's root module
--   weights        the chunk ModuleScripts "1", "2", and so on ("modules" source only)
-- This Script sits in ServerScriptService. It starts again under an Actor, because `BindToMessage` works only in a
-- script that is a descendant of the Actor it binds, and the workers send their replies to that Actor.
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local ServerScriptService = game:GetService("ServerScriptService")
local HttpService = game:GetService("HttpService")

local main = script:GetActor()
if main == nil then
    local actor = Instance.new("Actor")
    actor.Name = "OleloMain"
    script:Clone().Parent = actor
    actor.Parent = ServerScriptService
    return
end

local olelo = ReplicatedStorage.olelo
local roblox = require(olelo.roblox)
local weights = require(olelo.roblox.weights)

-- Fill in the numbers info.json of the chunk folder gives. "http" reads byte ranges of URL, needs HttpService enabled
-- (Game Settings, Security) and chunks of 4 MiB or more, since the request rate is limited. A ModuleScript chunk is
-- at most 149,991 bytes.
local SOURCE = "modules"
local SIZE = 0
local CHUNK_BYTES = 65536
local URL = "http://127.0.0.1:8765/model.olelo"
local WORKERS = 0

local source, spec
if SOURCE == "http" then
    source = weights.fromHttp(HttpService, URL, SIZE, CHUNK_BYTES)
    spec = { kind = "http", url = URL, size = SIZE, chunkBytes = CHUNK_BYTES }
else
    local folder = ReplicatedStorage.weights
    source = weights.fromModules(folder, SIZE, CHUNK_BYTES, require)
    spec = { kind = "modules", folder = folder, size = SIZE, chunkBytes = CHUNK_BYTES }
end

local parent = Instance.new("Folder")
parent.Name = "Workers"
parent.Parent = ServerScriptService

-- With workers the setup runs in the serial phase and the chat in the parallel phase, where a reply costs microseconds
-- and not a frame. A thread that `task.wait` or a reply handler wakes goes back to the phase it works in.
local parallel = false
local engine = roblox.load({
    clock = os.clock,
    wait = function(seconds)
        local waited = task.wait(seconds)
        if parallel then
            task.desynchronize()
        end
        return waited
    end,
    spawn = task.spawn,
    restore = function()
        if parallel then
            task.desynchronize()
        else
            task.synchronize()
        end
    end,
    print = print,
    warn = warn,
}, {
    source = source,
    sourceSpec = spec,
    workers = WORKERS,
    place = {
        main = main,
        parent = parent,
        script = olelo.roblox.bootstrap,
        root = olelo,
        new = Instance.new,
    },
    context = 1024,
})

local chat = engine:chat({})
if WORKERS > 0 then
    parallel = true
    task.desynchronize()
end
local reply = ""
for piece in chat:send("What is a hash table?", { maxTokens = 128 }) do
    reply ..= piece
end
-- Events fire and Actors are destroyed in the serial phase.
task.synchronize()
parallel = false
print(reply)
engine:close()
