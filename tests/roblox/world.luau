-- The Roblox side of a test, shared by the Roblox tests: the model of Roblox with the bundled sources in its instance
-- tree, the container as chunk ModuleScripts, and the references the Actors' results are compared with.
const base64 = require("@tools/roblox/base64")
const bundle = require("@tools/roblox/bundle")
const engineModule = require("@src/api/engine")
const hostApi = require("@src/host")
const lute = require("@backends/lute")
const mockModule = require("@tests/roblox/mock")
const probe = require("@src/kernels/probe")
const apiTypes = require("@src/api/types")

const CONTEXT = 192
const CHUNK_BYTES = 65536

--- The Roblox side of a test: the model, the bundle as an instance tree under `root`, and the platform.
export type World = {
    mock: mockModule.Mock,
    root: any,
    --- The Script under the main Actor, which the main code runs as.
    mainScript: any,
    place: any,
    platform: any,
    --- Whether the main script works in the parallel phase, which the platform's `wait` and `restore` keep.
    state: { parallel: boolean },
}

--- A model of Roblox with the bundled sources in ReplicatedStorage. The bootstrap Script stays a template.
const function world(): World
    const mock = mockModule.new()
    const root = mock.new("ModuleScript")
    root.Name = "olelo"
    root.Parent = mock.game:GetService("ReplicatedStorage")
    local template: any = nil
    for _, module in bundle.modules() do
        local at = root
        for i, name in module.path do
            local child = at:FindFirstChild(name)
            if child == nil then
                const leaf = i == #module.path
                child = mock.new(
                    if not leaf
                        then "Folder"
                        elseif string.match(module.file, "%.server%.luau$") then "Script"
                        else "ModuleScript"
                )
                child.Name = name
                child.Parent = at
            end
            at = child
        end
        at.Source = module.source
        if string.match(module.file, "%.server%.luau$") then
            template = at
        end
    end
    const parent = mock.new("Folder")
    parent.Name = "Workers"
    parent.Parent = mock.game
    const main = mock.new("Actor")
    main.Name = "Main"
    main.Parent = mock.game
    const mainScript = mock.new("Script")
    mainScript.Name = "MainScript"
    mainScript.Parent = main
    const state = { parallel = false }
    return {
        mock = mock,
        root = root,
        mainScript = mainScript,
        state = state,
        place = { main = main, parent = parent, script = template, root = root, new = mock.new },
        platform = {
            clock = os.clock,
            wait = function(seconds: number?): number
                const waited = mock.task.wait(seconds)
                if state.parallel then
                    mock.task.desynchronize()
                end
                return waited
            end,
            spawn = mock.task.spawn,
            restore = function()
                if state.parallel then
                    mock.task.desynchronize()
                else
                    mock.task.synchronize()
                end
            end,
            print = function(m: string)
                table.insert(mock.printed, m)
            end,
            warn = function(m: string)
                table.insert(mock.printed, m)
            end,
        },
    }
end

--- Runs `body` as the main script of the world, under the main Actor.
const function run(w: World, body: () -> ())
    w.mock.run(body, w.mainScript)
end

--- A Folder of ModuleScripts "1", "2" and so on that return the chunks of `data` as base64.
const function chunkFolder(w: World, data: string): any
    const folder = w.mock.new("Folder")
    folder.Name = "weights"
    folder.Parent = w.root.Parent
    for i = 1, math.ceil(#data / CHUNK_BYTES) do
        const module = w.mock.new("ModuleScript")
        module.Name = tostring(i)
        module.Source = `return "{base64.encode(string.sub(data, (i - 1) * CHUNK_BYTES + 1, i * CHUNK_BYTES))}"`
        module.Parent = folder
    end
    return folder
end

--- The engine of the container at `path` in one VM, the reference.
const function reference(path: string): apiTypes.Engine
    const lutehost = lute.host()
    -- The model's scheduler is not Lute's, so this engine never yields.
    const host: hostApi.Host = {
        name = "reference",
        clock = lutehost.clock,
        yield = function() end,
        log = lutehost.log,
    }
    return engineModule.new({
        host = host,
        source = lute.fileSource(path),
        context = CONTEXT,
        caps = { vectorRead = probe.run(host.clock).vectorRead },
    })
end

--- What `engineOver` may change: the platform, and the slice of the engine.
type Extra = { platform: any?, slice: number? }

--- The engine of the Roblox backend over chunk ModuleScripts of the container `data`, with `workers` Actors.
const function engineOver(w: World, data: string, workers: number, extra: Extra?): apiTypes.Engine
    const folder = chunkFolder(w, data)
    const roblox: any = w.mock.require(w.root.roblox)
    const spec = { kind = "modules", folder = folder, size = #data, chunkBytes = CHUNK_BYTES }
    const wk: any = w.mock.require(w.root.roblox.weights)
    return roblox.load((extra and extra.platform) or w.platform, {
        source = wk.fromModules(folder, #data, CHUNK_BYTES, w.mock.require),
        sourceSpec = spec,
        workers = workers,
        place = w.place,
        context = CONTEXT,
        chunk = 32,
        slice = if extra then extra.slice else nil,
    })
end

return {
    CONTEXT = CONTEXT,
    CHUNK_BYTES = CHUNK_BYTES,
    world = world,
    run = run,
    chunkFolder = chunkFolder,
    reference = reference,
    engineOver = engineOver,
}
