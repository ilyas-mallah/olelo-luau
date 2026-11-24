--!nocheck
-- The script of a worker Actor. `workers.spawn` clones it into each Actor beside an ObjectValue named Root that points
-- at the root module of the bundle, since a clone cannot find the modules by its own place in the tree.
local worker = require(script.Parent:WaitForChild("Root").Value.roblox.worker)

local platform = {
    clock = os.clock,
    wait = task.wait,
    spawn = task.spawn,
    print = print,
    warn = warn,
}
local actor = script:GetActor()
worker.bind(actor, platform, game:GetService("HttpService"), require)
