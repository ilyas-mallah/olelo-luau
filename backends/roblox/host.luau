--!native
-- The Roblox backend's `Host`. The services a script reaches as globals come in through a `Platform`, so the backend
-- imports none and a test can give it a mock.

const host = require("@src/host")

--- The parts of Roblox the backend uses.
export type Platform = {
    --- Seconds, monotonic: `os.clock`.
    clock: () -> number,
    --- `task.wait` and `task.spawn`.
    wait: (seconds: number?) -> number,
    spawn: (target: any, ...any) -> thread,
    --- `print` and `warn`.
    print: (message: string) -> (),
    warn: (message: string) -> (),
    --- Called when a thread that waited for a worker's reply wakes, which a parallel reply handler does in the parallel
    --- phase. Returns the thread to the phase the main script runs in, `task.desynchronize` or `task.synchronize`.
    restore: (() -> ())?,
}

--- A host that gives the frame back with `task.wait`.
const function new(p: Platform): host.Host
    return {
        name = "roblox",
        clock = p.clock,
        yield = function()
            p.wait()
        end,
        log = function(level: string, message: string)
            if level == "warn" or level == "error" then
                p.warn(`[{level}] {message}`)
            else
                p.print(`[{level}] {message}`)
            end
        end,
    }
end

return {
    new = new,
}
