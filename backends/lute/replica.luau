--!native
-- The module a replica VM of the Lute backend runs: a whole engine and the unit that serves conversations on it.
-- `spawnReplicas` creates a VM from it, calls `load` once and then the scheduler sends the unit's requests, which are the
-- other functions. Every function is a call from the main VM, so its arguments and its result are copies.

const engine = require("@src/api/engine")
const lute = require("@backends/lute")
const probe = require("@src/kernels/probe")
const unit = require("@src/serve/unit")
const apiTypes = require("@src/api/types")
const serveTypes = require("@src/serve/types")

local held: serveTypes.Unit? = nil

--- The loaded unit, or an error when `load` has not run.
const function loaded(): serveTypes.Unit
    if held == nil then
        error("the replica has not loaded its model, call load first", 0)
    end
    return held :: serveTypes.Unit
end

return {
    --- Loads the container file at `path` into this VM. Returns the bytes of weights.
    load = function(
        path: string,
        context: number,
        chunk: number,
        vectorRead: boolean?,
        kv: string?,
        seed: number?
    ): number
        const host = lute.host()
        const caps = { vectorRead = if vectorRead ~= nil then vectorRead else probe.run(host.clock).vectorRead }
        const e = engine.new({
            host = host,
            source = lute.fileSource(path),
            context = context,
            chunk = chunk,
            caps = caps,
            kv = kv,
            seed = seed,
        })
        held = unit.new(e)
        return e:memory().weights
    end,
    open = function(sid: number, options: serveTypes.OpenOptions?, state: apiTypes.State?)
        loaded():open(sid, options, state)
    end,
    begin = function(sid: number, text: string, options: serveTypes.ReplyOptions?)
        loaded():begin(sid, text, options)
    end,
    step = function(sid: number, maxTokens: number?): serveTypes.StepOut
        return loaded():step(sid, maxTokens)
    end,
    cancel = function(sid: number)
        loaded():cancel(sid)
    end,
    close = function(sid: number)
        loaded():close(sid)
    end,
    info = function(): serveTypes.Info
        return loaded():info()
    end,
}
