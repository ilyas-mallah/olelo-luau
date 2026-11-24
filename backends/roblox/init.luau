-- The Roblox backend's entry: the pieces, and the call that builds an engine from a chunked container.

const engine = require("@src/api/engine")
const host = require("@src/host")
const hostModule = require("@self/host")
const probe = require("@src/kernels/probe")
const weights = require("@self/weights")
const workers = require("@self/workers")
const apiTypes = require("@src/api/types")

--- What `load` takes.
export type Options = {
    --- The container's weight source, see `weights`. The main worker reads the embedding and the norms from it.
    source: host.WeightSource,
    --- The same source as a description a worker can open itself, `worker.SourceSpec`, for `workers` above 0.
    sourceSpec: any?,
    workers: number?,
    place: workers.Place?,
    context: number?,
    chunk: number?,
    seed: number?,
    --- Seconds of work between yields to the frame, 0.012 when nil.
    slice: number?,
}

--- The default `slice`: 60 frames per second held with 12 ms of work per frame in Studio (docs/roblox.md, Slices).
const SLICE = 0.012

--- The engine for a container. With workers it starts that many Actors first, which waits.
const function load(p: hostModule.Platform, options: Options): apiTypes.Engine
    const h = hostModule.new(p)
    const context = options.context or 2048
    const chunk = options.chunk or 32
    const vectorRead = probe.run(p.clock).vectorRead
    local transports: { host.Transport }? = nil
    if (options.workers or 0) > 0 then
        transports = workers.start(p, options.place :: workers.Place, options.workers :: number, {
            context = context,
            chunk = chunk,
            vectorRead = vectorRead,
            source = options.sourceSpec,
        })
    end
    return engine.new({
        host = h,
        source = weights.paced(options.source, p.clock, p.wait, weights.LOAD_SLICE),
        context = context,
        chunk = chunk,
        seed = options.seed,
        slice = options.slice or SLICE,
        caps = { vectorRead = vectorRead },
        transports = transports,
    })
end

return {
    SLICE = SLICE,
    host = hostModule.new,
    weights = weights,
    workers = workers,
    load = load,
}
