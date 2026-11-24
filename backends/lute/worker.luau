--!native
-- The module a worker VM of the Lute backend runs. `spawnWorkers` creates a VM from it, calls `load` once and then sends
-- the requests of the protocol, which are the other functions. Every function is a call from the main VM, so its
-- arguments and its result are copies.

const lute = require("@backends/lute")
const probe = require("@src/kernels/probe")
const worker = require("@src/parallel/worker")
const parallelTypes = require("@src/parallel/types")

local held: parallelTypes.Worker? = nil

--- The loaded worker, or an error when `load` has not run.
const function loaded(): parallelTypes.Worker
    if held == nil then
        error("the worker has not loaded its share, call load first", 0)
    end
    return held :: parallelTypes.Worker
end

return {
    --- Loads share `index` of `workers` of the container file at `path`. Returns the bytes of weights held.
    load = function(
        path: string,
        index: number,
        workers: number,
        context: number,
        chunk: number,
        vectorRead: boolean?,
        kv: string?
    ): number
        const host = lute.host()
        const caps = { vectorRead = if vectorRead ~= nil then vectorRead else probe.run(host.clock).vectorRead }
        const w = worker.new(lute.fileSource(path), host.clock, index, workers, context, chunk, caps, kv)
        held = w
        return w.weightBytes
    end,
    attn = function(seq: number, layer: number, pos0: number, count: number, xn: buffer): buffer
        const w = loaded()
        return w:attn(seq, layer, pos0, count, xn)
    end,
    mlp = function(seq: number, layer: number, count: number, xn: buffer): buffer
        const w = loaded()
        return w:mlp(seq, layer, count, xn)
    end,
    logits = function(seq: number, xn: buffer): buffer
        const w = loaded()
        return w:logits(seq, xn)
    end,
    trim = function(seq: number, length: number)
        const w = loaded()
        w:trim(seq, length)
    end,
    free = function(seq: number)
        const w = loaded()
        w:free(seq)
    end,
    save = function(seq: number, length: number, key: number)
        loaded():save(seq, length, key)
    end,
    restore = function(seq: number, key: number)
        loaded():restore(seq, key)
    end,
    drop = function(key: number)
        loaded():drop(key)
    end,
    length = function(key: number): number
        return loaded():length(key)
    end,
    stats = function(): parallelTypes.Stats
        const w = loaded()
        return w:stats()
    end,
}
