-- The Lute backend: a wall clock, no yielding, and weights read straight from a file into buffers.

const fs = require("@std/fs")
const host = require("@src/host")
const probe = require("@src/kernels/probe")
const task = require("@std/task")
const time = require("@std/time")
const vm = require("@lute/vm")

--- A host that runs in the foreground of a Lute process.
const function newHost(): host.Host
    const start = time.now()
    return {
        name = "lute",
        clock = function(): number
            return time.since(start)
        end,
        yield = function() end,
        log = function(level: string, message: string)
            print(`[{level}] {message}`)
        end,
    }
end

--- Reads a container file. Each read goes from the file into the destination buffer with no string in between.
const function fileSource(path: string): host.WeightSource
    const size = fs.metadata(path).size
    const handle = fs.open(path, "r")
    return {
        size = function(): number
            return size
        end,
        read = function(offset: number, length: number, dst: buffer, dstOffset: number)
            local done = 0
            while done < length do
                const got = fs.readIntoBuffer(handle, dst, offset + done, length - done, dstOffset + done)
                if got <= 0 then
                    error(`{path}: short read at byte {offset + done}, wanted {length - done} more`, 0)
                end
                done += got
            end
        end,
    }
end

--- What `spawnWorkers` takes.
export type WorkerOptions = {
    --- Positions a sequence may use, as the engine's `context`.
    context: number,
    --- Tokens per pass, as the engine's `chunk`.
    chunk: number,
    --- Kernel capability of the workers. When nil this VM probes once and the workers take its answer, since probes of
    --- several workers that run at once time each other.
    vectorRead: boolean?,
    --- The KV cache format of the workers, as the engine's `kv`.
    kv: string?,
}

--- A line to the worker VM `handle`. A request runs in a thread of its own, since a call into another VM yields its
--- caller, and `join` polls the scheduler until every reply is in.
const function vmTransport(handle: { [any]: any }): host.Transport
    return {
        call = function(op: string, ...: any): host.Future
            const future: host.Future = { done = false, value = nil, failure = nil }
            const args = table.pack(...)
            task.spawn(function()
                const ok, value = pcall(handle[op], table.unpack(args, 1, args.n))
                if ok then
                    future.value = value
                else
                    future.failure = tostring(value)
                end
                future.done = true
            end)
            return future
        end,
        join = function(futures: { host.Future }): { any }
            local waiting = true
            while waiting do
                waiting = false
                for _, f in futures do
                    if not f.done then
                        waiting = true
                        break
                    end
                end
                if waiting then
                    task.wait()
                end
            end
            const out: { any } = {}
            for i, f in futures do
                const failure = f.failure
                if failure ~= nil then
                    error(failure, 0)
                end
                out[i] = f.value
            end
            return out
        end,
        close = function() end,
    }
end

--- Starts `workers` worker VMs that each load their share of the container file `path`, and returns a transport to each.
--- Call it from the main chunk or a thread, since it waits for the loads.
const function spawnWorkers(path: string, workers: number, options: WorkerOptions): { host.Transport }
    const transports = table.create<<host.Transport>>(workers)
    const handles = table.create<<{ [any]: any }>>(workers)
    for i = 1, workers do
        const handle = vm.create("@self/worker")
        handles[i] = handle
        transports[i] = vmTransport(handle)
    end
    const vectorRead = if options.vectorRead ~= nil then options.vectorRead else probe.run(newHost().clock).vectorRead
    const futures = table.create<<host.Future>>(workers)
    for i = 1, workers do
        futures[i] =
            transports[i].call("load", path, i, workers, options.context, options.chunk, vectorRead, options.kv)
    end
    transports[1].join(futures)
    return transports
end

--- What `spawnReplicas` takes.
export type ReplicaOptions = {
    --- Positions a conversation may use and tokens per prefill pass, as the engine's `context` and `chunk`.
    context: number,
    chunk: number?,
    --- Kernel capability of the replicas, probed once by this VM when nil.
    vectorRead: boolean?,
    --- The KV cache format, as the engine's `kv`.
    kv: string?,
    seed: number?,
}

--- Starts `count` VMs that each hold the whole model of the container file `path` and serve conversations on it, and
--- returns a transport to each, for the lanes of a scheduler. Replicas share nothing and exchange nothing. Call it from
--- the main chunk or a thread, since it waits for the loads.
const function spawnReplicas(path: string, count: number, options: ReplicaOptions): { host.Transport }
    const transports = table.create<<host.Transport>>(count)
    for i = 1, count do
        transports[i] = vmTransport(vm.create("@self/replica"))
    end
    const vectorRead = if options.vectorRead ~= nil then options.vectorRead else probe.run(newHost().clock).vectorRead
    const futures = table.create<<host.Future>>(count)
    for i = 1, count do
        futures[i] =
            transports[i].call("load", path, options.context, options.chunk or 32, vectorRead, options.kv, options.seed)
    end
    transports[1].join(futures)
    return transports
end

--- A transport to a unit that lives in this VM, for a group of tensor parallel workers whose main worker is here. Each
--- request runs in a thread of its own, since a pass over workers yields while it waits for them, so several units in
--- this VM make progress at once.
const function threadedTransport(handlers: { [string]: (...any) -> ...any }): host.Transport
    return {
        call = function(op: string, ...: any): host.Future
            const future: host.Future = { done = false, value = nil, failure = nil }
            const args = table.pack(...)
            task.spawn(function()
                const handler = handlers[op]
                const ok, value = pcall(function(): ...any
                    if handler == nil then
                        error(`the unit has no op {op}`, 0)
                    end
                    return handler(table.unpack(args, 1, args.n))
                end)
                if ok then
                    future.value = value
                else
                    future.failure = tostring(value)
                end
                future.done = true
            end)
            return future
        end,
        join = function(futures: { host.Future }): { any }
            local waiting = true
            while waiting do
                waiting = false
                for _, f in futures do
                    if not f.done then
                        waiting = true
                        break
                    end
                end
                if waiting then
                    task.wait()
                end
            end
            const out: { any } = {}
            for i, f in futures do
                const failure = f.failure
                if failure ~= nil then
                    error(failure, 0)
                end
                out[i] = f.value
            end
            return out
        end,
        close = function() end,
    }
end

return {
    spawnReplicas = spawnReplicas,
    threadedTransport = threadedTransport,
    host = newHost,
    fileSource = fileSource,
    spawnWorkers = spawnWorkers,
    vmTransport = vmTransport,
}
