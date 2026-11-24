--!native
-- The transport to workers that live in this VM. It runs each request when it is sent and copies buffers and tables
-- in both directions, so code that works over it breaks the same way over a transport between VMs if it shares a
-- buffer. Tests and the single process mode use it.

const host = require("@src/host")
const worker = require("./worker")
const kernelsTypes = require("@src/kernels/types")
const types = require("./types")

--- A copy of a request argument or a reply: buffers and tables of plain values are cloned, the rest is immutable.
const function isolate(v: any): any
    if typeof(v) == "buffer" then
        const b = v :: buffer
        const out = buffer.create(buffer.len(b))
        buffer.copy(out, 0, b, 0, buffer.len(b))
        return out
    elseif type(v) == "table" then
        return table.clone(v)
    end
    return v
end

--- A line to `w` in this VM.
const function inProcess(w: types.Worker): host.Transport
    return {
        call = function(op: string, ...: any): host.Future
            const args = table.pack(...)
            for i = 1, args.n do
                args[i] = isolate(args[i])
            end
            const handler = (w :: { [string]: (types.Worker, ...any) -> ...any })[op]
            if type(handler) ~= "function" then
                return { done = true, value = nil, failure = `the worker has no op {op}` }
            end
            const ok, value = pcall(handler, w, table.unpack(args, 1, args.n))
            if not ok then
                return { done = true, value = nil, failure = tostring(value) }
            end
            return { done = true, value = isolate(value), failure = nil }
        end,
        join = function(futures: { host.Future }): { any }
            const out = table.create<<any>>(#futures)
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

--- `workers` workers of the container `source` in this VM, each behind a transport.
const function inProcessWorkers(
    source: host.WeightSource,
    clock: () -> number,
    workers: number,
    context: number,
    chunk: number,
    caps: kernelsTypes.Caps,
    kv: string?
): { host.Transport }
    const out: { host.Transport } = {}
    for i = 1, workers do
        table.insert(out, inProcess(worker.new(source, clock, i, workers, context, chunk, caps, kv)))
    end
    return out
end

return {
    inProcess = inProcess,
    inProcessWorkers = inProcessWorkers,
}
