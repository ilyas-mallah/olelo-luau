--!native
-- The module an Actor's script runs to become a worker of tensor parallel inference, or a replica that serves whole
-- conversations. `bind` answers the requests of docs/parallel.md as messages: "req" with an id, the op and its arguments
-- comes in, "rep" with the id, a flag and the value or the error message goes back to the Actor that sent "hello". The
-- first load decides what the Actor is: "load" makes it a worker, "replica" a unit (`src/serve/unit.luau`) with an engine
-- of its own. The Actor sets its attribute `ready` once every topic is bound, because a message sent earlier is dropped
-- (docs/roblox.md).

const engine = require("@src/api/engine")
const hostModule = require("./host")
const probe = require("@src/kernels/probe")
const unit = require("@src/serve/unit")
const weights = require("./weights")
const worker = require("@src/parallel/worker")
const apiTypes = require("@src/api/types")
const parallelTypes = require("@src/parallel/types")
const serveTypes = require("@src/serve/types")

--- The topic of the requests that run in the parallel phase, and of the ones that need the serial phase.
const TOPIC = "req"
const SERIAL_TOPIC = "sreq"

--- The ops that open a weight source, which `require` or `HttpService` do, so they are sent on `SERIAL_TOPIC`.
const SERIAL_OPS: { [string]: boolean } = { load = true, replica = true }

--- The attribute a worker Actor sets once it has bound its topics.
const READY = "ready"

--- Where a worker reads its share of the container from.
export type SourceSpec = {
    --- "modules": the folder of chunk ModuleScripts. "http": the url.
    kind: string,
    folder: any?,
    url: string?,
    size: number,
    chunkBytes: number,
}

--- The weight source a spec describes. `request` is `HttpService` and `require` the script's.
const function openSource(spec: SourceSpec, request: any, require: (module: any) -> any): any
    if spec.kind == "modules" then
        return weights.fromModules(spec.folder, spec.size, spec.chunkBytes, require)
    elseif spec.kind == "http" then
        return weights.fromHttp(request, spec.url :: string, spec.size, spec.chunkBytes)
    end
    error(`unknown weight source {spec.kind}`, 0)
end

--- Makes `actor` a worker. `request` is `HttpService` and `require` the script's `require`.
const function bind(actor: any, p: hostModule.Platform, request: any, require: (module: any) -> any)
    const function paced(source: any): any
        return weights.paced(source, p.clock, p.wait, weights.LOAD_SLICE)
    end
    local held: parallelTypes.Worker? = nil
    local serving: serveTypes.Unit? = nil
    local replyTo: any = nil
    const function loaded(): parallelTypes.Worker
        if held == nil then
            error("the worker has not loaded its share, send load first", 0)
        end
        return held :: parallelTypes.Worker
    end
    const function serves(): serveTypes.Unit
        if serving == nil then
            error("the replica has not loaded its model, send replica first", 0)
        end
        return serving :: serveTypes.Unit
    end
    const handlers: { [string]: (...any) -> ...any } = {
        load = function(
            index: number,
            workers: number,
            context: number,
            chunk: number,
            vectorRead: boolean?,
            spec: SourceSpec
        ): number
            const caps = { vectorRead = if vectorRead ~= nil then vectorRead else probe.run(p.clock).vectorRead }
            const w =
                worker.new(paced(openSource(spec, request, require)), p.clock, index, workers, context, chunk, caps)
            held = w
            return w.weightBytes
        end,
        attn = function(seq: number, layer: number, pos0: number, count: number, xn: buffer): buffer
            return loaded():attn(seq, layer, pos0, count, xn)
        end,
        mlp = function(seq: number, layer: number, count: number, xn: buffer): buffer
            return loaded():mlp(seq, layer, count, xn)
        end,
        logits = function(seq: number, xn: buffer): buffer
            return loaded():logits(seq, xn)
        end,
        trim = function(seq: number, length: number)
            loaded():trim(seq, length)
        end,
        free = function(seq: number)
            loaded():free(seq)
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
        stats = function(): any
            return loaded():stats()
        end,
        replica = function(
            context: number,
            chunk: number,
            vectorRead: boolean?,
            kv: string?,
            seed: number?,
            spec: SourceSpec
        ): number
            const h = hostModule.new(p)
            const e = engine.new({
                host = h,
                source = paced(openSource(spec, request, require)),
                context = context,
                chunk = chunk,
                caps = { vectorRead = if vectorRead ~= nil then vectorRead else probe.run(p.clock).vectorRead },
                kv = kv,
                seed = seed,
            })
            serving = unit.new(e)
            return e:memory().weights
        end,
        open = function(sid: number, options: serveTypes.OpenOptions?, state: apiTypes.State?)
            serves():open(sid, options, state)
        end,
        begin = function(sid: number, text: string, options: serveTypes.ReplyOptions?)
            serves():begin(sid, text, options)
        end,
        step = function(sid: number, maxTokens: number?): serveTypes.StepOut
            return serves():step(sid, maxTokens)
        end,
        cancel = function(sid: number)
            serves():cancel(sid)
        end,
        close = function(sid: number)
            serves():close(sid)
        end,
        info = function(): serveTypes.Info
            return serves():info()
        end,
    }
    const function answer(id: number, op: string, ...: any)
        const handler = handlers[op]
        if handler == nil then
            replyTo:SendMessage("rep", id, false, `the worker has no op {op}`)
            return
        end
        const ok, value = pcall(handler, ...)
        replyTo:SendMessage("rep", id, ok, if ok then value else tostring(value))
    end
    actor:BindToMessage("hello", function(main: any)
        replyTo = main
    end)
    -- `require` and `HttpService` are refused in the parallel phase, so the ops that open a source come on the serial topic.
    actor:BindToMessage(SERIAL_TOPIC, answer)
    actor:BindToMessageParallel(TOPIC, answer)
    actor:SetAttribute(READY, true)
end

return {
    TOPIC = TOPIC,
    SERIAL_TOPIC = SERIAL_TOPIC,
    SERIAL_OPS = SERIAL_OPS,
    READY = READY,
    bind = bind,
    openSource = openSource,
}
