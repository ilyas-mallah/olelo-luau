-- The lanes of a scheduler for a layout (`src/serve/layout.luau`) on Lute: replica VMs when the layout has one worker per
-- lane, otherwise groups of worker VMs whose main worker is a unit in this VM. The scheduler on top is the host
-- agnostic one of `src/serve/scheduler.luau`.

const engine = require("@src/api/engine")
const host = require("@src/host")
const lute = require("@backends/lute")
const probe = require("@src/kernels/probe")
const unit = require("@src/serve/unit")
const apiTypes = require("@src/api/types")
const serveTypes = require("@src/serve/types")

export type Options = {
    --- Positions a conversation may use, and tokens per prefill pass, 32 when nil.
    context: number,
    chunk: number?,
    kv: string?,
    seed: number?,
    --- Kernel capability, probed once by this VM when nil.
    vectorRead: boolean?,
}

export type Pool = {
    lanes: { serveTypes.LaneSpec },
    --- Ends the engines of the groups and the lines to every VM.
    close: () -> (),
}

--- Starts the VMs of `l` for the container file `path`. Call it from the main chunk or a thread, since it waits for the
--- loads.
const function open(path: string, l: serveTypes.Layout, o: Options): Pool
    const chunk = o.chunk or 32
    const vectorRead = if o.vectorRead ~= nil then o.vectorRead else probe.run(lute.host().clock).vectorRead
    const specs: { serveTypes.LaneSpec } = {}
    const closers: { () -> () } = {}
    if l.workers == 1 then
        const transports = lute.spawnReplicas(
            path,
            l.lanes,
            { context = o.context, chunk = chunk, vectorRead = vectorRead, kv = o.kv, seed = o.seed }
        )
        for i, t in transports do
            specs[i] = { transport = t, workers = 1, slots = l.slots }
            table.insert(closers, t.close)
        end
    else
        for i = 1, l.lanes do
            const workers = lute.spawnWorkers(
                path,
                l.workers,
                { context = o.context, chunk = chunk, vectorRead = vectorRead, kv = o.kv }
            )
            const h = lute.host()
            const e = engine.new({
                host = h,
                source = lute.fileSource(path),
                context = o.context,
                chunk = chunk,
                caps = { vectorRead = vectorRead },
                kv = o.kv,
                seed = o.seed,
                transports = workers,
            })
            const u = unit.new(e)
            specs[i] = {
                transport = lute.threadedTransport({
                    open = function(sid: number, options: serveTypes.OpenOptions?, state: apiTypes.State?)
                        u:open(sid, options, state)
                    end,
                    begin = function(sid: number, text: string, options: serveTypes.ReplyOptions?)
                        u:begin(sid, text, options)
                    end,
                    step = function(sid: number, maxTokens: number?): serveTypes.StepOut
                        return u:step(sid, maxTokens)
                    end,
                    cancel = function(sid: number)
                        u:cancel(sid)
                    end,
                    close = function(sid: number)
                        u:close(sid)
                    end,
                    info = function(): serveTypes.Info
                        return u:info()
                    end,
                }),
                workers = l.workers,
                slots = l.slots,
            }
            table.insert(closers, function()
                e:close()
            end)
        end
    end
    return {
        lanes = specs,
        close = function()
            for _, close in closers do
                close()
            end
        end,
    }
end

return {
    open = open,
}
