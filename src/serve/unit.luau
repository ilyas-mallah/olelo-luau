--!native
-- A unit serves conversations on one engine: a replica (the whole model in one VM) or a group of tensor parallel workers
-- behind one main worker. A scheduler talks to it through a transport, so every argument and result is plain data. A
-- conversation is a session id, and a job on it (a reply, or the rebuild of its cache) runs in steps, so the scheduler
-- decides who goes next.

const chat = require("@src/api/chat")
const apiTypes = require("@src/api/types")
const types = require("./types")

type Info = types.Info
type OpenOptions = types.OpenOptions
type ReplyOptions = types.ReplyOptions
type StepOut = types.StepOut
type Unit = types.Unit

type Held = {
    chat: apiTypes.Chat,
    turn: apiTypes.Turn?,
    --- A rebuild of the cache rather than a reply.
    restoring: boolean,
}

--- A unit on `e`. `clock` is the engine's own, so the seconds of a step are on one clock.
const function new(e: apiTypes.Engine): Unit
    const held: { [number]: Held } = {}
    local count = 0
    const clock = e.core.host.clock
    const mem = e:memory()

    const function find(sid: number): Held
        const h = held[sid]
        if h == nil then
            error(`conversation {sid} is not open`, 0)
        end
        return h
    end

    return {
        open = function(_self: Unit, sid: number, options: OpenOptions?, state: apiTypes.State?)
            if held[sid] ~= nil then
                error(`conversation {sid} is already open`, 0)
            end
            const c = e:chat({
                system = if options then options.system else nil,
                seed = if options then options.seed else nil,
            })
            local turn: apiTypes.Turn? = nil
            if state ~= nil and #state.tokens > 0 then
                turn = c:resume(state)
            end
            held[sid] = { chat = c, turn = turn, restoring = turn ~= nil }
            count += 1
        end,
        begin = function(_self: Unit, sid: number, text: string, options: ReplyOptions?)
            const h = find(sid)
            if h.turn ~= nil then
                error(`conversation {sid} has a job in progress`, 0)
            end
            h.turn = h.chat:begin(text, options or {})
            h.restoring = false
        end,
        step = function(_self: Unit, sid: number, maxTokens: number?): StepOut
            const h = find(sid)
            const turn = h.turn
            if turn == nil then
                error(`conversation {sid} has no job`, 0)
            end
            const t0 = clock()
            const r = turn:step(maxTokens)
            const out: StepOut = {
                text = table.concat(r.pieces),
                done = r.done,
                kind = r.kind,
                tokens = r.tokens,
                seconds = clock() - t0,
            }
            if r.done then
                h.turn = nil
                if not h.restoring then
                    const s = h.chat:stats()
                    out.finish = s.finish
                    out.promptTokens = s.promptTokens
                    out.newTokens = s.newTokens
                    out.cachedTokens = s.cachedTokens
                    out.prefillRate = s.prefillRate
                    out.decodeRate = s.decodeRate
                    const snapshot = h.chat:snapshot()
                    out.history = snapshot.tokens
                    out.pending = snapshot.pending
                end
                h.restoring = false
            end
            return out
        end,
        cancel = function(_self: Unit, sid: number)
            const h = held[sid]
            const turn = if h ~= nil then h.turn else nil
            if turn ~= nil then
                turn:cancel()
            end
        end,
        close = function(_self: Unit, sid: number)
            const h = held[sid]
            if h ~= nil then
                h.chat:close()
                held[sid] = nil
                count -= 1
            end
        end,
        info = function(_self: Unit): Info
            return {
                sessions = count,
                weightBytes = mem.weights,
                sessionBytes = mem.cache,
                kv = e.kv,
                context = e.context,
            }
        end,
    }
end

return {
    new = new,
}
