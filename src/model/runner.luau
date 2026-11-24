--!native
-- What runs the model for a conversation: one VM with the whole model, or a main worker that divides each pass among
-- workers. A conversation holds one sequence of a runner and calls nothing else of the model.

const graph = require("@src/model/graph")
const session = require("@src/model/session")
const runtimeTypes = require("@src/runtime/types")
const types = require("./types")

type Hook = types.Hook
type Runner = types.Runner
type Sequence = types.Sequence

--- A runner that holds the whole model in this VM.
const function single(m: types.Model, chunk: number, b: runtimeTypes.Budget?): Runner
    const saved: { [number]: types.Snapshot } = {}
    return {
        yields = false,
        vocab = m.config.vocab,
        open = function(_self: Runner): Sequence
            return session.new(m, chunk)
        end,
        forward = function(
            _self: Runner,
            seq: Sequence,
            ids: { number },
            first: number,
            count: number,
            wantLogits: boolean,
            hook: Hook?
        ): buffer?
            return graph.forward(seq :: types.Session, ids, first, count, wantLogits, b, hook)
        end,
        rewind = function(_self: Runner, seq: Sequence, length: number)
            seq.length = length
        end,
        save = function(_self: Runner, seq: Sequence, length: number, key: number)
            saved[key] = session.snapshot(seq :: types.Session, length)
        end,
        restore = function(_self: Runner, seq: Sequence, key: number): number
            const snap = saved[key]
            if snap == nil then
                error(`nothing is saved under {key}`, 0)
            end
            assert(seq.length == 0, "a snapshot goes into an empty sequence")
            session.restore(seq :: types.Session, snap)
            return snap.length
        end,
        drop = function(_self: Runner, key: number)
            saved[key] = nil
        end,
        close = function(_self: Runner, _seq: Sequence) end,
        shutdown = function(_self: Runner) end,
    }
end

return {
    single = single,
}
