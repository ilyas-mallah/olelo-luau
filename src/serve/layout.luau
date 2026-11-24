--!native
-- Chooses how many replicas and tensor parallel worker groups serve a number of users, from the model's size, the cores
-- and the memory. The rule rests on measurements (docs/serve.md, docs/benchmarks.md): replicas add up to the cores they
-- have and cost a whole model each, workers in a group speed up one conversation but with less than one replica's
-- throughput per core, and a model gains from workers in proportion to the bytes its weights take.

const types = require("./types")

type Layout = types.Layout
type Machine = types.Machine
type Model = types.Model

export type Options = {
    --- Most workers in a group, 4 when nil.
    maxWorkers: number?,
    --- Weight bytes that earn a model one more worker in its group, 50 MB when nil.
    bytesPerWorker: number?,
    --- Memory a VM takes beyond the engine's estimate, 150 MB when nil.
    vmBytes: number?,
}

--- Memory a VM takes beyond the engine's estimate: about 81 MiB for SmolLM2-135M and 133 MiB for Qwen2.5-0.5B were measured.
const VM_BYTES = 150 * 1048576
--- SmolLM2-135M Q8 (152 MB) stops gaining at 3 workers and Qwen2.5-0.5B Q8 (560 MB) gains 3.6 times at 4 and 3.7 at 8.
const BYTES_PER_WORKER = 50 * 1048576
const MAX_WORKERS = 4

--- The layout for `users` conversations at once.
const function choose(machine: Machine, model: Model, users: number, options: Options?): Layout
    const o: Options = options or {}
    const vm = o.vmBytes or VM_BYTES
    const usable = machine.memoryBytes - (machine.reserveBytes or 1073741824)
    const people = math.max(1, users)

    -- One VM holds the model once, a group of w workers holds it once in all and w VMs.
    const function bytesOf(lanes: number, workers: number, slots: number): number
        return lanes * (model.weightBytes + workers * vm + slots * model.sessionBytes)
    end

    const wide = math.clamp(
        math.floor(model.weightBytes / (o.bytesPerWorker or BYTES_PER_WORKER) + 0.5),
        1,
        o.maxWorkers or MAX_WORKERS
    )
    const replicas = math.max(1, math.min(machine.threads, math.floor(usable / bytesOf(1, 1, 1))))

    if people >= replicas then
        -- More conversations than VMs: the most replicas, since they add up and workers do not.
        local slots = math.ceil(people / replicas)
        while slots > 1 and bytesOf(replicas, 1, slots) > usable do
            slots -= 1
        end
        const bytes = bytesOf(replicas, 1, slots)
        return {
            lanes = replicas,
            workers = 1,
            slots = slots,
            bytes = bytes,
            fits = bytes <= usable,
            reason = `{people} users reach the {replicas} VMs the threads and memory allow, so every lane is a replica`,
        }
    end

    -- Fewer conversations than VMs: give each its own lane and spend the spare threads on workers.
    local workers = math.min(wide, replicas // people)
    while workers > 1 and bytesOf(people, workers, 1) > usable do
        workers -= 1
    end
    const bytes = bytesOf(people, workers, 1)
    const mb = math.floor(model.weightBytes / 1048576)
    return {
        lanes = people,
        workers = workers,
        slots = 1,
        bytes = bytes,
        fits = bytes <= usable,
        reason = if workers == 1
            then `{people} users take {people} replicas, and no thread is left for workers`
            else `{people} users take a group of {workers} workers each, which {mb} MB of weights gains from`,
    }
end

return {
    choose = choose,
}
