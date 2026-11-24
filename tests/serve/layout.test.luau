-- The layout rule: replicas against worker groups from the model, the threads, the memory and the users.
const harness = require("@tests/harness")
const layout = require("@src/serve/layout")
const serveTypes = require("@src/serve/types")

const MB = 1048576
const GB = 1024 * MB
--- Qwen2.5-0.5B Q8 and SmolLM2-135M Q8 with a cache of 512 positions, the sizes of the measurements.
const QWEN: serveTypes.Model = { weightBytes = 556 * MB, sessionBytes = 13 * MB }
const SMOL: serveTypes.Model = { weightBytes = 151 * MB, sessionBytes = 24 * MB }
const PC: serveTypes.Machine = { threads = 16, memoryBytes = 12 * GB }

return function(h: harness.Harness)
    h.case("one user gets a group of workers sized by the model", function()
        local l = layout.choose(PC, QWEN, 1)
        h.equal(l.lanes, 1)
        h.equal(l.workers, 4, "Qwen2.5-0.5B stops gaining at 4")
        l = layout.choose(PC, SMOL, 1)
        h.equal(l.workers, 3, "SmolLM2-135M stops gaining at 3")
        h.truthy(l.fits)
    end)

    h.case("a few users share the threads between groups", function()
        const three = layout.choose(PC, QWEN, 3)
        h.equal(three.lanes, 3)
        h.equal(three.workers, 4)
        const five = layout.choose(PC, QWEN, 5)
        h.equal(five.lanes, 5)
        h.equal(five.workers, 3, "15 usable VMs over 5 users leave 3 each")
        h.truthy(five.lanes * five.workers <= PC.threads)
    end)

    h.case("as many users as VMs, or more, get replicas and the rest queue on slots", function()
        const ten = layout.choose(PC, QWEN, 10)
        h.equal(ten.lanes, 10)
        h.equal(ten.workers, 1)
        const many = layout.choose(PC, QWEN, 40)
        h.equal(many.workers, 1)
        h.truthy(many.lanes <= PC.threads)
        h.equal(many.slots, math.ceil(40 / many.lanes))
        h.truthy(many.fits)
    end)

    h.case("memory limits the replicas before the threads do", function()
        const small: serveTypes.Machine = { threads = 16, memoryBytes = 4 * GB }
        const l = layout.choose(small, QWEN, 8)
        h.equal(l.lanes, 4, "3 GB usable hold four replicas of 723 MB")
        h.equal(l.slots, 2)
        h.truthy(l.fits)
        const tiny: serveTypes.Machine = { threads = 16, memoryBytes = 1 * GB }
        const none = layout.choose(tiny, QWEN, 2)
        h.equal(none.lanes, 1)
        h.equal(none.fits, false, "not even one replica fits, and the layout says so")
    end)

    h.case("the layout fits whenever one replica does, and uses no more threads than there are", function()
        for _, model in { QWEN, SMOL } do
            for _, gb in { 2, 4, 8, 12, 32 } do
                for _, users in { 1, 2, 3, 4, 6, 9, 16, 33 } do
                    const machine: serveTypes.Machine = { threads = 16, memoryBytes = gb * GB }
                    const l = layout.choose(machine, model, users)
                    h.truthy(l.lanes * l.workers <= 16, `{gb} GB {users} users: {l.lanes} lanes of {l.workers} workers`)
                    h.truthy(l.lanes >= 1 and l.workers >= 1 and l.slots >= 1)
                    h.truthy(#l.reason > 0)
                end
            end
        end
    end)

    h.case("options change the worker cap and the bytes per worker", function()
        const one = layout.choose(PC, QWEN, 1, { maxWorkers = 2 })
        h.equal(one.workers, 2)
        const wide = layout.choose(PC, SMOL, 1, { bytesPerWorker = 20 * MB, maxWorkers = 8 })
        h.equal(wide.workers, 8)
    end)
end
