-- The spill guard passes every generated kernel with room to spare, and it fails a kernel that falls off the native
-- spill cliff, which proves the guard can see the cliff. Needs luau-compile of the pinned Luau.
const fs = require("@std/fs")
const generate = require("@tools/gen/generate")
const harness = require("@tests/harness")
const spill = require("@tools/gen/spill")
const tile = require("@tools/gen/tile")

const SCRATCH = "out/spill-test"

return function(h: harness.Harness)
    if not spill.available() then
        h.skip("spill guard", "luau-compile is not available, set LUAU_COMPILE")
        return
    end

    h.case("every generated module stays on native code with spill slots to spare", function()
        for _, file in generate.renderEvery() do
            const report = spill.analyze(file.path, SCRATCH)
            h.truthy(report.ok, `{file.path}: {report.problem}`)
            h.truthy(
                report.maxSlot <= spill.SLOT_LIMIT - 8,
                `{file.path} uses slot {report.maxSlot} of {spill.SLOT_LIMIT}`
            )
        end
    end)

    h.case("every generated module stays on native code on every target", function()
        for _, file in generate.renderEvery() do
            for _, target in spill.TARGETS do
                const report = spill.analyze(file.path, SCRATCH, target)
                h.truthy(report.ok, `{file.path} on {target}: {report.problem}`)
            end
        end
    end)

    h.case(
        "a 4 row by 2 token vector tile is kept on x64 and skipped on the Windows ABI, and the guard says so",
        function()
            -- Measured in Roblox Studio: this tile falls to the interpreter there (docs/kernels.md).
            const path = "tests/gen/cliff_ms.luau"
            h.equal(spill.analyze(path, SCRATCH, "x64").ok, true)
            const ms = spill.analyze(path, SCRATCH, "x64_ms")
            h.equal(ms.ok, false)
            h.truthy(ms.skipped >= 1, "the Windows ABI did not skip the tile")
        end
    )

    h.case("a vector tile with all weights loaded first is skipped, and the guard says so", function()
        -- 8 rows x 4 unrolled vectors loaded up front needs more than 23 spill slots, the cliff the load order avoids.
        const spec: tile.ModuleSpec = {
            file = "cliff",
            family = "float_vector",
            format = "f32",
            layout = "rowmajor",
            activation = "f32",
            strictness = "nonstrict",
            summary = "a tile that does not fit",
            tiles = { { name = "R8S1U4", fmt = "f32", src = "rv", R = 8, S = 1, U = 4, order = "all" } },
        }
        fs.createDirectory(SCRATCH, { makeParents = true })
        const path = `{SCRATCH}/cliff.luau`
        fs.writeStringToFile(path, generate.renderModule(spec))
        const report = spill.analyze(path, SCRATCH)
        h.equal(report.ok, false)
        h.truthy(report.skipped >= 1, "the cliff tile was not reported as skipped")
    end)
end
