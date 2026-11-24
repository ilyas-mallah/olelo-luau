-- Every module of the core is native: it starts with `--!native`, and native lowering skips none of its functions, so
-- no hot kernel runs interpreted. Needs luau-compile of the pinned Luau for the lowering half.
const fs = require("@std/fs")
const harness = require("@tests/harness")
const files = require("@tests/files")
const spill = require("@tools/gen/spill")

const SCRATCH = "out/native-test"

return function(h: harness.Harness)
    const sources = files.luauFiles("src")

    h.case("every module under src starts with --!native", function()
        h.truthy(#sources > 10)
        for _, path in sources do
            const text = fs.readFileToString(path)
            h.truthy(string.find(string.sub(text, 1, 200), "--!native", 1, true) ~= nil, `{path} has no --!native`)
        end
    end)

    if not spill.available() then
        h.skip("native lowering of src", "luau-compile is not available, set LUAU_COMPILE")
        return
    end

    h.case("native lowering skips no function of src", function()
        for _, path in sources do
            const report = spill.analyze(path, SCRATCH)
            h.truthy(report.ok, `{path}: {report.problem}`)
        end
    end)
end
