-- Runs the spill guard over every generated kernel module and exits 1 when any module has a function that native
-- lowering would skip. Needs `luau-compile` of the pinned Luau, see `spill.luau`. Run from the root.
const generate = require("@tools/gen/generate")
const process = require("@std/process")
const spill = require("@tools/gen/spill")

local failed = 0
print("module                 target  functions  max slot  spills  result")
for _, file in generate.renderEvery() do
    for _, target in spill.TARGETS do
        const report = spill.analyze(file.path, "out/spill", target)
        const name = string.match(file.path, "([%w_]+)%.luau$") :: string
        const result = if report.ok then "ok" else `FAIL {report.problem}`
        print(
            string.format(
                "%-22s %-6s %9d  %8d  %6d  %s",
                name,
                target,
                #report.functions,
                report.maxSlot,
                report.spillsToSlot,
                result
            )
        )
        if not report.ok then
            failed += 1
        end
    end
end
if failed > 0 then
    print(`{failed} module and target pair(s) fall off the native spill cliff`)
    process.exit(1)
end
