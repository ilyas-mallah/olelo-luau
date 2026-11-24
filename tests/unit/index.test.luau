-- `tests/README.md` lists exactly the suites `tests/run.luau` runs, each once, so the index stays the list of what the
-- tests prove.
const fs = require("@std/fs")
const harness = require("@tests/harness")

--- The names in the first column of the index table, as `folder/name`.
const function indexed(readme: string): { string }
    const out: { string } = {}
    for name in string.gmatch(readme, "\n| `([%w_]+/[%w_]+)` |") do
        table.insert(out, name)
    end
    return out
end

--- The suite names `run.luau` registers.
const function registered(run: string): { string }
    const out: { string } = {}
    for name in string.gmatch(run, 'name = "([%w_]+/[%w_]+)"') do
        table.insert(out, name)
    end
    return out
end

--- The `*.test.luau` files under `tests/`, as `folder/name`.
const function onDisk(): { string }
    const out: { string } = {}
    for _, folder in { "unit", "kernels", "gen", "api", "parallel", "serve", "roblox", "parity" } do
        for _, entry in fs.listDirectory(`tests/{folder}`) do
            const name = string.match(entry.name, "^([%w_]+)%.test%.luau$")
            if name then
                table.insert(out, `{folder}/{name}`)
            end
        end
    end
    table.sort(out)
    return out
end

return function(h: harness.Harness)
    h.case("the index, run.luau and the files agree", function()
        const fromIndex = indexed(fs.readFileToString("tests/README.md"))
        const fromRun = registered(fs.readFileToString("tests/run.luau"))
        const files = onDisk()
        table.sort(fromIndex)
        table.sort(fromRun)
        h.equal(table.concat(fromIndex, ","), table.concat(files, ","), "tests/README.md differs from the files")
        h.equal(table.concat(fromRun, ","), table.concat(files, ","), "tests/run.luau differs from the files")
    end)
end
