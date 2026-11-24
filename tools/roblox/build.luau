-- Writes the Roblox bundle: the rewritten module tree and a Rojo project that maps it into ReplicatedStorage. Run from
-- the repository root: `lute run tools/roblox/build.luau [outputFolder]`, `build/roblox` when none is given. It fails
-- when a module is longer than Roblox accepts, see docs/roblox.md.
const bundle = require("./bundle")
const fs = require("@std/fs")
const path = require("@std/path")
const process = require("@std/process")

const PROJECT = [[{
    "name": "olelo",
    "tree": {
        "$className": "DataModel",
        "ReplicatedStorage": {
            "olelo": { "$path": "src" }
        }
    }
}
]]

const out = process.args[2] or "build/roblox"
const modules = bundle.modules()
const tooLarge = bundle.oversize(modules)
if #tooLarge > 0 then
    error(`a ModuleScript holds at most {bundle.LIMIT} characters, over: {table.concat(tooLarge, ", ")}`, 0)
end
for _, module in modules do
    const file = `{out}/{module.file}`
    fs.createDirectory(path.dirname(file), { makeParents = true })
    fs.writeStringToFile(file, module.source)
end
fs.createDirectory(out, { makeParents = true })
fs.writeStringToFile(`{out}/default.project.json`, PROJECT)
print(`wrote {#modules} modules and default.project.json to {out}`)
