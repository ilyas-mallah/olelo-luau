-- Turns the source tree into the module tree Roblox loads. A Roblox script finds a module by its place in the instance
-- tree, so `require("@src/ops/norm")` becomes `require(script.Parent.Parent.ops.norm)`, and `const` becomes `local` for a
-- Luau that does not read it yet. `src/` becomes the root module and `backends/roblox/` a folder `roblox` in it, which is
-- also the layout `default.project.json` of the output gives Rojo.
const fs = require("@std/fs")

--- The most characters a ModuleScript holds in Roblox Studio. A longer module arrives empty.
const LIMIT = 199997

--- A module of the output: its path from the root as folder and module names, and its text.
export type Module = {
    --- Names from the root, `{}` for the root module itself.
    path: { string },
    --- The file name relative to the output folder.
    file: string,
    source: string,
}

--- The instance expression of the module `target` (names from the root) seen from the module at `from`.
const function expression(from: { string }, target: { string }): string
    local out = "script"
    for _ = 1, #from do
        out ..= ".Parent"
    end
    for _, name in target do
        out ..= `.{name}`
    end
    return out
end

--- Rewrites the requires and the `const` bindings of one module at `path`. In an `init` module, which is the module of
--- its folder, `@self/child` is the child of the script itself.
const function rewrite(source: string, path: { string }): string
    const parent = table.move(path, 1, #path - 1, 1, {})
    local out = string.gsub(source, 'require%("@self/([%w_/]+)"%)', function(target: string): string
        return `require(script.{string.gsub(target, "/", ".")})`
    end)
    out = string.gsub(out, 'require%("@src/([%w_/]+)"%)', function(target: string): string
        const names = string.split(target, "/")
        return `require({expression(path, names)})`
    end)
    out = string.gsub(out, 'require%("%./([%w_]+)"%)', function(name: string): string
        const sibling = table.clone(parent)
        table.insert(sibling, name)
        return `require({expression(path, sibling)})`
    end)
    out = string.gsub(out, "\n(%s*)const ", "\n%1local ")
    out = string.gsub(out, "^(%s*)const ", "%1local ")
    return out
end

--- Every module of the bundle, rewritten, shallowest first so that the module of a folder (its ) comes before its
--- children, then in file order.
const function modules(): { Module }
    const out: { Module } = {}
    const function walk(folder: string, prefix: { string }, into: string)
        for _, entry in fs.listDirectory(folder) do
            const full = `{folder}/{entry.name}`
            if entry.type == "dir" then
                const nested = table.clone(prefix)
                table.insert(nested, entry.name)
                walk(full, nested, `{into}/{entry.name}`)
            elseif string.match(entry.name, "%.server%.luau$") then
                -- A Script keeps its text, it finds modules through an ObjectValue.
                const name = string.gsub(entry.name, "%.server%.luau$", "")
                const path = table.clone(prefix)
                table.insert(path, name)
                table.insert(out, { path = path, file = `{into}/{entry.name}`, source = fs.readFileToString(full) })
            elseif string.match(entry.name, "%.luau$") then
                const name = string.gsub(entry.name, "%.luau$", "")
                const path = table.clone(prefix)
                local file = `{into}/{entry.name}`
                if name == "init" then
                    file = `{into}/init.luau`
                else
                    table.insert(path, name)
                end
                table.insert(out, { path = path, file = file, source = rewrite(fs.readFileToString(full), path) })
            end
        end
    end
    walk("src", {}, "src")
    walk("backends/roblox", { "roblox" }, "src/roblox")
    table.sort(out, function(a: Module, b: Module): boolean
        if #a.path ~= #b.path then
            return #a.path < #b.path
        end
        return a.file < b.file
    end)
    return out
end

--- The files of `list` whose text is over `LIMIT` characters, as `file (characters)`.
const function oversize(list: { Module }): { string }
    const out: { string } = {}
    for _, module in list do
        const length = utf8.len(module.source) or #module.source
        if length > LIMIT then
            table.insert(out, `{module.file} ({length})`)
        end
    end
    return out
end

return {
    LIMIT = LIMIT,
    oversize = oversize,
    rewrite = rewrite,
    expression = expression,
    modules = modules,
}
