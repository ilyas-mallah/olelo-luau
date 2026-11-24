-- Lists the Luau files under a folder, for tests that read the source tree.
const fs = require("@std/fs")

--- Every `.luau` file under `root`, as a path from the repository root, sorted. Walks with a work list.
const function luauFiles(root: string): { string }
    const out: { string } = {}
    const pending = { root }
    while #pending > 0 do
        const folder = table.remove(pending) :: string
        for _, entry in fs.listDirectory(folder) do
            const full = `{folder}/{entry.name}`
            if entry.type == "dir" then
                table.insert(pending, full)
            elseif string.match(entry.name, "%.luau$") then
                table.insert(out, full)
            end
        end
    end
    table.sort(out)
    return out
end

return {
    luauFiles = luauFiles,
}
