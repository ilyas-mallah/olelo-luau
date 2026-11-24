-- Cuts a container into the chunk ModuleScripts `weights.fromModules` reads: files `1.luau`, `2.luau` and so on, each
-- returning `chunkBytes` bytes of the container as base64 text, and `info.json` with the size and the chunk size. Rojo
-- maps the folder to a Folder of ModuleScripts. Run from the repository root:
--
--   lute run tools/roblox/chunks.luau <model.olelo> <outputFolder> [chunkBytes]
--
-- The chunk size defaults to 65536, and at most 149,991 bytes fit a module (docs/roblox.md, Weights).
const base64 = require("./base64")
const bundle = require("./bundle")
const fs = require("@std/fs")
const process = require("@std/process")

const modelPath, outFolder = process.args[2], process.args[3]
if modelPath == nil or outFolder == nil then
    error("usage: lute run tools/roblox/chunks.luau <model.olelo> <outputFolder> [chunkBytes]", 0)
end
const chunkBytes = tonumber(process.args[4]) or 65536
-- `return "` and `"` are 9 characters around the base64 text, and a file has no final newline.
if (math.ceil(chunkBytes / 3) * 4 + 9) > bundle.LIMIT then
    error(`a chunk of {chunkBytes} bytes is over the {bundle.LIMIT} characters of a ModuleScript`, 0)
end
const data = fs.readFileToString(modelPath)
fs.createDirectory(outFolder, { makeParents = true })
const count = math.ceil(#data / chunkBytes)
for i = 1, count do
    const text = base64.encode(string.sub(data, (i - 1) * chunkBytes + 1, i * chunkBytes))
    fs.writeStringToFile(`{outFolder}/{i}.luau`, `return "{text}"\n`)
end
fs.writeStringToFile(`{outFolder}/info.json`, `\{"size": {#data}, "chunkBytes": {chunkBytes}, "chunks": {count}\}\n`)
print(`wrote {count} chunks of {chunkBytes} bytes and info.json to {outFolder}`)
