-- Writes the generated kernel modules into `src/kernels/generated/`. With `--check` it writes nothing and exits 1 when a
-- file differs from what the generator renders, which is how CI keeps the checked-in kernels honest. Run from the root.
const fs = require("@std/fs")
const generate = require("@tools/gen/generate")
const process = require("@std/process")

local check = false
for i = 2, #process.args do
    const argument = process.args[i]
    if argument == "--check" then
        check = true
    else
        print(`unknown argument {argument}`)
        process.exit(2)
    end
end

local drift = 0
for _, file in generate.renderEvery() do
    if check then
        const current = if fs.exists(file.path) then fs.readFileToString(file.path) else nil
        if current ~= file.text then
            print(`differs: {file.path}`)
            drift += 1
        end
    else
        fs.writeStringToFile(file.path, file.text)
        print(`wrote {file.path}`)
    end
end

if drift > 0 then
    print(`{drift} generated file(s) differ from the generator, run lute run tools/gen/main.luau`)
    process.exit(1)
end
