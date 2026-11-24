-- Runs several kernels, each as separate processes pinned to two CPUs, one process at a time, and prints the median
-- of the process medians with the range. The method of the kernel report.
--
--   lute run tools/bench/kernels.luau [processes=5] [cpus=14,15] [rows=1024] [cols=4096] [tokens=1] [id ...]
--
-- An argument that is a path to a `.luau` file is run as it is, which is how a scratch benchmark that prints a `RES`
-- line is timed in the same session. Every id and file is run `processes` times, interleaved by round. Set
-- LUTE_SET_FFLAGS in the environment to run with the vector flags.
const path = require("@std/path")
const process = require("@std/process")

local processes = 5
local cpus = "14,15"
local rows = "1024"
local cols = "4096"
local tokens = "1"
const targets: { string } = {}
for i = 2, #process.args do
    const argument = process.args[i]
    const key, value = string.match(argument, "^(%a+)=(.+)$")
    if key == "processes" then
        processes = tonumber(value) or processes
    elseif key == "cpus" then
        cpus = value :: string
    elseif key == "rows" then
        rows = value :: string
    elseif key == "cols" then
        cols = value :: string
    elseif key == "tokens" then
        tokens = value :: string
    else
        table.insert(targets, argument)
    end
end

const lute = path.format(process.execPath())
const samples: { [string]: { number } } = {}
const sums: { [string]: string } = {}
for round = 1, processes do
    for _, target in targets do
        local command: { string }
        if string.match(target, "%.luau$") then
            command = { "taskset", "-c", cpus, lute, "run", target }
        else
            command = { "taskset", "-c", cpus, lute, "run", "tools/bench/kernel.luau", target, rows, cols, tokens }
        end
        const result = process.run(command)
        const line = string.match(result.stdout, "RES [^\n]+")
        const rate = if line then string.match(line, "macs_per_s_med=(%S+)") else nil
        if rate == nil then
            print(`{target} round {round} failed: {result.stderr}{result.stdout}`)
        else
            samples[target] = samples[target] or {}
            table.insert(samples[target], tonumber(rate) :: number)
            sums[target] = string.match(line :: string, "csum=(%S+)") or "?"
        end
    end
end
for _, target in targets do
    const list = samples[target]
    if list and #list > 0 then
        table.sort(list)
        print(
            string.format(
                "%-28s %8.1f MMAC/s median of %d process medians, range %.1f to %.1f, csum %s",
                target,
                list[(#list + 1) // 2] / 1e6,
                #list,
                list[1] / 1e6,
                list[#list] / 1e6,
                sums[target]
            )
        )
    end
end
