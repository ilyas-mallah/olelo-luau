-- The spill guard. A function whose register allocation needs more than 23 stack spill slots is skipped by native
-- lowering and runs in the interpreter, about 19 times slower. This compiles a module with `luau-compile` and reads
-- the lowering statistics, so the cliff becomes a build error.
const fs = require("@std/fs")
const json = require("@std/json")
const process = require("@std/process")

--- Lowering numbers of one function.
export type FunctionStat = {
    name: string,
    irCount: number,
    asmCount: number,
}

--- What native lowering did with one module.
export type Report = {
    path: string,
    --- Functions native lowering refused, which run interpreted.
    skipped: number,
    spillsToSlot: number,
    --- The highest spill slot used, against a limit of 23.
    maxSlot: number,
    functions: { FunctionStat },
    ok: boolean,
    --- Why the module failed, nil when it passed.
    problem: string?,
}

--- Both vector flags, as the timed runs and the tests set them.
const VECTOR_FLAGS = "DebugLuauBufferVectorLibrary=true,DebugLuauBufferVectorFastcalls=true"

--- The slot limit of the native backends (`kSpillSlots`), for reports. The skip itself is read from the statistics.
const SLOT_LIMIT = 23

--- The targets every kernel must lower on. The Windows x64 ABI (`x64_ms`, Roblox Studio and Windows servers) keeps fewer
--- vector registers free than the System V one, so it spills more and skips functions the Linux target keeps.
const TARGETS = { "x64", "x64_ms", "a64" }

--- The `luau-compile` to run, `LUAU_COMPILE` or the one on `PATH`. It must be built from the pinned Luau (0.747).
const function compilerPath(): string
    return process.env.LUAU_COMPILE or "luau-compile"
end

--- Whether the compiler can be run.
const function available(): boolean
    const ok, result = pcall(function()
        return process.run({ compilerPath(), "--help" })
    end)
    return ok and (result.stdout ~= "" or result.stderr ~= "")
end

--- Compiles `path` natively for `target` (x64 when nil) without running it and reads the lowering statistics. `scratch` is
--- a folder for the statistics file.
const function analyze(path: string, scratch: string, target: string?): Report
    fs.createDirectory(scratch, { makeParents = true })
    const statsFile = `{scratch}/stats.json`
    if fs.exists(statsFile) then
        fs.remove(statsFile)
    end
    const result = process.run({
        compilerPath(),
        "--codegennull",
        "--record-stats=function",
        `--stats-file={statsFile}`,
        `--target={target or "x64"}`,
        "-O2",
        `--fflags={VECTOR_FLAGS}`,
        path,
    })
    const report: Report =
        { path = path, skipped = 0, spillsToSlot = 0, maxSlot = 0, functions = {}, ok = false, problem = nil }
    if not result.ok or not fs.exists(statsFile) then
        report.problem = `luau-compile failed: {result.stderr}`
        return report
    end
    const root = json.asObject(json.deserialize(fs.readFileToString(statsFile)))
    local entry: json.Value = nil
    if root then
        for _, value in root do
            if json.asObject(value) then
                entry = value
                break
            end
        end
    end
    const file = json.asObject(entry)
    const lower = if file then json.asObject(file.lowerStats) else nil
    if not lower then
        report.problem = "luau-compile wrote no lowering statistics"
        return report
    end
    report.skipped = (lower.skippedFunctions :: number?) or 0
    report.spillsToSlot = (lower.spillsToSlot :: number?) or 0
    report.maxSlot = (lower.maxSpillSlotsUsed :: number?) or 0
    const list = json.asArray(lower.functions)
    if list then
        for _, item in list do
            const f = json.asObject(item)
            if f then
                table.insert(report.functions, {
                    name = (f.name :: string?) or "?",
                    irCount = (f.irCount :: number?) or 0,
                    asmCount = (f.asmCount :: number?) or 0,
                })
            end
        end
    end
    if report.skipped > 0 then
        report.problem =
            `{report.skipped} function(s) skipped by native lowering (max spill slot {report.maxSlot} of {SLOT_LIMIT})`
    elseif #report.functions == 0 then
        report.problem = "no functions in the lowering statistics"
    else
        for _, f in report.functions do
            if f.asmCount == 0 then
                report.problem = `function {f.name} has no native code`
            end
        end
    end
    report.ok = report.problem == nil
    return report
end

return {
    VECTOR_FLAGS = VECTOR_FLAGS,
    SLOT_LIMIT = SLOT_LIMIT,
    TARGETS = TARGETS,
    compilerPath = compilerPath,
    available = available,
    analyze = analyze,
}
