-- A streaming chat in the terminal.
--
--   lute run examples/chat.luau <model.olelo> [--prompt "text"] [--system "text"] [--temp 0.7] [--top-k 40]
--       [--top-p 0.9] [--min-p 0.05] [--repeat 1.1] [--max 256] [--context 2048] [--seed 1] [--workers 4]
--
-- Without a sampling flag it samples with temperature 0.7, top-p 0.9 and repetition penalty 1.1, which a small model
-- needs to avoid looping. --temp 0 is greedy. --max counts new tokens, and an answer cut off by it shows finish "length" in
-- the stats line.
--
-- Without --system the model gets a short description of what it is and what it cannot do. The models' own default
-- prompts name their makers and not Olelo, which makes a small model describe itself wrongly.
--
-- With --prompt it answers once and exits. Otherwise it reads lines from the terminal, and /reset, /stats and /quit
-- are commands. For full speed run it with a lute that sets the vector flags, see tools/lute-fix/README.md. With
-- --workers n the layers are divided among n worker VMs (docs/parallel.md), which needs a lute with the vm.create fix.
const engine = require("@src/api/engine")
const io = require("@std/io")
const lute = require("@backends/lute")
const process = require("@std/process")
const apiTypes = require("@src/api/types")

const args = process.args
local modelPath: string? = nil
local prompt: string? = nil
const DEFAULT_SYSTEM = "You are Olelo, an AI chat assistant. You do not have access to internet or tools. "
    .. "Keep answers short and say when you are not sure."
local system: string? = nil
local context = 2048
local seed = 1
local workers = 0
const sampling: apiTypes.SendOptions = { temperature = 0.7, topP = 0.9, repetitionPenalty = 1.1 }
local i = 2
while i <= #args do
    const flag = args[i]
    const value = args[i + 1]
    if flag == "--prompt" then
        prompt = value
    elseif flag == "--system" then
        system = value
    elseif flag == "--temp" then
        sampling.temperature = tonumber(value)
    elseif flag == "--top-k" then
        sampling.topK = tonumber(value)
    elseif flag == "--top-p" then
        sampling.topP = tonumber(value)
    elseif flag == "--min-p" then
        sampling.minP = tonumber(value)
    elseif flag == "--repeat" then
        sampling.repetitionPenalty = tonumber(value)
    elseif flag == "--max" then
        sampling.maxTokens = tonumber(value)
    elseif flag == "--context" then
        context = tonumber(value) or context
    elseif flag == "--seed" then
        seed = tonumber(value) or seed
    elseif flag == "--workers" then
        workers = tonumber(value) or workers
    elseif modelPath == nil and string.sub(flag, 1, 2) ~= "--" then
        modelPath = flag
        i -= 1
    else
        print(`unknown argument {flag}`)
        process.exit(2)
    end
    i += 2
end
const path = modelPath
if path == nil then
    print("usage: lute run examples/chat.luau <model.olelo> [--prompt text] [--temp t] [--max n] ...")
    process.exit(2)
end

const host = lute.host()
local started = host.clock()
const e = engine.new({
    host = host,
    source = lute.fileSource(path :: string),
    context = context,
    seed = seed,
    transports = if workers > 0
        then lute.spawnWorkers(path :: string, workers, { context = context, chunk = 32 })
        else nil,
})
const memory = e:memory()
print(
    string.format(
        "loaded in %.1f s, %.0f MiB, %s kernels%s, temp %s, top-k %s, top-p %s, min-p %s, repeat %s, max %s",
        host.clock() - started,
        memory.total / 1048576,
        if e.caps.vectorRead then "vector" else "scalar read",
        if workers > 0 then `, {workers} workers` else "",
        tostring(sampling.temperature),
        tostring(sampling.topK or "off"),
        tostring(sampling.topP),
        tostring(sampling.minP or "off"),
        tostring(sampling.repetitionPenalty),
        tostring(sampling.maxTokens or 256)
    )
)

const chat = e:chat({ system = system or DEFAULT_SYSTEM })

const function say(text: string)
    io.write("\n")
    local count = 0
    for piece in chat:send(text, sampling) do
        io.write(piece)
        count += 1
    end
    const s = chat:stats()
    io.write(
        string.format(
            "\n[%d prompt tokens at %.0f/s, %d new tokens, first token %.2f s, %.1f tokens/s, %s]\n",
            s.promptTokens,
            s.prefillRate,
            s.newTokens,
            s.timeToFirstToken,
            s.decodeRate,
            s.finish
        )
    )
end

if prompt ~= nil then
    say(prompt)
    process.exit(0)
end

local empty = 0
while true do
    const line = io.input("\n> ")
    if line == "" then
        empty += 1
        if empty >= 3 then
            break
        end
    else
        empty = 0
        if line == "/quit" then
            break
        elseif line == "/reset" then
            chat:reset()
            print("conversation cleared")
        elseif line == "/stats" then
            print(`{chat:used()} positions used of {context}`)
        else
            say(line)
        end
    end
end
