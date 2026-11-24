-- The Roblox backend against a Luau model of Roblox (tests/roblox/mock.luau): the bundle's rewritten sources run as
-- ModuleScripts in an instance tree, the weights come from chunk ModuleScripts or HTTP range requests, and the workers
-- are Actors that each have a VM of their own. Passing means the wiring is right and the results equal the single VM's.
-- The model enforces the rules of docs/roblox.md that decide whether code runs at all (tests/roblox/mock.luau), and
-- says nothing about Roblox Studio's timing or limits.
const attentionGen = require("@tools/gen/attention")
const attentionQ8Gen = require("@tools/gen/attention_q8")
const base64 = require("@tools/roblox/base64")
const bundle = require("@tools/roblox/bundle")
const fs = require("@std/fs")
const harness = require("@tests/harness")
const weights = require("@backends/roblox/weights")
const worlds = require("@tests/roblox/world")

const PROMPT =
    "the quick brown fox jumps over the lazy dog and the dog does not mind, what is a hash table? hello. the quick brown fox jumps over the lazy dog and the dog does not mind, what is a hash table? hello"

--- The greedy tokens of the Roblox engine and the reference on the same prompt, which must be equal. The reference runs
--- and the file is read outside the model, since Lute's own calls yield to Lute's scheduler.
const function compare(h: harness.Harness, path: string, workers: number, latency: number, parallelMain: boolean?)
    const data = fs.readFileToString(path)
    const expected = worlds.reference(path)
    const ids = expected:tokenize(PROMPT)
    const want = table.concat(expected:generate(ids, 10), ",")
    const w = worlds.world()
    w.mock.setLatency(latency)
    worlds.run(w, function()
        const engine = worlds.engineOver(w, data, workers)
        h.equal(table.concat(engine:tokenize(PROMPT), ","), table.concat(ids, ","), "the same tokens for the prompt")
        if parallelMain then
            w.state.parallel = true
            w.mock.task.desynchronize()
        end
        h.equal(table.concat(engine:generate(ids, 10), ","), want)
        if parallelMain then
            -- Actors are destroyed in the serial phase.
            w.mock.task.synchronize()
            w.state.parallel = false
        end
        if workers > 0 then
            const sent, bytes = w.mock.messages()
            h.truthy(sent > 0 and bytes > 0, "the workers got messages")
        end
        engine:close()
    end)
end

return function(h: harness.Harness)
    h.case("the bundle rewrites requires and const, and leaves no @ path", function()
        h.equal(
            bundle.rewrite('const a = require("@src/ops/norm")\nconst b = require("./host")', { "roblox", "worker" }),
            "local a = require(script.Parent.Parent.ops.norm)\nlocal b = require(script.Parent.Parent.roblox.host)"
        )
        h.equal(
            bundle.rewrite('const x = require("./sibling")', { "sibling2" }),
            "local x = require(script.Parent.sibling)"
        )
        h.equal(bundle.rewrite('const m = require("@src/host")', {}), "local m = require(script.host)")
        -- In an init module, the module of its folder, `@self/child` is a child of the script.
        h.equal(
            bundle.rewrite('const a = require("@self/host")\nconst b = require("@src/ops/norm")', { "roblox" }),
            "local a = require(script.host)\nlocal b = require(script.Parent.ops.norm)"
        )
        const modules = bundle.modules()
        h.truthy(#modules > 40, "the whole source is there")
        for _, module in modules do
            h.truthy(string.find(module.source, 'require("@', 1, true) == nil, `{module.file} has an @ require`)
            h.truthy(string.find(module.source, "\nconst ", 1, true) == nil, `{module.file} has a const`)
            h.truthy(string.find(module.source, "^const ") == nil, `{module.file} starts with a const`)
        end
    end)

    h.case("every bundled module loads in the model and the instance paths resolve", function()
        const w = worlds.world()
        worlds.run(w, function()
            local count = 0
            const function visit(inst: any)
                if inst.ClassName == "ModuleScript" and inst.Source ~= nil then
                    w.mock.require(inst)
                    count += 1
                end
                for _, child in inst._children do
                    visit(child)
                end
            end
            visit(w.root)
            h.truthy(count > 40)
            h.equal(w.place.script.ClassName, "Script")
        end)
    end)

    h.case("every bundled module is under the ModuleScript limit, the attention in one module per head size", function()
        const modules = bundle.modules()
        h.equal(table.concat(bundle.oversize(modules), ", "), "")
        h.equal(bundle.LIMIT, 199997)
        local biggest = 0
        local perSize = 0
        for _, module in modules do
            biggest = math.max(biggest, #module.source)
            if string.match(module.file, "attention_unrolled_%d+%.luau$") then
                perSize += 1
            end
        end
        h.truthy(biggest > 100000, "the largest module is the head size that needed the split")
        h.equal(perSize, #attentionGen.LANES, "a module per head size")
        h.equal(
            #bundle.oversize({ { path = { "a" }, file = "src/a.luau", source = string.rep("x", bundle.LIMIT + 1) } }),
            1
        )
        h.equal(
            #bundle.oversize({ { path = { "a" }, file = "src/a.luau", source = string.rep("x", bundle.LIMIT) } }),
            0
        )
    end)

    h.case("the attention aggregate gathers the module of each head size under byLanes and q8ByLanes", function()
        const w = worlds.world()
        worlds.run(w, function()
            const aggregate: any = w.mock.require(w.root.ops.generated)
            for _, lanes in attentionGen.LANES do
                h.equal(type(aggregate.byLanes[lanes]), "function", `byLanes[{lanes}]`)
                const module: any = w.mock.require(w.root.ops.generated:FindFirstChild(`attention_unrolled_{lanes}`))
                h.truthy(aggregate.byLanes[lanes] == module, `byLanes[{lanes}] is the module of its size`)
            end
            for _, lanes in attentionQ8Gen.LANES do
                h.equal(type(aggregate.q8ByLanes[lanes]), "function", `q8ByLanes[{lanes}]`)
                const module: any = w.mock.require(w.root.ops.generated:FindFirstChild(`attention_q8_unrolled_{lanes}`))
                h.truthy(aggregate.q8ByLanes[lanes] == module, `q8ByLanes[{lanes}] is the module of its size`)
            end
        end)
    end)

    h.case("a chunked source keeps the last four chunks, so two interleaved streams load each chunk once", function()
        const loads: { number } = {}
        const source = weights.fromChunks(10 * 16, 16, function(index: number): buffer
            table.insert(loads, index)
            return buffer.create(16)
        end)
        const dst = buffer.create(1)
        for round = 1, 3 do
            for _, index in { 0, 9 } do
                source.read(index * 16 + round, 1, dst, 0)
            end
        end
        h.equal(table.concat(loads, ","), "0,9", "two streams far apart")
        table.clear(loads)
        for index = 2, 5 do
            source.read(index * 16, 1, dst, 0)
        end
        for index = 2, 5 do
            source.read(index * 16, 1, dst, 0)
        end
        h.equal(table.concat(loads, ","), "2,3,4,5", "four chunks stay")
        source.read(6 * 16, 1, dst, 0)
        source.read(2 * 16, 1, dst, 0)
        h.equal(table.concat(loads, ","), "2,3,4,5,6,2", "a fifth chunk pushes out the oldest")
    end)

    h.case("base64 text decodes to the bytes it encodes, padding or not", function()
        for length = 0, 40 do
            const data = buffer.create(length)
            for i = 0, length - 1 do
                buffer.writeu8(data, i, (i * 37 + length * 11) % 256)
            end
            const text = base64.encode(buffer.tostring(data))
            h.equal(buffer.tostring(weights.decodeBase64(text)), buffer.tostring(data), `length {length}`)
            h.equal(
                buffer.tostring(weights.decodeBase64(string.gsub(text, "=+$", ""))),
                buffer.tostring(data),
                `length {length} without padding`
            )
        end
    end)

    h.case("a chunked source reads across chunk edges as the file does", function()
        const data = fs.readFileToString("tests/fixtures/tiny-llama/model.olelo")
        const w = worlds.world()
        const folder = worlds.chunkFolder(w, data)
        const source = weights.fromModules(folder, #data, worlds.CHUNK_BYTES, w.mock.require)
        h.equal(source.size(), #data)
        for _, span in
            {
                { 0, 100 },
                { worlds.CHUNK_BYTES - 7, 20 },
                { worlds.CHUNK_BYTES * 2 - 1, worlds.CHUNK_BYTES + 5 },
                { #data - 9, 9 },
            }
        do
            const offset, length = span[1], span[2]
            if offset + length <= #data then
                const dst = buffer.create(length + 3)
                source.read(offset, length, dst, 3)
                h.equal(buffer.readstring(dst, 3, length), string.sub(data, offset + 1, offset + length))
            end
        end
        h.fails(function()
            source.read(#data - 2, 3, buffer.create(3), 0)
        end)
        h.fails(function()
            source.read(0, 1, buffer.create(1), 0)
            weights.fromModules(w.mock.new("Folder"), 10, 5, w.mock.require).read(0, 1, buffer.create(1), 0)
        end)
    end)

    h.case("a paced source waits once its slice of work has passed and reads what the source holds", function()
        const data = fs.readFileToString("tests/fixtures/tiny-llama/model.olelo")
        const w = worlds.world()
        const source = weights.fromModules(worlds.chunkFolder(w, data), #data, worlds.CHUNK_BYTES, w.mock.require)
        local now, waits, reads = 0, 0, 0
        const paced = weights.paced(source, function()
            return now
        end, function()
            waits += 1
            return 0
        end, 100)
        const dst = buffer.create(#data)
        for offset = 0, #data - 1, 1000 do
            now += 40
            reads += 1
            paced.read(offset, math.min(1000, #data - offset), dst, offset)
        end
        h.equal(buffer.tostring(dst), data)
        h.equal(paced.size(), #data)
        h.equal(waits, reads // 3)
    end)

    h.case("an HTTP source asks for byte ranges and reads what comes back", function()
        const data = fs.readFileToString("tests/fixtures/tiny-llama/model.olelo")
        const w = worlds.world()
        const requests: { string } = {}
        w.mock.setHttp(function(request: { [string]: any })
            table.insert(requests, request.Headers.Range)
            const first, last = string.match(request.Headers.Range, "bytes=(%d+)%-(%d+)")
            return {
                Success = true,
                StatusCode = 206,
                Body = string.sub(data, (tonumber(first) :: number) + 1, (tonumber(last) :: number) + 1),
            }
        end)
        const http = w.mock.game:GetService("HttpService")
        const source = weights.fromHttp(http, "https://example.invalid/model.olelo", #data, 4096)
        const dst = buffer.create(10000)
        source.read(4000, 10000, dst, 0)
        h.equal(buffer.readstring(dst, 0, 10000), string.sub(data, 4001, 14000))
        h.equal(table.concat(requests, " "), "bytes=0-4095 bytes=4096-8191 bytes=8192-12287 bytes=12288-16383")
        -- A server that ignores the range answers 200 with the whole file.
        w.mock.setHttp(function()
            return { Success = true, StatusCode = 200, Body = data }
        end)
        const whole = weights.fromHttp(http, "https://example.invalid/model.olelo", #data, 4096)
        whole.read(5000, 100, dst, 0)
        h.equal(buffer.readstring(dst, 0, 100), string.sub(data, 5001, 5100))
        w.mock.setHttp(function()
            return { Success = false, StatusCode = 403, Body = "" }
        end)
        h.fails(function()
            weights.fromHttp(http, "https://example.invalid/model.olelo", #data, 4096).read(0, 1, dst, 0)
        end)
    end)

    h.case("the engine over chunk ModuleScripts gives the tokens of the engine over a file", function()
        compare(h, "tests/fixtures/wide-llama/model-q8.olelo", 0, 0)
    end)

    for _, workers in { 1, 2, 3 } do
        h.case(`the engine over {workers} Actors gives the tokens of the single VM`, function()
            compare(h, "tests/fixtures/wide-llama/model-q8.olelo", workers, 0)
        end)
    end

    h.case("the Actors work with messages that take frames to arrive", function()
        compare(h, "tests/fixtures/wide-qwen3/model-q4.olelo", 2, 2)
    end)

    h.case("a worker answers an op sent before load with the error, and the Actors keep VMs of their own", function()
        const w = worlds.world()
        worlds.run(w, function()
            const workers: any = w.mock.require(w.root.roblox.workers)
            const transports = workers.spawn(w.platform, w.place, 2)
            const future = transports[1].call("stats")
            h.fails(function()
                transports[1].join({ future })
            end)
            const unknown = transports[2].call("nope")
            h.fails(function()
                transports[2].join({ unknown })
            end)
            for _, t in transports do
                t.close()
            end
            h.equal(#w.place.parent._children, 0, "closed Actors leave the tree")
        end)
    end)

    h.case("the model empties a module over the limit, so requiring it fails as in Studio", function()
        const w = worlds.world()
        worlds.run(w, function()
            const fits = w.mock.new("ModuleScript")
            fits.Source = `return "{string.rep("x", bundle.LIMIT - 10)}"`
            h.truthy(#fits.Source > 0)
            h.equal(#w.mock.require(fits), bundle.LIMIT - 10)
            const over = w.mock.new("ModuleScript")
            over.Source = `return "{string.rep("x", bundle.LIMIT)}"`
            h.equal(over.Source, "")
            h.fails(function()
                w.mock.require(over)
            end, "an empty module returns no value")
        end)
    end)

    h.case("the model refuses in the parallel phase what Roblox refuses there", function()
        const w = worlds.world()
        worlds.run(w, function()
            const module = w.mock.new("ModuleScript")
            module.Source = "return 1"
            const http = w.mock.game:GetService("HttpService")
            h.equal(w.mock.phase(), "serial")
            w.mock.task.desynchronize()
            h.equal(w.mock.phase(), "parallel")
            h.fails(function()
                w.mock.require(module)
            end, "require")
            h.fails(function()
                http:RequestAsync({ Url = "https://example.invalid" })
            end, "RequestAsync")
            h.fails(function()
                w.place.main:SetAttribute("a", 1)
            end, "SetAttribute")
            h.fails(function()
                w.place.main:Clone():Destroy()
            end, "Destroy")
            w.mock.task.synchronize()
            h.equal(w.mock.require(module), 1)
            h.equal(w.mock.phase(), "serial")
        end)
    end)

    h.case("BindToMessage works only for a script under the Actor", function()
        const w = worlds.world()
        local message = ""
        -- The main code runs as no script of the Actor.
        w.mock.run(function()
            const workers: any = w.mock.require(w.root.roblox.workers)
            const ok, err = pcall(workers.spawn, w.platform, w.place, 1)
            h.equal(ok, false)
            message = tostring(err)
        end)
        h.truthy(string.find(message, "descendant of the Actor", 1, true) ~= nil, message)
    end)

    h.case("spawn waits for each worker to bind, so no message is dropped", function()
        const w = worlds.world()
        worlds.run(w, function()
            -- A message sent as the Actor is parented arrives before its script has bound the topic.
            const early = w.place.new("Actor")
            const root = w.place.new("ObjectValue")
            root.Name = "Root"
            root.Value = w.root
            root.Parent = early
            w.place.script:Clone().Parent = early
            early.Parent = w.place.parent
            early:SendMessage("hello", w.place.main)
            w.mock.task.wait()
            w.mock.task.wait()
            h.equal(w.mock.dropped(), 1, "the model drops a message sent before the topic is bound")
            h.equal(early:GetAttribute("ready"), true, "the worker script ran and bound its topics")
            const workers: any = w.mock.require(w.root.roblox.workers)
            const transports = workers.spawn(w.platform, w.place, 3)
            h.equal(w.mock.dropped(), 1, "spawn lost no message")
            for _, actor in w.place.parent._children do
                h.equal(actor:GetAttribute("ready"), true)
            end
            for _, t in transports do
                t.close()
            end
        end)
    end)

    h.case("spawn gives up on a worker that never binds", function()
        const w = worlds.world()
        const silent = w.mock.new("Script")
        silent.Name = "silent"
        silent.Source = "return nil"
        const place = table.clone(w.place)
        place.script = silent
        local now = 0
        const platform = table.clone(w.platform)
        platform.clock = function(): number
            now += 5
            return now
        end
        local message = ""
        worlds.run(w, function()
            const workers: any = w.mock.require(w.root.roblox.workers)
            const ok, err = pcall(workers.spawn, platform, place, 1)
            h.equal(ok, false)
            message = tostring(err)
        end)
        h.truthy(string.find(message, "did not bind its messages", 1, true) ~= nil, message)
    end)

    h.case("load goes on the serial topic and the other ops on the parallel one", function()
        const data = fs.readFileToString("tests/fixtures/wide-llama/model-q8.olelo")
        const w = worlds.world()
        const sent: { string } = {}
        w.place.new = function(className: string): any
            const inst = w.mock.new(className)
            if className == "Actor" then
                const send = inst.SendMessage
                inst.SendMessage = function(self: any, topic: string, ...: any)
                    table.insert(sent, if topic == "hello" then topic else `{topic}:{(select(2, ...))}`)
                    send(self, topic, ...)
                end
            end
            return inst
        end
        worlds.run(w, function()
            const engine = worlds.engineOver(w, data, 1)
            engine:generate(engine:tokenize("hello"), 3)
            engine:close()
        end)
        const function has(entry: string): boolean
            return table.find(sent, entry) ~= nil
        end
        h.truthy(has("sreq:load"), "load is serial")
        h.truthy(has("req:attn") and has("req:mlp") and has("req:logits"), "the passes are parallel")
        h.truthy(not has("req:load") and not has("sreq:attn"), "no op on the other topic")
        h.equal(sent[1], "hello", "hello comes first")
    end)

    h.case("a thread a reply wakes returns to its phase through Platform.restore", function()
        const data = fs.readFileToString("tests/fixtures/wide-llama/model-q8.olelo")
        const w = worlds.world()
        local phase = ""
        worlds.run(w, function()
            const engine = worlds.engineOver(w, data, 1)
            phase = w.mock.phase()
            engine:close()
        end)
        h.equal(phase, "serial")
    end)

    h.case(
        "without Platform.restore the thread a reply wakes stays in the parallel phase and require is refused",
        function()
            const data = fs.readFileToString("tests/fixtures/wide-llama/model-q8.olelo")
            const w = worlds.world()
            const platform = table.clone(w.platform)
            platform.restore = nil
            h.fails(function()
                worlds.run(w, function()
                    worlds.engineOver(w, data, 1, { platform = platform })
                end)
            end)
        end
    )

    h.case("the Actors work with the main script in the parallel phase", function()
        compare(h, "tests/fixtures/wide-llama/model-q8.olelo", 2, 0, true)
        compare(h, "tests/fixtures/wide-qwen3/model-q4.olelo", 2, 2, true)
    end)

    h.case("the default slice is 0.012 seconds", function()
        const data = fs.readFileToString("tests/fixtures/wide-llama/model-q8.olelo")
        const w = worlds.world()
        worlds.run(w, function()
            const roblox: any = w.mock.require(w.root.roblox)
            h.equal(roblox.SLICE, 0.012)
            const engine = worlds.engineOver(w, data, 0)
            h.equal(engine.core.budget.slice, 0.012)
            const sliced = worlds.engineOver(w, data, 0, { slice = 0.004 })
            h.equal(sliced.core.budget.slice, 0.004)
        end)
    end)

    h.case("a chat on one script with a slice returns its pieces from the iterator, as with onText", function()
        const path = "tests/fixtures/wide-llama/model-q8.olelo"
        const data = fs.readFileToString(path)
        local expected = ""
        for piece in worlds.reference(path):chat({}):send("what is a hash table?", { maxTokens = 12 }) do
            expected ..= piece
        end
        h.truthy(expected ~= "")
        const w = worlds.world()
        worlds.run(w, function()
            -- A slice used up yields to the frame with `task.wait`, which a coroutine of the iterator would take
            -- for the end of a piece.
            const engine = worlds.engineOver(w, data, 0, { slice = 1e-9 })
            const before = w.mock.frame()
            local viaIterator = ""
            for piece in engine:chat({}):send("what is a hash table?", { maxTokens = 12 }) do
                viaIterator ..= piece
            end
            h.equal(viaIterator, expected)
            h.truthy(w.mock.frame() > before, "the pass yielded to the frame")
            local viaText = ""
            engine:chat({}):send("what is a hash table?", {
                maxTokens = 12,
                onText = function(piece: string)
                    viaText ..= piece
                end,
            })
            h.equal(viaText, expected)
        end)
    end)

    for _, workers in { 0, 2 } do
        h.case(`the example Script starts under an Actor and answers a prompt with {workers} workers`, function()
            const path = "tests/fixtures/wide-llama/model-q8.olelo"
            const data = fs.readFileToString(path)
            local expected = ""
            for piece in worlds.reference(path):chat({}):send("What is a hash table?", { maxTokens = 128 }) do
                expected ..= piece
            end
            h.truthy(expected ~= "")
            const w = worlds.world()
            worlds.chunkFolder(w, data)
            local source = fs.readFileToString("examples/roblox/server.server.luau")
            for _, change in
                {
                    { "local SIZE = %d+", `local SIZE = {#data}` },
                    { "local WORKERS = %d+", `local WORKERS = {workers}` },
                }
            do
                const changed, count = string.gsub(source, change[1], change[2])
                h.equal(count, 1, change[1])
                source = changed
            end
            const example = w.mock.new("Script")
            example.Name = "server"
            example.Source = source
            example.Parent = w.mock.game:GetService("ServerScriptService")
            w.mock.run(function()
                -- Runs the Script in ServerScriptService, which starts itself again under an Actor.
                w.mock.require(example)
                for _ = 1, 100000 do
                    if table.find(w.mock.printed, expected) ~= nil then
                        break
                    end
                    w.mock.task.wait()
                end
                h.truthy(table.find(w.mock.printed, expected) ~= nil, "the reply of the single VM was printed")
            end)
        end)
    end

    h.case("a chat over Actors answers as a chat in one VM does", function()
        const path = "tests/fixtures/wide-llama/model-q8.olelo"
        const data = fs.readFileToString(path)
        local expected = ""
        for piece in worlds.reference(path):chat({}):send("what is a hash table?", { maxTokens = 12 }) do
            expected ..= piece
        end
        const w = worlds.world()
        worlds.run(w, function()
            const parallel = worlds.engineOver(w, data, 2)
            local out = ""
            for piece in parallel:chat({}):send("what is a hash table?", { maxTokens = 12 }) do
                out ..= piece
            end
            h.equal(out, expected)
            parallel:close()
        end)
    end)
end
