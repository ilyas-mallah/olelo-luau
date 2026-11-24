-- Tensor parallel inference over in process workers against the same model in one VM: the logits of a prompt and of
-- decode steps, rewinds, several sequences, and the engine's greedy continuation. The workers hold a share of the
-- weights each, so the two differ only by the order of the sums.
const data = require("@tests/data")
const engineModule = require("@src/api/engine")
const harness = require("@tests/harness")
const hostApi = require("@src/host")
const lute = require("@backends/lute")
const process = require("@std/process")
const runner = require("@src/model/runner")
const transport = require("@src/parallel/transport")
const apiTypes = require("@src/api/types")
const modelTypes = require("@src/model/types")

const PROMPT =
    "the quick brown fox jumps over the lazy dog and the dog does not mind, what is a hash table? hello. the quick brown fox jumps over the lazy dog and the dog does not mind, what is a hash table? hello"
const CONTEXT = 192

--- An engine on the container at `path`: the whole model in this VM, or with `workers` workers in this VM, or with
--- `vms` worker VMs of the Lute backend.
const function open(path: string, workers: number?, vms: boolean?, kv: string?): apiTypes.Engine
    const host = lute.host()
    const caps = { vectorRead = data.vectorFast() }
    const source = lute.fileSource(path)
    local transports: { hostApi.Transport }? = nil
    if workers ~= nil and vms then
        transports = lute.spawnWorkers(
            path,
            workers :: number,
            { context = CONTEXT, chunk = 32, vectorRead = caps.vectorRead, kv = kv }
        )
    elseif workers ~= nil then
        transports = transport.inProcessWorkers(source, host.clock, workers :: number, CONTEXT, 32, caps, kv)
    end
    return engineModule.new({
        host = host,
        source = source,
        context = CONTEXT,
        caps = caps,
        transports = transports,
        kv = kv,
    })
end

--- The logits of the last token of `ids` run in passes of up to `chunk` tokens on `seq`.
const function prefill(r: modelTypes.Runner, seq: modelTypes.Sequence, ids: { number }, chunk: number): buffer
    local at = 1
    local out: buffer? = nil
    while at <= #ids do
        const count = math.min(chunk, #ids - at + 1)
        out = r:forward(seq, ids, at, count, at + count > #ids, nil)
        at += count
    end
    const copy = buffer.create(r.vocab * 4)
    buffer.copy(copy, 0, out :: buffer, 0, r.vocab * 4)
    return copy
end

const function argmax(b: buffer, vocab: number): number
    local best, id = -math.huge, 0
    for i = 0, vocab - 1 do
        const v = buffer.readf32(b, i * 4)
        if v > best then
            best, id = v, i
        end
    end
    return id
end

--- The largest difference of two logit rows relative to the largest logit.
const function relativeDifference(a: buffer, b: buffer, vocab: number): number
    local worst, size = 0, 0
    for i = 0, vocab - 1 do
        const x, y = buffer.readf32(a, i * 4), buffer.readf32(b, i * 4)
        worst = math.max(worst, math.abs(x - y))
        size = math.max(size, math.abs(x))
    end
    return worst / size
end

--- Runs the same prompt and decode steps on both engines and compares the logits at each step.
const function compare(h: harness.Harness, path: string, workers: number, tolerance: number, vms: boolean?, kv: string?)
    const single = open(path, nil, nil, kv)
    const parallel = open(path, workers, vms, kv)
    const ids = single:tokenize(PROMPT)
    h.truthy(#ids > 16, "the prompt takes more than one pass")
    const a, b = single.core.runner, parallel.core.runner
    const vocab = a.vocab
    const sa, sb = a:open(), b:open()
    local la, lb = prefill(a, sa, ids, 16), prefill(b, sb, ids, 16)
    h.truthy(
        relativeDifference(la, lb, vocab) <= tolerance,
        `prompt logits differ by {relativeDifference(la, lb, vocab)}`
    )
    for step = 1, 4 do
        const id = argmax(la, vocab)
        h.equal(argmax(lb, vocab), id, `step {step} picks the same token`)
        la = prefill(a, sa, { id }, 1)
        lb = prefill(b, sb, { id }, 1)
        h.truthy(
            relativeDifference(la, lb, vocab) <= tolerance,
            `step {step} logits differ by {relativeDifference(la, lb, vocab)}`
        )
    end
    h.equal(sb.length, sa.length)
    parallel:close()
end

return function(h: harness.Harness)
    for _, workers in { 1, 2, 3, 4 } do
        h.case(`{workers} workers hold tiny-llama f32 and give its logits`, function()
            compare(h, "tests/fixtures/tiny-llama/model.olelo", workers, if workers == 1 then 1e-6 else 1e-4)
        end)
    end

    h.case("the heads of tiny-qwen2 divided among 6 workers, one head each", function()
        -- 6 workers need 6 groups of MLP columns, which the tiny MLP (100 columns) does not have, so 4 is the most.
        compare(h, "tests/fixtures/tiny-qwen2/model.olelo", 4, 1e-4)
    end)

    for _, kind in { "llama", "qwen2", "qwen3", "gemma3" } do
        for _, format in { "", "-bf16", "-q8", "-q8v", "-q4" } do
            if format == "-q8v" and not data.vectorFast() then
                h.skip(`wide-{kind}-q8v`, "the vector layout needs the fast vector read")
            else
                h.case(`wide-{kind}{format} over 2, 3 and 4 workers`, function()
                    for _, workers in { 2, 3, 4 } do
                        compare(h, `tests/fixtures/wide-{kind}/model{format}.olelo`, workers, 1e-4)
                    end
                end)
            end
        end
    end

    h.case("one worker's logits are the single VM's logits bit for bit", function()
        compare(h, "tests/fixtures/wide-gemma3/model-q8.olelo", 1, 0)
    end)

    if data.vectorFast() then
        h.case("one worker's logits are the single VM's logits bit for bit in the vector layout", function()
            compare(h, "tests/fixtures/wide-gemma3/model-q8v.olelo", 1, 0)
        end)
    end
    -- Every worker quantizes the rows of its own KV heads, so the Q8 cache does not depend on the division.
    h.case("workers on the Q8 KV cache give the single VM's logits on the Q8 KV cache", function()
        compare(h, "tests/fixtures/wide-gemma3/model-q8.olelo", 1, 0, nil, "q8")
        for _, kind in { "llama", "qwen2", "qwen3", "gemma3" } do
            for _, workers in { 2, 3, 4 } do
                compare(h, `tests/fixtures/wide-{kind}/model-q8.olelo`, workers, 1e-4, nil, "q8")
            end
        end
        compare(h, "tests/fixtures/wide-llama/model.olelo", 3, 1e-4, nil, "q8")
    end)

    h.case("a rewound sequence on workers with the Q8 KV cache continues as a fresh one", function()
        const parallel = open("tests/fixtures/wide-llama/model-q8.olelo", 3, nil, "q8")
        const r = parallel.core.runner
        const ids = parallel:tokenize(PROMPT)
        const first = r:open()
        const reference = prefill(r, first, ids, 32)
        r:rewind(first, 10)
        const rest: { number } = {}
        for i = 11, #ids do
            table.insert(rest, ids[i])
        end
        const again = prefill(r, first, rest, 32)
        h.truthy(relativeDifference(reference, again, r.vocab) <= 1e-6, "the same logits after the rewind")
        parallel:close()
    end)

    h.case("the engine's greedy continuation is the same over workers", function()
        const single = open("tests/fixtures/wide-qwen3/model-q4.olelo")
        const parallel = open("tests/fixtures/wide-qwen3/model-q4.olelo", 3)
        const ids = single:tokenize(PROMPT)
        h.equal(table.concat(parallel:generate(ids, 12), ","), table.concat(single:generate(ids, 12), ","))
        parallel:close()
    end)

    h.case("a rewound sequence continues as a fresh one, and sequences do not disturb each other", function()
        const parallel = open("tests/fixtures/wide-llama/model-q8.olelo", 3)
        const r = parallel.core.runner
        const ids = parallel:tokenize(PROMPT)
        const first, second = r:open(), r:open()
        const reference = prefill(r, first, ids, 32)
        -- The second sequence takes another prompt in between.
        prefill(r, second, { ids[1], ids[2], ids[3] }, 32)
        h.equal(first.length, #ids)
        -- Rewind the first to 10 positions and run the rest again.
        r:rewind(first, 10)
        h.equal(first.length, 10)
        const rest: { number } = {}
        for i = 11, #ids do
            table.insert(rest, ids[i])
        end
        const again = prefill(r, first, rest, 32)
        h.truthy(relativeDifference(reference, again, r.vocab) <= 1e-6, "the same logits after the rewind")
        h.equal(second.length, 3)
        r:close(second)
        parallel:close()
    end)

    h.case("a worker refuses a request that does not start where its cache ends", function()
        const parallel = open("tests/fixtures/wide-llama/model.olelo", 2)
        const r = parallel.core.runner
        const seq = r:open()
        prefill(r, seq, { 1, 2, 3 }, 32)
        seq.length = 1
        h.fails(function()
            prefill(r, seq, { 4 }, 32)
        end)
        parallel:close()
    end)

    h.case("a chat over workers answers as a chat in one VM does", function()
        const single = open("tests/fixtures/wide-llama/model-q8.olelo")
        const parallel = open("tests/fixtures/wide-llama/model-q8.olelo", 2)
        const reply = function(e: apiTypes.Engine): string
            const chat = e:chat({})
            local out = ""
            for piece in chat:send("what is a hash table?", { maxTokens = 12 }) do
                out ..= piece
            end
            out ..= "|"
            for piece in chat:send("thanks", { maxTokens = 8 }) do
                out ..= piece
            end
            return out
        end
        h.equal(reply(parallel), reply(single))
        parallel:close()
    end)

    -- A lute that crashes under repeated calls between VMs (the Ref release race) must not run these.
    if process.env.OLELO_VM ~= "1" then
        h.skip("worker VMs", "OLELO_VM is not 1, which says the lute has the fix for the vm.create crash")
        return
    end

    h.case("worker VMs give the logits of the single VM", function()
        for _, workers in { 1, 2, 3, 4 } do
            compare(h, "tests/fixtures/wide-llama/model-q8.olelo", workers, if workers == 1 then 0 else 1e-4, true)
        end
        compare(h, "tests/fixtures/wide-gemma3/model-q4.olelo", 3, 1e-4, true)
        compare(h, "tests/fixtures/tiny-qwen3/model.olelo", 4, 1e-4, true)
    end)

    h.case("worker VMs hold the Q8 KV cache of their heads and give the single VM's logits", function()
        compare(h, "tests/fixtures/wide-llama/model-q8.olelo", 1, 0, true, "q8")
        compare(h, "tests/fixtures/wide-llama/model-q8.olelo", 3, 1e-4, true, "q8")
    end)

    h.case("a chat over worker VMs answers as a chat in one VM does", function()
        const single = open("tests/fixtures/wide-llama/model-q8.olelo")
        const parallel = open("tests/fixtures/wide-llama/model-q8.olelo", 3, true)
        local out, expected = "", ""
        for piece in parallel:chat({}):send("what is a hash table?", { maxTokens = 12 }) do
            out ..= piece
        end
        for piece in single:chat({}):send("what is a hash table?", { maxTokens = 12 }) do
            expected ..= piece
        end
        h.equal(out, expected)
        parallel:close()
    end)

    h.case("the stats of a worker VM count its requests and bytes", function()
        const parallel = open("tests/fixtures/wide-llama/model-q8.olelo", 2, true)
        const ids = parallel:tokenize("hello there")
        parallel:generate(ids, 3)
        const stats = (parallel.core.runner :: any):stats()
        h.truthy(stats.exchanges > 0 and stats.waitSeconds > 0)
        h.equal(#stats.workers, 2)
        -- One pass for the prompt and one per token, over 2 layers.
        h.equal(stats.workers[1].attnCalls, 2 * 4)
        h.truthy(stats.workers[1].weightBytes > 0 and stats.workers[1].bytesIn > 0 and stats.workers[1].bytesOut > 0)
        parallel:close()
    end)
end
