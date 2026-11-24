-- Prefix sharing: conversations that start with the same system prompt prefill it once, on every runner and KV format,
-- and answer exactly as they do without sharing.
const engineApi = require("@src/api/engine")
const harness = require("@tests/harness")
const hostApi = require("@src/host")
const lute = require("@backends/lute")
const probe = require("@src/kernels/probe")
const transport = require("@src/parallel/transport")
const apiTypes = require("@src/api/types")

const MODEL = "tests/fixtures/tiny-llama/model.olelo"
const WIDE = "tests/fixtures/wide-llama/model-q8.olelo"
const SYSTEM = "You are a careful assistant who answers briefly and plainly, and never makes things up."
const OTHER = "You answer every question with a short poem about the sea and the wind."

type Setup = { kv: string?, prefixes: number?, workers: number? }

const function open(path: string, options: Setup?): apiTypes.Engine
    const host = lute.host()
    const caps = { vectorRead = probe.run(host.clock).vectorRead }
    const o: Setup = options or {}
    local transports: { hostApi.Transport }? = nil
    if o.workers ~= nil then
        transports =
            transport.inProcessWorkers(lute.fileSource(path), host.clock, o.workers :: number, 192, 8, caps, o.kv)
    end
    return engineApi.new({
        host = host,
        source = lute.fileSource(path),
        context = 192,
        chunk = 8,
        caps = caps,
        kv = o.kv,
        prefixes = o.prefixes,
        transports = transports,
    })
end

const function reply(chat: apiTypes.Chat, text: string, maxTokens: number): string
    const parts: { string } = {}
    for piece in chat:send(text, { maxTokens = maxTokens }) do
        table.insert(parts, piece)
    end
    return table.concat(parts)
end

const SETUPS: { { name: string, path: string, options: Setup } } = {
    { name = "one VM, f32 cache", path = MODEL, options = {} },
    { name = "one VM, Q8 cache", path = MODEL, options = { kv = "q8" } },
    { name = "3 workers, Q8 cache", path = WIDE, options = { kv = "q8", workers = 3 } },
    { name = "2 workers, f32 cache", path = WIDE, options = { workers = 2 } },
}

return function(h: harness.Harness)
    for _, setup in SETUPS do
        h.case(
            `the second conversation with a system prompt copies its cache and answers the same, {setup.name}`,
            function()
                const sharing = open(setup.path, setup.options)
                const alone = open(setup.path, { kv = setup.options.kv, prefixes = 0, workers = setup.options.workers })
                const first = sharing:chat({ system = SYSTEM })
                const second = sharing:chat({ system = SYSTEM })
                const plain = alone:chat({ system = SYSTEM })
                const a = reply(first, "hello there", 8)
                h.equal(first:stats().cachedTokens, 0, "the first conversation prefills everything")
                const b = reply(second, "hello there", 8)
                const stats = second:stats()
                h.truthy(stats.cachedTokens >= 8, `the second conversation copied {stats.cachedTokens} tokens`)
                h.truthy(stats.cachedTokens < stats.promptTokens, "the user's own tokens are still prefilled")
                h.equal(a, b)
                h.equal(reply(plain, "hello there", 8), b, "sharing does not change the answer")
                -- A second turn goes on from the conversation's own cache.
                h.equal(reply(second, "what is a hash table?", 8), reply(plain, "what is a hash table?", 8))
                h.equal(second:stats().cachedTokens, 0, "a later turn takes nothing from the prefix")
                sharing:close()
                alone:close()
            end
        )
    end

    h.case("another system prompt does not hit, and the store drops its least recently used prefix", function()
        const e = open(MODEL, { prefixes = 1 })
        reply(e:chat({ system = SYSTEM }), "hello", 3)
        const miss = e:chat({ system = OTHER })
        reply(miss, "hello", 3)
        h.equal(miss:stats().cachedTokens, 0)
        -- The store holds one prefix, the second system prompt replaced the first.
        const back = e:chat({ system = SYSTEM })
        reply(back, "hello", 3)
        h.equal(back:stats().cachedTokens, 0)
        const again = e:chat({ system = SYSTEM })
        reply(again, "hello", 3)
        h.truthy(again:stats().cachedTokens >= 8)
    end)

    h.case("a conversation rebuilt from a snapshot starts from the shared prefix", function()
        const e = open(MODEL)
        const live = e:chat({ system = SYSTEM })
        reply(live, "hello there", 6)
        reply(live, "what is a hash table?", 6)
        const state = live:snapshot()
        const rebuilt = e:chat({ system = SYSTEM })
        const turn = rebuilt:resume(state)
        local prefill = 0
        while not turn:step().done do
            prefill += 1
        end
        -- Only the tokens after the system prompt are run.
        const whole = (#state.tokens - 1 + 7) // 8
        h.truthy(prefill < whole, `{prefill} prefill steps against {whole} without the prefix`)
        h.equal(reply(rebuilt, "thanks", 6), reply(live, "thanks", 6))
    end)

    h.case("an engine with no prefix store works as before", function()
        const e = open(MODEL, { prefixes = 0 })
        h.equal(e.core.prefixes, nil)
        reply(e:chat({ system = SYSTEM }), "hello", 3)
        const chat = e:chat({ system = SYSTEM })
        reply(chat, "hello", 3)
        h.equal(chat:stats().cachedTokens, 0)
    end)
end
