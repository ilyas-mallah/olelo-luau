-- Conversations on the tiny model: cache reuse, streaming forms, stop strings, cancel and limits.
const engineApi = require("@src/api/engine")
const harness = require("@tests/harness")
const lute = require("@backends/lute")
const apiTypes = require("@src/api/types")

const MODEL = "tests/fixtures/tiny-llama/model.olelo"

const function newEngine(context: number?, seed: number?): apiTypes.Engine
    return engineApi.new({
        host = lute.host(),
        source = lute.fileSource(MODEL),
        context = context or 256,
        seed = seed or 1,
        chunk = 8,
    })
end

--- Everything one `send` returns, as one string.
const function reply(chat: apiTypes.Chat, text: string, options: apiTypes.SendOptions?): string
    const parts: { string } = {}
    for piece in chat:send(text, options) do
        table.insert(parts, piece)
    end
    return table.concat(parts)
end

const THINKING = "tests/fixtures/tiny-qwen3/model.olelo"

return function(h: harness.Harness)
    h.case(
        "load keeps the cache a live conversation shares with the template, and continues like a fresh load",
        function()
            const function engine(): apiTypes.Engine
                return engineApi.new({
                    host = lute.host(),
                    source = lute.fileSource(THINKING),
                    context = 256,
                    chunk = 8,
                })
            end
            const live = engine():chat()
            const first = reply(live, "hello", { maxTokens = 6 })
            const second = reply(live, "what is a hash table?", { maxTokens = 6 })
            const messages: { { role: string, content: string } } = {
                { role = "user", content = "hello" },
                { role = "assistant", content = first },
                { role = "user", content = "what is a hash table?" },
                { role = "assistant", content = second },
            }
            const used = live:used()
            const kept = live:load(messages)
            -- The live cache holds the empty think block of every reply, the template of a history message has none, so
            -- the two agree up to the first reply and no further.
            h.truthy(kept > 0 and kept < used, `kept {kept} of {used} positions`)
            const fresh = engine():chat()
            h.equal(fresh:load(messages), 0)
            h.equal(table.concat(fresh:tokens(), ","), table.concat(live:tokens(), ","))
            h.equal(live:used(), fresh:used())
            h.equal(reply(live, "thanks", { maxTokens = 8 }), reply(fresh, "thanks", { maxTokens = 8 }))
        end
    )

    h.case("load of the same messages again keeps every position", function()
        const chat = newEngine():chat()
        const messages = {
            { role = "user", content = "hello" },
            { role = "assistant", content = "hi there" },
        }
        chat:load(messages)
        const used = chat:used()
        h.equal(chat:load(messages), used)
        h.equal(chat:used(), used)
        h.equal(chat:load({}), 0)
        h.equal(chat:used(), 0)
        h.fails(function()
            chat:load({ { role = "user", content = "hello" } })
        end)
    end)

    h.case("a second turn on the cache equals a run from scratch", function()
        const e = newEngine()
        const chat = e:chat()
        reply(chat, "hello", { maxTokens = 6 })
        const before = chat:tokens()
        reply(chat, "what is a hash table?", { maxTokens = 6 })
        const after = chat:tokens()
        const stats = chat:stats()
        -- The prompt of turn two is the pending token plus the template text, which the history holds after `before`.
        const prompt: { number } = table.clone(before)
        for i = #before + 1, #before + stats.promptTokens - 1 do
            table.insert(prompt, after[i])
        end
        const fresh = e:generate(prompt, stats.newTokens)
        for i = 1, stats.newTokens do
            h.equal(fresh[i], after[#prompt + i], `new token {i}`)
        end
    end)

    h.case("onText gives the same text as the iterator", function()
        const a, b = newEngine():chat(), newEngine():chat()
        const viaIterator = reply(a, "hello there", { maxTokens = 12 })
        const parts: { string } = {}
        const iterator = b:send("hello there", {
            maxTokens = 12,
            onText = function(piece: string)
                table.insert(parts, piece)
            end,
        })
        h.equal(iterator(), nil)
        h.equal(table.concat(parts), viaIterator)
        h.truthy(#viaIterator > 0)
    end)

    h.case("stats describe the reply", function()
        const chat = newEngine():chat()
        reply(chat, "hello", { maxTokens = 5 })
        const s = chat:stats()
        h.equal(s.newTokens, 5)
        h.equal(s.finish, "length")
        h.truthy(s.promptTokens > 5 and s.timeToFirstToken > 0 and s.decodeRate > 0)
        h.equal(chat:used(), s.promptTokens + 4, "the last sampled token waits outside the cache")
    end)

    h.case("a stop string ends the reply before it", function()
        const text = reply(newEngine():chat(), "hello", { maxTokens = 16 })
        h.truthy(#text >= 6)
        const stop = string.sub(text, 3, 4)
        const chat = newEngine():chat()
        const cut = reply(chat, "hello", { maxTokens = 16, stop = { stop } })
        const at = string.find(text, stop, 1, true) :: number
        h.equal(cut, string.sub(text, 1, at - 1))
        h.equal(chat:stats().finish, "text")
    end)

    h.case("the stop string that starts first wins, and one that spans several tokens still stops", function()
        const text = reply(newEngine():chat(), "hello", { maxTokens = 16 })
        h.truthy(#text >= 10, `the reply is {#text} bytes`)
        const early = string.sub(text, 3, 5)
        const late = string.sub(text, 8, 10)
        const at = string.find(text, early, 1, true) :: number
        for _, stops in { { early, late }, { late, early } } do
            const chat = newEngine():chat()
            const pieces: { string } = {}
            for piece in chat:send("hello", { maxTokens = 16, stop = stops }) do
                table.insert(pieces, piece)
            end
            h.equal(table.concat(pieces), string.sub(text, 1, at - 1))
            h.equal(chat:stats().finish, "text")
        end
        -- A stop that never appears holds nothing back at the end.
        h.equal(reply(newEngine():chat(), "hello", { maxTokens = 16, stop = { "\1\2\3" } }), text)
    end)

    h.case("ignoreStop runs to the token limit", function()
        const chat = newEngine():chat()
        reply(chat, "hello", { maxTokens = 24, ignoreStop = true })
        const s = chat:stats()
        h.equal(s.newTokens, 24)
        h.equal(s.finish, "length")
    end)

    h.case("cancel ends the reply at the next token", function()
        const chat = newEngine():chat()
        local pieces = 0
        chat:send("hello", {
            maxTokens = 50,
            onText = function(_piece: string)
                pieces += 1
                chat:cancel()
            end,
        })
        h.equal(chat:stats().finish, "cancelled")
        h.truthy(chat:stats().newTokens < 5)
    end)

    h.case("a prompt that does not fit is refused with the numbers", function()
        const chat = newEngine(32):chat()
        const ok, err = pcall(function()
            reply(chat, string.rep("the quick brown fox ", 10))
        end)
        h.truthy(not ok and string.find(tostring(err), "does not fit", 1, true) ~= nil, tostring(err))
    end)

    h.case("a reply stops when the context is full", function()
        const chat = newEngine(48):chat()
        reply(chat, "hello", { maxTokens = 200 })
        h.equal(chat:stats().finish, "context")
        h.equal(chat:used(), 48)
    end)

    h.case("sampling repeats with a seed and differs between seeds", function()
        const options: apiTypes.SendOptions = { maxTokens = 20, temperature = 1.5 }
        const a = reply(newEngine(nil, 5):chat(), "hello", options)
        const b = reply(newEngine(nil, 5):chat(), "hello", options)
        const c = reply(newEngine(nil, 6):chat(), "hello", options)
        h.equal(a, b)
        h.truthy(a ~= c)
    end)

    h.case("reset forgets the conversation", function()
        const chat = newEngine():chat()
        const first = reply(chat, "hello", { maxTokens = 8 })
        chat:reset()
        h.equal(chat:used(), 0)
        h.equal(reply(chat, "hello", { maxTokens = 8 }), first)
    end)

    h.case("an explicit system message replaces the default", function()
        const e = newEngine()
        const plain, custom = e:chat(), e:chat({ system = "the quick brown fox" })
        reply(plain, "hello", { maxTokens = 1 })
        reply(custom, "hello", { maxTokens = 1 })
        h.truthy(plain:stats().promptTokens ~= custom:stats().promptTokens)
    end)
    h.case("a turn runs in steps and gives the text of send", function()
        const viaSend = reply(newEngine():chat(), "hello there", { maxTokens = 9 })
        const chat = newEngine():chat()
        const turn = chat:begin("hello there", { maxTokens = 9 })
        local prefill, decode, text = 0, 0, ""
        local result = turn:step(3)
        while true do
            if result.kind == "prefill" then
                prefill += 1
                h.truthy(result.tokens >= 1 and result.tokens <= 8, "a prefill step takes one chunk")
                h.equal(#result.pieces, 0)
            else
                decode += 1
                h.truthy(result.tokens <= 3, "a decode step respects its budget")
            end
            text ..= table.concat(result.pieces)
            if result.done then
                break
            end
            result = turn:step(3)
        end
        h.equal(text, viaSend)
        h.equal(prefill, (chat:stats().promptTokens + 7) // 8, "one step per chunk of 8 prompt tokens")
        h.equal(decode, 3, "9 tokens at 3 per step")
        h.equal(chat:stats().finish, "length")
        h.equal(turn:step(3).done, true, "a finished turn stays finished")
    end)

    h.case("cancelling a turn ends it at its next step", function()
        const chat = newEngine():chat()
        const turn = chat:begin("hello", { maxTokens = 50 })
        while turn:step(1).kind == "prefill" do
        end
        turn:cancel()
        h.equal(turn:step(1).done, true)
        h.equal(chat:stats().finish, "cancelled")
    end)

    h.case("a chat takes its own seed", function()
        const e = newEngine(nil, 5)
        const options: apiTypes.SendOptions = { maxTokens = 20, temperature = 1.5 }
        const a = reply(e:chat({ seed = 11 }), "hello", options)
        const b = reply(e:chat({ seed = 11 }), "hello", options)
        const c = reply(e:chat({ seed = 12 }), "hello", options)
        h.equal(a, b)
        h.truthy(a ~= c)
    end)
    h.case("a snapshot rebuilds a conversation that continues token for token", function()
        for _, maxTokens in { 6, 7 } do
            const e = newEngine()
            const live = e:chat()
            reply(live, "hello", { maxTokens = maxTokens })
            reply(live, "what is a hash table?", { maxTokens = maxTokens })
            const state = live:snapshot()
            h.truthy(state.pending, "the last sampled token is not in the cache")
            const rebuilt = e:chat()
            const turn = rebuilt:resume(state)
            local steps = 0
            while not turn:step().done do
                steps += 1
            end
            h.truthy(steps >= 1, "the cache is prefilled in chunks")
            h.equal(rebuilt:used(), live:used())
            h.equal(table.concat(rebuilt:tokens(), ","), table.concat(live:tokens(), ","))
            h.equal(reply(rebuilt, "thanks", { maxTokens = 8 }), reply(live, "thanks", { maxTokens = 8 }))
        end
    end)

    h.case("a refused prompt leaves the conversation as it was", function()
        const chat = newEngine(96):chat()
        reply(chat, "hello", { maxTokens = 6 })
        const before, used = chat:tokens(), chat:used()
        h.fails(function()
            chat:begin(string.rep("the quick brown fox ", 40))
        end)
        h.equal(chat:used(), used)
        h.equal(table.concat(chat:tokens(), ","), table.concat(before, ","))
        h.equal(#reply(chat, "thanks", { maxTokens = 4 }) > 0, true)
    end)
end
