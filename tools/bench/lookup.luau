-- What a prompt lookup drafter would accept. Generates a greedy reply to each prompt, then replays the tokens through the
-- drafter offline: at each position it proposes the tokens that followed the last earlier match of the latest n tokens,
-- and the verification pass accepts the draft's longest prefix that equals the real continuation (greedy output is the
-- same with and without speculation). Prints passes, tokens per pass and, from the measured pass costs, the speedup.
--
--   lute run tools/bench/lookup.luau <model.olelo> [tokens=96] [costs=1:60,2:100,...]
--
-- `costs` is the milliseconds of a pass of k tokens with every logits row, `tools/bench/profile.luau` BATCH lines.
-- Costs of other k are interpolated, and extended past the last one at the slope of the last two.
const engine = require("@src/api/engine")
const lute = require("@backends/lute")
const process = require("@std/process")

const path = process.args[2]
if path == nil then
    print("usage: lute run tools/bench/lookup.luau <model.olelo> [tokens] [costs=1:60,2:100]")
    process.exit(2)
end
local tokens = 96
const costK: { number } = {}
const costMs: { number } = {}
for i = 3, #process.args do
    const arg = process.args[i]
    const list = string.match(arg, "^costs=(.+)$")
    if list ~= nil then
        for k, ms in string.gmatch(list, "(%d+):([%d%.]+)") do
            table.insert(costK, tonumber(k) :: number)
            table.insert(costMs, tonumber(ms) :: number)
        end
    elseif tonumber(arg) ~= nil then
        tokens = tonumber(arg) :: number
    end
end

const PARAGRAPH = "Luau is a fast, small, safe, gradually typed embeddable scripting language derived from Lua. It is "
    .. "used by Roblox to build games and by many other projects as a scripting layer. Its virtual machine can compile "
    .. "hot functions to native code, which makes numeric loops several times faster than the interpreter, and its "
    .. "buffer type gives direct access to raw memory without the overhead of tables."

const PROMPTS: { { name: string, text: string } } = {
    { name = "summarize", text = "Summarize this paragraph in one sentence: " .. PARAGRAPH },
    { name = "repeat", text = "Repeat this text exactly, word for word: " .. PARAGRAPH },
    {
        name = "rewrite",
        text = "Rewrite this paragraph as a list of bullet points, keeping its wording: " .. PARAGRAPH,
    },
    {
        name = "extract",
        text = "List every technical term from this paragraph, one per line: " .. PARAGRAPH,
    },
    { name = "code", text = "Write a Luau function that returns the sum of the squares of a table of numbers." },
    { name = "explain", text = "Explain in two short paragraphs what a tensor is and why neural networks use them." },
}

--- Milliseconds of a pass of `k` tokens.
const function cost(k: number): number
    local n = #costK
    if n == 0 then
        return k
    end
    if k <= costK[1] then
        return costMs[1] * k / costK[1]
    end
    for i = 2, n do
        if k <= costK[i] then
            const f = (k - costK[i - 1]) / (costK[i] - costK[i - 1])
            return costMs[i - 1] + f * (costMs[i] - costMs[i - 1])
        end
    end
    if n == 1 then
        return costMs[1] * k / costK[1]
    end
    return costMs[n] + (k - costK[n]) * (costMs[n] - costMs[n - 1]) / (costK[n] - costK[n - 1])
end

--- The tokens that followed the latest earlier occurrence of the last `n` tokens of `t[1..pos]`, at most `d` of them,
--- trying n from `nmax` down to `nmin`.
const function propose(t: { number }, pos: number, nmax: number, nmin: number, d: number): { number }
    for n = nmax, nmin, -1 do
        if pos >= n + 1 then
            for start = pos - n, 1, -1 do
                local same = true
                for j = 1, n do
                    if t[start + j - 1] ~= t[pos - n + j] then
                        same = false
                        break
                    end
                end
                if same then
                    const out: { number } = {}
                    for j = 1, d do
                        const v = t[start + n + j - 1]
                        if v == nil or start + n + j - 1 > pos then
                            break
                        end
                        table.insert(out, v)
                    end
                    if #out > 0 then
                        return out
                    end
                end
            end
        end
    end
    return {}
end

const e = engine.new({ host = lute.host(), source = lute.fileSource(path), context = 1024 })
const runs: { { name: string, ids: { number }, promptLen: number } } = {}
for _, p in PROMPTS do
    const chat = e:chat()
    for _ in chat:send(p.text, { maxTokens = tokens }) do
    end
    const stats = chat:stats()
    const ids = chat:tokens()
    table.insert(runs, { name = p.name, ids = ids, promptLen = stats.promptTokens })
    print(string.format("GEN %s prompt=%d new=%d", p.name, stats.promptTokens, stats.newTokens))
end

const VARIANTS = { { 3, 1 }, { 3, 2 }, { 2, 1 }, { 4, 2 } }
const DRAFTS = { 2, 3, 4, 6, 8 }
for _, v in VARIANTS do
    for _, d in DRAFTS do
        for _, run in runs do
            const t = run.ids
            local pos = run.promptLen
            local passes, produced, proposed, accepted = 0, 0, 0, 0
            local ms = 0
            while pos < #t do
                const draft = propose(t, pos, v[1], v[2], d)
                local a = 0
                while a < #draft and pos + 1 + a <= #t and draft[a + 1] == t[pos + 1 + a] do
                    a += 1
                end
                -- The pass verifies the draft and yields one more token, the first one that differs or the next one.
                const gained = math.min(a + 1, #t - pos)
                passes += 1
                produced += gained
                proposed += #draft
                accepted += a
                ms += cost(1 + #draft)
                pos += gained
            end
            const plain = cost(1) * produced
            print(
                string.format(
                    "SPEC %s nmax=%d nmin=%d draft=%d tokens=%d passes=%d tokens_per_pass=%.2f accepted=%d/%d speedup=%.2f",
                    run.name,
                    v[1],
                    v[2],
                    d,
                    produced,
                    passes,
                    produced / passes,
                    accepted,
                    proposed,
                    plain / ms
                )
            )
        end
    end
end
