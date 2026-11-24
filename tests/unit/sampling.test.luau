const harness = require("@tests/harness")
const rng = require("@src/sampling/rng")
const sampler = require("@src/sampling/sampler")
const samplingTypes = require("@src/sampling/types")

--- A logits buffer from a list of values.
const function logits(values: { number }): buffer
    const b = buffer.create(#values * 4)
    for i, v in values do
        buffer.writef32(b, (i - 1) * 4, v)
    end
    return b
end

return function(h: harness.Harness)
    h.case("the generator matches the reference stream", function()
        -- Values from the same PCG algorithm run on 64 bit integers in Python.
        const a = rng.new(1)
        for _, want in { 2831084092, 645514520, 2544563910, 710022002, 3500610781 } do
            h.equal(rng.nextU32(a), want)
        end
        const b = rng.new(123456789)
        for _, want in { 4272394698, 3879534080, 1293520846 } do
            h.equal(rng.nextU32(b), want)
        end
    end)

    h.case("floats are in range and a seed repeats", function()
        const a, b = rng.new(5), rng.new(5)
        for _ = 1, 1000 do
            const x = rng.nextFloat(a)
            h.truthy(x >= 0 and x < 1)
            h.equal(x, rng.nextFloat(b))
        end
    end)

    h.case("temperature 0 is greedy", function()
        const s = sampler.new(1)
        h.equal(sampler.sample(s, logits({ 1, 5, 3, 4 }), 4, {}, {}), 1)
        h.equal(sampler.sample(s, logits({ -9, -5, -3, -4 }), 4, {}, { temperature = 0 }), 2)
    end)

    h.case("repetition penalty moves the best token", function()
        const s = sampler.new(1)
        const l = logits({ 2, 5, 4.5 })
        h.equal(sampler.sample(s, l, 3, {}, {}), 1)
        h.equal(sampler.sample(s, l, 3, { 1 }, { repetitionPenalty = 2 }), 2)
        h.equal(sampler.sample(s, l, 3, { 1 }, { repetitionPenalty = 1.0 }), 1)
    end)

    h.case("top k 1 and a tiny top p are greedy", function()
        const s = sampler.new(3)
        const l = logits({ 1, 2, 3, 2.5, 0 })
        for _ = 1, 50 do
            h.equal(sampler.sample(s, l, 5, {}, { temperature = 1, topK = 1 }), 2)
            h.equal(sampler.sample(s, l, 5, {}, { temperature = 1, topP = 0.01 }), 2)
        end
    end)

    h.case("min p drops unlikely tokens", function()
        const s = sampler.new(4)
        const l = logits({ 10, 9.9, 0, 0, 0 })
        for _ = 1, 200 do
            const id = sampler.sample(s, l, 5, {}, { temperature = 1, minP = 0.1 })
            h.truthy(id == 0 or id == 1)
        end
    end)

    h.case("sampling follows the probabilities", function()
        const s = sampler.new(9)
        const l = logits({ 0, math.log(3) })
        local ones = 0
        for _ = 1, 4000 do
            ones += sampler.sample(s, l, 2, {}, { temperature = 1 })
        end
        -- Token 1 has probability 3 / 4.
        h.near(ones / 4000, 0.75, 0.03)
    end)

    h.case("the same seed gives the same tokens", function()
        const l = logits({ 1, 1.1, 0.9, 1.05, 0.95 })
        const a, b = sampler.new(11), sampler.new(11)
        for _ = 1, 100 do
            h.equal(
                sampler.sample(a, l, 5, {}, { temperature = 0.8, topK = 3 }),
                sampler.sample(b, l, 5, {}, { temperature = 0.8, topK = 3 })
            )
        end
    end)
    --- The probability of every token under the settings, from a full sort in doubles: top k, then min p, then top p
    --- over what is left, as the sampler orders them.
    const function reference(values: { number }, o: samplingTypes.Options): { [number]: number }
        const temperature = o.temperature or 1
        local best = -math.huge
        for _, v in values do
            best = math.max(best, v)
        end
        const order: { number } = {}
        for i = 1, #values do
            order[i] = i
        end
        table.sort(order, function(a: number, b: number): boolean
            return values[a] > values[b]
        end)
        local keep = #order
        if (o.topK or 0) > 0 then
            keep = math.min(keep, o.topK :: number)
        end
        const p: { number } = {}
        for k = 1, keep do
            p[k] = math.exp((values[order[k]] - best) / temperature)
        end
        if (o.minP or 0) > 0 then
            for k = 1, keep do
                if p[k] < (o.minP :: number) * p[1] then
                    keep = k - 1
                    break
                end
            end
        end
        local total = 0
        for k = 1, keep do
            total += p[k]
        end
        if (o.topP or 1) < 1 then
            local cumulative = 0
            for k = 1, keep do
                cumulative += p[k] / total
                if cumulative >= (o.topP :: number) then
                    keep = k
                    break
                end
            end
            total = 0
            for k = 1, keep do
                total += p[k]
            end
        end
        const out: { [number]: number } = {}
        for k = 1, keep do
            out[order[k] - 1] = p[k] / total
        end
        return out
    end

    h.case("every cut keeps the tokens and the probabilities a full sort would", function()
        const rngs = rng.new(21)
        const values: { number } = {}
        for i = 1, 40 do
            values[i] = (rng.nextFloat(rngs) - 0.5) * 6
        end
        const l = logits(values)
        for _, o in
            {
                { temperature = 1, topK = 5 },
                { temperature = 0.7, topK = 12, topP = 0.8 },
                { temperature = 1, topP = 0.7 },
                { temperature = 1.3, minP = 0.05 },
                { temperature = 1, minP = 0.1, topP = 0.9 },
                { temperature = 0.9, topK = 8, minP = 0.2 },
            } :: { samplingTypes.Options }
        do
            const want = reference(values, o)
            const s = sampler.new(5)
            const seen: { [number]: number } = {}
            const draws = 6000
            for _ = 1, draws do
                const id = sampler.sample(s, l, 40, {}, o)
                h.truthy(want[id] ~= nil, `token {id} is outside the kept set`)
                seen[id] = (seen[id] or 0) + 1
            end
            for id, p in want do
                const got = (seen[id] or 0) / draws
                h.truthy(math.abs(got - p) < 4 * math.sqrt(p * (1 - p) / draws) + 1e-3, `token {id}: {got} against {p}`)
            end
        end
    end)

    h.case("a cut over a large flat vocabulary selects more than the first head", function()
        const values: { number } = {}
        for i = 1, 2000 do
            values[i] = (i % 7) * 0.001
        end
        const s = sampler.new(2)
        const l = logits(values)
        const seen: { [number]: boolean } = {}
        local distinct = 0
        for _ = 1, 6000 do
            const id = sampler.sample(s, l, 2000, {}, { temperature = 1, topP = 0.9 })
            if not seen[id] then
                seen[id] = true
                distinct += 1
            end
        end
        h.truthy(distinct > 1000, `{distinct} tokens were drawn, about 1800 are kept`)
        local low = 0
        for _ = 1, 400 do
            const id = sampler.sample(s, l, 2000, {}, { temperature = 1, minP = 0.5 })
            low += if id >= 0 then 1 else 0
        end
        h.equal(low, 400)
    end)

    h.case("top p over a large flat vocabulary draws the distribution of a full sort", function()
        const rngs = rng.new(44)
        const vocab = 800
        const values: { number } = {}
        for i = 1, vocab do
            values[i] = (rng.nextFloat(rngs) - 0.5) * 6
        end
        const l = logits(values)
        for _, o in
            {
                { temperature = 3, topP = 0.8 },
                { temperature = 3, topP = 0.95, minP = 0.1 },
            } :: { samplingTypes.Options }
        do
            const want = reference(values, o)
            const s = sampler.new(9)
            const seen: { [number]: number } = {}
            const draws = 20000
            for _ = 1, draws do
                const id = sampler.sample(s, l, vocab, {}, o)
                seen[id] = (seen[id] or 0) + 1
            end
            local outside = 0
            for id, count in seen do
                if want[id] == nil then
                    outside += count
                end
            end
            h.equal(outside, 0, "every draw is inside the kept set")
            for id, p in want do
                const got = (seen[id] or 0) / draws
                h.truthy(math.abs(got - p) < 5 * math.sqrt(p * (1 - p) / draws) + 2e-3, `token {id}: {got} against {p}`)
            end
        end
    end)

    h.case("the repetition penalty counts a repeated token once", function()
        const s = sampler.new(1)
        const l = logits({ 5, 4.2, 3.9 })
        h.equal(sampler.sample(s, l, 3, { 0, 0, 0 }, { repetitionPenalty = 1.1 }), 0)
        h.equal(sampler.sample(s, l, 3, { 0 }, { repetitionPenalty = 1.1 }), 0)
        h.equal(sampler.sample(s, l, 3, { 0 }, { repetitionPenalty = 1.3 }), 1)
        -- Negative logits are multiplied, so they fall too.
        h.equal(sampler.sample(s, logits({ -1, -1.2 }), 2, { 0 }, { repetitionPenalty = 1.5 }), 1)
    end)

    h.case("frequency and presence penalties lower a token by its count and by its presence", function()
        const s = sampler.new(1)
        const l = logits({ 5, 4.5 })
        h.equal(sampler.sample(s, l, 2, { 0 }, { frequencyPenalty = 0.3 }), 0)
        h.equal(sampler.sample(s, l, 2, { 0, 0, 0 }, { frequencyPenalty = 0.3 }), 1)
        h.equal(sampler.sample(s, logits({ 5, 4.8 }), 2, { 0, 0, 0 }, { presencePenalty = 0.1 }), 0)
        h.equal(sampler.sample(s, logits({ 5, 4.8 }), 2, { 0, 0, 0 }, { presencePenalty = 0.5 }), 1)
        h.equal(sampler.sample(s, logits({ 5, 4.8 }), 2, { 1 }, { presencePenalty = 0.5 }), 0)
    end)

    h.case("only the recent window counts for the penalties", function()
        const s = sampler.new(1)
        const l = logits({ 5, 5.5 })
        h.equal(sampler.sample(s, l, 2, { 0, 1, 1, 1 }, { presencePenalty = 1, repetitionWindow = 3 }), 0)
        h.equal(sampler.sample(s, l, 2, { 0, 1, 1, 1 }, { presencePenalty = 1, repetitionWindow = 4 }), 1)
    end)

    h.case("a step over a large vocabulary stays cheap whatever the cut", function()
        const vocab = 60000
        const rngs = rng.new(3)
        const big = buffer.create(vocab * 4)
        for i = 0, vocab - 1 do
            buffer.writef32(big, i * 4, (rng.nextFloat(rngs) - 0.5) * 16)
        end
        const s = sampler.new(1)
        const t0 = os.clock()
        for _ = 1, 5 do
            sampler.sample(
                s,
                big,
                vocab,
                { 1, 2, 3 },
                { temperature = 1, topK = 40, topP = 0.9, repetitionPenalty = 1.1 }
            )
        end
        const seconds = (os.clock() - t0) / 5
        -- A full sort of the candidates took 0.15 s here, the selection is a few passes over the logits.
        h.truthy(seconds < 0.05, `{seconds} s per token`)
        -- A flat distribution needs most of the vocabulary for top p or min p, which sorting could not do in time.
        const t1 = os.clock()
        for _ = 1, 3 do
            sampler.sample(s, big, vocab, {}, { temperature = 1, topP = 0.9 })
            sampler.sample(s, big, vocab, {}, { temperature = 1, minP = 0.01 })
        end
        const flat = (os.clock() - t1) / 6
        h.truthy(flat < 0.1, `{flat} s per token with top p or min p over a flat vocabulary`)
    end)
end
