--!native
-- Picks the next token from a row of logits: penalties, temperature, top k, min p and top p, in that order. Temperature 0
-- is greedy. Candidates more than 20 temperatures below the best logit are dropped, they hold under 2e-9 of the best
-- one's probability each. The best tokens are selected with a bounded insertion, never by sorting the whole vocabulary,
-- so a step costs a few passes over the logits whatever the settings.

const rng = require("./rng")
const types = require("./types")

type Options = types.Options

export type Sampler = {
    rng: types.Rng,
    vals: { number },
    --- Scratch: occurrences of recent tokens, and the selected best tokens with their logits.
    counts: { [number]: number },
    ids: { number },
    tops: { number },
    --- Scratch: the mass of each bucket of `drawHistogram`.
    mass: { number },
}

const CUTOFF = 20
--- Best tokens selected at first when top p cuts the list. A cut that needs more takes `drawHistogram`.
const FIRST_HEAD = 256
--- Parts of the logit range `drawHistogram` sums the mass in.
const BUCKETS = 1024

--- A sampler seeded with `seed`.
const function new(seed: number): Sampler
    return {
        rng = rng.new(seed),
        vals = {},
        counts = {},
        ids = {},
        tops = {},
        mass = table.create<<number>>(BUCKETS, 0),
    }
end

--- Puts the `m` largest of the first `n` values that are at least `floor` into `ids` and `tops`, largest first, and
--- returns how many there are.
const function selectTop(s: Sampler, n: number, m: number, floor: number): number
    const vals, ids, tops = s.vals, s.ids, s.tops
    table.clear(ids)
    table.clear(tops)
    local count = 0
    local limit = floor
    for i = 1, n do
        const v = vals[i]
        if v >= limit and (count < m or v > tops[count]) then
            -- Binary search for the first position whose value is below v.
            local lo, hi = 1, count + 1
            while lo < hi do
                const mid = (lo + hi) // 2
                if tops[mid] >= v then
                    lo = mid + 1
                else
                    hi = mid
                end
            end
            table.insert(tops, lo, v)
            table.insert(ids, lo, i)
            if count < m then
                count += 1
            else
                table.remove(tops)
                table.remove(ids)
            end
            if count == m then
                limit = math.max(floor, tops[count])
            end
        end
    end
    return count
end

--- The index of the bucket of `v` among `BUCKETS` equal parts of the logit range from `best` down, 1 for the best.
const function bucketOf(v: number, best: number, scale: number): number
    return math.min(BUCKETS, math.floor((best - v) * scale) + 1)
end

--- Top p over the tokens at or above `cut`, for when the best `FIRST_HEAD` tokens do not hold the mass, as with a flat
--- distribution over a large vocabulary. One pass sums the mass of every token into the bucket of its logit. The bucket
--- where the cumulative mass reaches `topP` is sorted, which holds a few tokens, and the draw is over the buckets above
--- it and the sorted tokens up to the one that reaches `topP`. Costs three passes whatever the shape of the distribution.
const function drawHistogram(
    s: Sampler,
    n: number,
    cut: number,
    best: number,
    inv: number,
    topP: number,
    total: number
): number
    const vals, mass, ids, tops = s.vals, s.mass, s.ids, s.tops
    const span = best - cut
    if span <= 0 then
        -- Every token above the cut has the same logit, so any of them is as good as the next.
        local first = 0
        for i = 1, n do
            if vals[i] >= cut then
                first += 1
            end
        end
        local pick = math.floor(rng.nextFloat(s.rng) * first)
        for i = 1, n do
            if vals[i] >= cut then
                if pick == 0 then
                    return i - 1
                end
                pick -= 1
            end
        end
        return 0
    end
    const scale = BUCKETS / span
    for b = 1, BUCKETS do
        mass[b] = 0
    end
    for i = 1, n do
        const v = vals[i]
        if v >= cut then
            const b = bucketOf(v, best, scale)
            mass[b] += math.exp((v - best) * inv)
        end
    end
    const target = topP * total
    local before, crossing = 0, BUCKETS
    for b = 1, BUCKETS do
        if before + mass[b] >= target then
            crossing = b
            break
        end
        before += mass[b]
    end
    table.clear(ids)
    table.clear(tops)
    for i = 1, n do
        const v = vals[i]
        if v >= cut and bucketOf(v, best, scale) == crossing then
            table.insert(ids, i)
            table.insert(tops, v)
        end
    end
    const order = table.create<<number>>(#ids)
    for k = 1, #ids do
        order[k] = k
    end
    table.sort(order, function(a: number, b: number): boolean
        return tops[a] > tops[b]
    end)
    local added, kept = 0, 0
    for r = 1, #order do
        added += math.exp((tops[order[r]] - best) * inv)
        kept = r
        if before + added >= target then
            break
        end
    end
    local draw = rng.nextFloat(s.rng) * (before + added)
    for i = 1, n do
        const v = vals[i]
        if v >= cut and bucketOf(v, best, scale) < crossing then
            draw -= math.exp((v - best) * inv)
            if draw < 0 then
                return i - 1
            end
        end
    end
    for r = 1, kept do
        draw -= math.exp((tops[order[r]] - best) * inv)
        if draw < 0 then
            return ids[order[r]] - 1
        end
    end
    return ids[order[math.max(kept, 1)]] - 1
end

--- The token id picked from the first `n` values of the f32 `logits`. `history` holds the ids so far.
const function sample(s: Sampler, logits: buffer, n: number, history: { number }, options: Options): number
    const vals = s.vals
    local best, bestId = -math.huge, 0
    for i = 1, n do
        const v = buffer.readf32(logits, (i - 1) * 4)
        vals[i] = v
        if v > best then
            best, bestId = v, i
        end
    end
    const repetition = options.repetitionPenalty or 1
    const frequency = options.frequencyPenalty or 0
    const presence = options.presencePenalty or 0
    if repetition ~= 1 or frequency ~= 0 or presence ~= 0 then
        const window = options.repetitionWindow or 64
        const counts = s.counts
        table.clear(counts)
        for h = math.max(1, #history - window + 1), #history do
            const id = history[h] + 1
            if id <= n then
                counts[id] = (counts[id] or 0) + 1
            end
        end
        local bestHit = false
        for id, count in counts do
            local v = vals[id]
            if repetition ~= 1 then
                v = if v > 0 then v / repetition else v * repetition
            end
            vals[id] = v - frequency * count - presence
            if id == bestId then
                bestHit = true
            end
        end
        -- Penalties lower logits only, so the best token changes only when the best one was penalized.
        if bestHit then
            best, bestId = -math.huge, 0
            for i = 1, n do
                if vals[i] > best then
                    best, bestId = vals[i], i
                end
            end
        end
    end
    const temperature = options.temperature or 0
    if temperature <= 0 then
        return bestId - 1
    end

    const topK = options.topK or 0
    const topP = options.topP or 1
    const minP = options.minP or 0
    const inv = 1 / temperature
    const floor = best - CUTOFF * temperature
    -- Tokens below this logit are cut by min p whatever else is set.
    const cut = if minP > 0 then math.max(floor, best + temperature * math.log(minP)) else floor
    if topK <= 0 and topP >= 1 then
        -- No list: draw over every token above the cut, in two passes.
        local total = 0
        for i = 1, n do
            const v = vals[i]
            if v >= cut then
                total += math.exp((v - best) * inv)
            end
        end
        local draw = rng.nextFloat(s.rng) * total
        for i = 1, n do
            const v = vals[i]
            if v >= cut then
                draw -= math.exp((v - best) * inv)
                if draw < 0 then
                    return i - 1
                end
            end
        end
        return bestId - 1
    end

    -- The mass of the whole set top p measures against: every token above the cut, or the top k of them.
    local fullTotal = 0
    if topK <= 0 then
        for i = 1, n do
            const v = vals[i]
            if v >= cut then
                fullTotal += math.exp((v - best) * inv)
            end
        end
    end
    const ids, tops = s.ids, s.tops
    const count = selectTop(s, n, if topK > 0 then topK else FIRST_HEAD, cut)
    -- Kept tokens, from the best down: all of the head, or until the cumulative mass reaches top p.
    local keep = count
    local total = 0
    for k = 1, count do
        total += math.exp((tops[k] - best) * inv)
    end
    if topP < 1 then
        const mass = if topK > 0 then total else fullTotal
        local cumulative = 0
        local reached = false
        for k = 1, count do
            cumulative += math.exp((tops[k] - best) * inv) / mass
            if cumulative >= topP then
                keep = k
                reached = true
                break
            end
        end
        if not reached and topK <= 0 and count == FIRST_HEAD and count < n then
            -- The head ended before the mass was reached.
            return drawHistogram(s, n, cut, best, inv, topP, fullTotal)
        end
        if reached then
            total = 0
            for k = 1, keep do
                total += math.exp((tops[k] - best) * inv)
            end
        end
    end
    local draw = rng.nextFloat(s.rng) * total
    for k = 1, keep do
        draw -= math.exp((tops[k] - best) * inv)
        if draw < 0 then
            return ids[k] - 1
        end
    end
    return ids[keep] - 1
end

return {
    new = new,
    sample = sample,
}
