--!native
-- Prefixes that conversations share, a system prompt above all. The runner keeps the KV cache of the first positions of a
-- sequence under a key, and a new conversation whose prompt starts with the same tokens copies it instead of running them
-- through the model. The store only knows which token lists have a key. The cache lives where the model runs, in this VM
-- or in each worker, so no cache crosses a transport.

const runner = require("@src/model/runner")
const modelTypes = require("@src/model/types")
const types = require("./types")

type Entry = types.Entry
type Store = types.Store

--- A store that keeps up to `capacity` prefixes.
const function new(r: modelTypes.Runner, capacity: number): Store
    return { runner = r, entries = {}, capacity = capacity, nextKey = 1, tick = 0, hits = 0, misses = 0 }
end

--- Whether `entry` is the start of `ids`.
const function starts(ids: { number }, tokens: { number }): boolean
    for i = 1, #tokens do
        if ids[i] ~= tokens[i] then
            return false
        end
    end
    return true
end

--- The longest saved prefix of `ids` that is at most `limit` tokens long, counted as a use. Nil when there is none.
const function find(s: Store, ids: { number }, limit: number): Entry?
    local best: Entry? = nil
    for _, e in s.entries do
        if #e.tokens <= limit and #e.tokens <= #ids and (best == nil or #e.tokens > #(best :: Entry).tokens) then
            if starts(ids, e.tokens) then
                best = e
            end
        end
    end
    if best == nil then
        s.misses += 1
        return nil
    end
    const found = best :: Entry
    s.tick += 1
    s.hits += 1
    found.used = s.tick
    return found
end

--- Saves the first `#tokens` positions of `seq`, which holds at least that many and the tokens `tokens`. Drops the least
--- recently used entry when the store is full. Does nothing when `tokens` is saved already.
const function add(s: Store, seq: modelTypes.Sequence, tokens: { number })
    if s.capacity <= 0 then
        return
    end
    for _, e in s.entries do
        if #e.tokens == #tokens and starts(e.tokens, tokens) then
            return
        end
    end
    if #s.entries >= s.capacity then
        local oldest = 1
        for i, e in s.entries do
            if e.used < s.entries[oldest].used then
                oldest = i
            end
        end
        s.runner:drop(s.entries[oldest].key)
        table.remove(s.entries, oldest)
    end
    const key = s.nextKey
    s.nextKey += 1
    s.runner:save(seq, #tokens, key)
    s.tick += 1
    table.insert(s.entries, { tokens = table.clone(tokens), key = key, used = s.tick })
end

return {
    new = new,
    find = find,
    add = add,
    starts = starts,
}
