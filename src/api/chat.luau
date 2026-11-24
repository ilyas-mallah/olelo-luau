--!native
-- A conversation: it keeps its KV cache between turns, so a new turn only prefills the new tokens. A turn is a resumable
-- job of steps, one prefill chunk or a few tokens each, which `send` runs to the end and a scheduler interleaves.

const prefix = require("./prefix")
const sampler = require("@src/sampling/sampler")
const templates = require("@src/tokenizer/templates")
const tokenizer = require("@src/tokenizer")
const tokenizerTypes = require("@src/tokenizer/types")
const types = require("./types")

type Chat = types.Chat
type ChatOptions = types.ChatOptions
type SendOptions = types.SendOptions
type State = types.State
type Stats = types.Stats
type StepResult = types.StepResult
type Turn = types.Turn

const DEFAULT_MAX_TOKENS = 256
--- A system prefix shorter than this is not worth a saved cache.
const MIN_SHARED = 8

--- The length of the longest suffix of `text` that begins one of `stops`.
const function holdBack(text: string, stops: { string }): number
    local hold = 0
    for _, stop in stops do
        for length = math.min(#stop - 1, #text), hold + 1, -1 do
            if string.sub(text, #text - length + 1) == string.sub(stop, 1, length) then
                hold = length
                break
            end
        end
    end
    return hold
end

--- A turn with nothing left to do.
const function finishedTurn(): Turn
    return {
        step = function(_self: Turn, _budget: number?): StepResult
            return { pieces = {}, done = true, kind = "prefill", tokens = 0 }
        end,
        cancel = function(_self: Turn) end,
    }
end

--- A new conversation on the engine's model.
const function new(c: types.Core, options: ChatOptions?): Chat
    const runner = c.runner
    const sess = runner:open()
    const seed = if options ~= nil then options.seed else nil
    const smp = sampler.new(if seed ~= nil then seed else c.seed)
    const spec = c.tokenizer.chat
    const system = if options then options.system else nil
    const history: { number } = {}
    local pending: number? = nil
    --- The last turn's closing text is already in the cache, as after `load`.
    local closed = false
    local cancelled = false
    local last: Stats = {
        promptTokens = 0,
        newTokens = 0,
        timeToFirstToken = 0,
        prefillRate = 0,
        cachedTokens = 0,
        decodeRate = 0,
        finish = "stop",
    }

    --- The tokens of the system message, which every conversation with it starts with, nil when the template has no
    --- separate system turn.
    const function systemTokens(): { number }?
        const text = templates.systemPrefix(spec, system)
        if text == nil then
            return nil
        end
        return tokenizer.encode(c.tokenizer, text)
    end

    --- Fills the empty cache from the longest saved prefix of `ids` of at most `limit` tokens. Returns its length, 0
    --- when none fits.
    const function useShared(ids: { number }, limit: number): number
        const store = c.prefixes
        if store == nil then
            return 0
        end
        const entry = prefix.find(store, ids, limit)
        if entry == nil then
            return 0
        end
        runner:restore(sess, entry.key)
        return #entry.tokens
    end

    const function begin(text: string, opts: SendOptions): Turn
        const clock = c.host.clock
        const start = clock()
        local prompt: string
        if sess.length == 0 and pending == nil then
            prompt = templates.opening(spec, system, text)
        elseif closed then
            prompt = templates.message(spec, "user", text)
        else
            prompt = templates.turnEnd(spec) .. templates.message(spec, "user", text)
        end
        prompt ..= templates.generationPrompt(spec)
        const ids: { number } = {}
        -- A token sampled at the end of the last reply is in the history but not in the cache yet.
        local inHistory = 0
        const carried = pending
        if carried ~= nil then
            ids[1] = carried
            inHistory = 1
        end
        for _, id in tokenizer.encode(c.tokenizer, prompt) do
            table.insert(ids, id)
        end
        if sess.length + #ids >= sess.context then
            error(`the prompt of {#ids} tokens does not fit the {sess.context - sess.length} positions left`, 0)
        end
        -- A refused prompt leaves the conversation as it was.
        cancelled = false
        closed = false
        pending = nil

        local logits: buffer? = nil
        local at = 1
        local cached = 0
        -- A new conversation starts from a shared system prefix when one is saved, or saves it after its own prefill.
        local toSave: { number }? = nil
        if sess.length == 0 and inHistory == 0 and c.prefixes ~= nil then
            cached = useShared(ids, #ids - 1)
            at = cached + 1
            if cached == 0 then
                const sys = systemTokens()
                if sys ~= nil and #sys >= MIN_SHARED and #sys < #ids and prefix.starts(ids, sys) then
                    toSave = sys
                end
            end
        end
        local prefilling = true
        local prefillDone = 0
        local finished = false
        const maxTokens = opts.maxTokens or DEFAULT_MAX_TOKENS
        const stops = opts.stop or {}
        const stream = tokenizer.stream(c.tokenizer)
        local held = ""
        local generated = 0
        local firstAt = 0
        local finish = "length"
        local pieces: { string } = {}

        const function emit(piece: string)
            table.insert(pieces, piece)
        end

        --- Shows text, holding back a tail that could still grow into a stop string. Returns false once a stop
        --- string was found.
        const function show(piece: string): boolean
            held ..= piece
            -- The stop string that starts first in the text wins, whatever its place in the list.
            local first: number? = nil
            for _, stop in stops do
                const found = string.find(held, stop, 1, true)
                if found ~= nil and (first == nil or found < first) then
                    first = found
                end
            end
            if first ~= nil then
                if first > 1 then
                    emit(string.sub(held, 1, first - 1))
                end
                held = ""
                return false
            end
            const hold = holdBack(held, stops)
            if #held > hold then
                emit(string.sub(held, 1, #held - hold))
                held = string.sub(held, #held - hold + 1)
            end
            return true
        end

        --- Samples the next token and runs it through the model. Returns true when the reply ended.
        const function decodeOne(): boolean
            if cancelled then
                finish = "cancelled"
                return true
            end
            if generated >= maxTokens then
                return true
            end
            const last = logits
            assert(last ~= nil, "a reply decodes after its prompt has produced logits")
            const id = sampler.sample(smp, last, c.config.vocab, history, opts)
            if c.tokenizer.stopIds[id] and not opts.ignoreStop then
                finish = "stop"
                return true
            end
            generated += 1
            table.insert(history, id)
            if generated == 1 then
                firstAt = clock()
            end
            if not show(tokenizer.push(stream, id)) then
                pending = id
                finish = "text"
                return true
            end
            if generated >= maxTokens then
                pending = id
                return true
            end
            if sess.length >= sess.context then
                pending = id
                finish = "context"
                return true
            end
            logits = runner:forward(sess, { id }, 1, 1, true, nil)
            return false
        end

        --- Releases the held text and records the numbers of the reply.
        const function conclude()
            if finish ~= "text" then
                show(tokenizer.flush(stream))
                if held ~= "" then
                    emit(held)
                    held = ""
                end
            end
            const done = clock()
            last = {
                promptTokens = #ids,
                newTokens = generated,
                timeToFirstToken = if generated > 0 then firstAt - start else done - start,
                prefillRate = (#ids - cached) / math.max(prefillDone - start, 1e-9),
                cachedTokens = cached,
                decodeRate = if generated > 1 then (generated - 1) / math.max(done - firstAt, 1e-9) else 0,
                finish = finish,
            }
            finished = true
        end

        const turn: Turn = {
            step = function(_self: Turn, budget: number?): StepResult
                pieces = {}
                if finished then
                    return { pieces = pieces, done = true, kind = "decode", tokens = 0 }
                end
                if prefilling then
                    const count = math.min(c.chunk, #ids - at + 1)
                    logits = runner:forward(sess, ids, at, count, at + count > #ids, nil)
                    at += count
                    if at > #ids then
                        for k = inHistory + 1, #ids do
                            table.insert(history, ids[k])
                        end
                        prefillDone = clock()
                        prefilling = false
                        const saving, store = toSave, c.prefixes
                        if saving ~= nil and store ~= nil then
                            prefix.add(store, sess, saving)
                        end
                    end
                    return { pieces = pieces, done = false, kind = "prefill", tokens = count }
                end
                const before = generated
                local ended = false
                for _ = 1, budget or 1 do
                    ended = decodeOne()
                    if ended then
                        break
                    end
                end
                if ended then
                    conclude()
                end
                return { pieces = pieces, done = ended, kind = "decode", tokens = generated - before }
            end,
            cancel = function(_self: Turn)
                cancelled = true
            end,
        }
        return turn
    end

    const chat: Chat = {
        begin = function(_self: Chat, text: string, sendOptions: SendOptions?): Turn
            return begin(text, sendOptions or {})
        end,
        send = function(_self: Chat, text: string, sendOptions: SendOptions?): () -> string?
            const opts: SendOptions = sendOptions or {}
            local turn: Turn? = nil
            local queue: { string } = {}
            local at = 1
            local done = false
            const function pull(): string?
                const current = if turn ~= nil then turn else begin(text, opts)
                turn = current
                while at > #queue and not done do
                    const result = current:step(1)
                    queue = result.pieces
                    at = 1
                    done = result.done
                end
                if at > #queue then
                    return nil
                end
                at += 1
                return queue[at - 1]
            end
            const onText = opts.onText
            if onText ~= nil then
                while true do
                    const piece = pull()
                    if piece == nil then
                        break
                    end
                    onText(piece)
                end
                return function(): string?
                    return nil
                end
            end
            return pull
        end,
        cancel = function(_self: Chat)
            cancelled = true
        end,
        stats = function(_self: Chat): Stats
            return last
        end,
        used = function(_self: Chat): number
            return sess.length
        end,
        tokens = function(_self: Chat): { number }
            return table.clone(history)
        end,
        restore = function(_self: Chat, messages: { tokenizerTypes.Message }): (Turn, number)
            if #messages == 0 then
                runner:rewind(sess, 0)
                table.clear(history)
                pending = nil
                closed = false
                return finishedTurn(), 0
            end
            if messages[#messages].role ~= "assistant" then
                error("the messages to load must end with a reply", 0)
            end
            const full: { tokenizerTypes.Message } = table.clone(messages)
            if system ~= nil and messages[1].role ~= "system" then
                table.insert(full, 1, { role = "system", content = system })
            end
            const ids = tokenizer.encode(c.tokenizer, templates.render(spec, full, false))
            if #ids >= sess.context then
                error(`the conversation of {#ids} tokens does not fit the context of {sess.context} positions`, 0)
            end
            -- The cache holds the first `sess.length` tokens of the history, a token sampled last is not in it yet.
            local keep = 0
            const limit = math.min(sess.length, #ids)
            while keep < limit and history[keep + 1] == ids[keep + 1] do
                keep += 1
            end
            runner:rewind(sess, keep)
            if keep == 0 and sess.length == 0 and #ids > 1 then
                keep = useShared(ids, #ids - 1)
            end
            table.move(ids, 1, #ids, 1, history)
            for _ = #ids + 1, #history do
                table.remove(history)
            end
            pending = nil
            closed = true
            local at = keep + 1
            local stopped = false
            const turn: Turn = {
                step = function(_self: Turn, _budget: number?): StepResult
                    if stopped or at > #ids then
                        return { pieces = {}, done = true, kind = "prefill", tokens = 0 }
                    end
                    const count = math.min(c.chunk, #ids - at + 1)
                    runner:forward(sess, ids, at, count, false, nil)
                    at += count
                    return { pieces = {}, done = at > #ids, kind = "prefill", tokens = count }
                end,
                cancel = function(_self: Turn)
                    stopped = true
                end,
            }
            return turn, keep
        end,
        snapshot = function(_self: Chat): State
            return { tokens = table.clone(history), pending = pending ~= nil }
        end,
        resume = function(_self: Chat, state: State): Turn
            const ids = state.tokens
            const inCache = #ids - (if state.pending then 1 else 0)
            if #ids == 0 then
                runner:rewind(sess, 0)
                table.clear(history)
                pending = nil
                closed = false
                return finishedTurn()
            end
            if inCache >= sess.context then
                error(`the conversation of {#ids} tokens does not fit the context of {sess.context} positions`, 0)
            end
            local keep = 0
            const limit = math.min(sess.length, inCache)
            while keep < limit and history[keep + 1] == ids[keep + 1] do
                keep += 1
            end
            runner:rewind(sess, keep)
            if keep == 0 and sess.length == 0 and inCache > 1 then
                keep = useShared(ids, inCache - 1)
            end
            table.clear(history)
            table.move(ids, 1, #ids, 1, history)
            pending = if state.pending then ids[#ids] else nil
            closed = false
            local at = keep + 1
            local stopped = false
            const turn: Turn = {
                step = function(_self: Turn, _budget: number?): StepResult
                    if stopped or at > inCache then
                        return { pieces = {}, done = true, kind = "prefill", tokens = 0 }
                    end
                    const count = math.min(c.chunk, inCache - at + 1)
                    runner:forward(sess, ids, at, count, false, nil)
                    at += count
                    return { pieces = {}, done = at > inCache, kind = "prefill", tokens = count }
                end,
                cancel = function(_self: Turn)
                    stopped = true
                end,
            }
            return turn
        end,
        load = function(self: Chat, messages: { tokenizerTypes.Message }): number
            const turn, keep = self:restore(messages)
            while not turn:step().done do
            end
            return keep
        end,
        reset = function(_self: Chat)
            runner:rewind(sess, 0)
            table.clear(history)
            pending = nil
            closed = false
        end,
        close = function(_self: Chat)
            runner:close(sess)
        end,
    }
    return chat
end

return {
    new = new,
}
