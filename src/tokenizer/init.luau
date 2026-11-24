--!native
-- Text to token ids and back, with special tokens and a decoder that never splits a character across two pieces.

const bpe = require("@self/bpe")
const container = require("@src/container")
const pretok = require("@self/pretok")
const unicode = require("@self/unicode")
const containerTypes = require("@src/container/types")
const types = require("@self/types")

export type Bpe = types.Bpe
export type Message = types.Message
export type Ranges = types.Ranges
export type Spec = types.Spec
export type Special = types.Special
export type Tables = types.Tables
export type Tokenizer = types.Tokenizer

--- Streaming decoder state: bytes of an unfinished character wait for the rest.
export type Stream = {
    tokenizer: Tokenizer,
    pending: string,
}

--- Builds the tokenizer from an open container.
const function new(c: containerTypes.Container): Tokenizer
    const header = c.header
    const info = header.tokenizer
    const spm = info.kind == "spm"
    if
        not spm
        and (
            info.kind ~= "bytebpe"
            or (info.pretokenizer ~= "gpt2" and info.pretokenizer ~= "qwen2" and info.pretokenizer ~= "llama3")
        )
    then
        error(`tokenizer {info.kind}/{info.pretokenizer} is not supported`, 0)
    end
    const stopIds: { [number]: boolean } = {}
    for _, id in info.eos do
        stopIds[id] = true
    end
    const specials: { Special } = info.specials
    -- Longest first, so a special token that starts with another one wins.
    table.sort(specials, function(a: Special, b: Special): boolean
        return #a.content > #b.content
    end)
    const byByte: { [number]: { Special } } = {}
    local startSet = ""
    for _, s in specials do
        const first = string.byte(s.content, 1)
        if first == nil then
            continue
        end
        if byByte[first] == nil then
            byByte[first] = {}
            const ch = string.char(first)
            startSet ..= if string.find(ch, "^%w$") ~= nil then ch else "%" .. ch
        end
        table.insert(byByte[first], s)
    end
    return {
        bpe = bpe.load(
            container.blob(c, "tokenizer.vocab"),
            container.blob(c, "tokenizer.merges"),
            specials,
            spm,
            info.ignoreMerges == true
        ),
        unicode = unicode.load(container.blob(c, "unicode")),
        pretokenizer = if spm then "none" else info.pretokenizer,
        splitDigits = if spm then false else info.splitDigits,
        specials = specials,
        specialsByByte = byByte,
        specialStart = "[" .. startSet .. "]",
        bos = if info.addBos then info.bos else nil,
        stopIds = stopIds,
        chat = header.chat,
    }
end

--- Replaces bytes that are not valid UTF-8 with U+FFFD.
const function sanitize(text: string): string
    local length, bad = utf8.len(text)
    if length ~= nil then
        return text
    end
    const parts: { string } = {}
    local pos = 1
    while pos <= #text do
        length, bad = utf8.len(text, pos)
        if length ~= nil then
            table.insert(parts, string.sub(text, pos))
            break
        end
        const invalid = bad
        assert(invalid ~= nil, "utf8.len gives the position of the invalid byte")
        table.insert(parts, string.sub(text, pos, invalid - 1))
        table.insert(parts, "\u{FFFD}")
        pos = invalid + 1
    end
    return table.concat(parts)
end

const function encodeText(t: Tokenizer, text: string, out: { number })
    if text == "" then
        return
    end
    if t.bpe.spm then
        bpe.encodeSentence(t.bpe, sanitize(text), out)
        return
    end
    for _, piece in pretok.split(t.unicode, sanitize(text), t.pretokenizer, t.splitDigits) do
        bpe.encodePiece(t.bpe, piece, out)
    end
end

--- The first special token's text at or after `pos`: its start, end and id, the longest where several start at the
--- same place.
const function findSpecial(t: Tokenizer, text: string, pos: number): (number?, number, number)
    local at = pos
    while true do
        const found = string.find(text, t.specialStart, at)
        if found == nil then
            return nil, 0, 0
        end
        for _, s in t.specialsByByte[string.byte(text, found)] do
            if string.sub(text, found, found + #s.content - 1) == s.content then
                return found, found + #s.content - 1, s.id
            end
        end
        at = found + 1
    end
end

--- The ids of `text`. With `parseSpecials` a special token's text, such as `<|im_end|>`, becomes its single id. With
--- `addBos` the model's beginning of sequence token comes first, if it has one, as the Hugging Face tokenizer does.
const function encode(t: Tokenizer, text: string, parseSpecials: boolean?, addBos: boolean?): { number }
    const out: { number } = {}
    const bos = t.bos
    if addBos == true and bos ~= nil then
        table.insert(out, bos)
    end
    if parseSpecials == false then
        encodeText(t, text, out)
        return out
    end
    local pos = 1
    while true do
        const start, stop, id = findSpecial(t, text, pos)
        if start == nil then
            encodeText(t, string.sub(text, pos), out)
            return out
        end
        encodeText(t, string.sub(text, pos, start - 1), out)
        table.insert(out, id)
        pos = stop + 1
    end
end

--- The bytes of `ids` as one string.
const function decode(t: Tokenizer, ids: { number }): string
    const parts: { string } = {}
    for i, id in ids do
        parts[i] = t.bpe.idToBytes[id + 1] or ""
    end
    return table.concat(parts)
end

--- A decoder for a stream of ids.
const function stream(t: Tokenizer): Stream
    return { tokenizer = t, pending = "" }
end

--- How many bytes at the end of `text` start a character that is not complete yet.
const function incompleteTail(text: string): number
    const n = #text
    for back = 1, math.min(3, n) do
        const b = string.byte(text, n - back + 1)
        if b >= 0xC0 then
            const need = if b >= 0xF0 then 4 elseif b >= 0xE0 then 3 else 2
            return if need > back then back else 0
        elseif b < 0x80 then
            return 0
        end
    end
    return 0
end

--- Adds one id and returns the text that became complete, possibly "".
const function push(s: Stream, id: number): string
    const text = s.pending .. (s.tokenizer.bpe.idToBytes[id + 1] or "")
    const hold = incompleteTail(text)
    s.pending = if hold > 0 then string.sub(text, #text - hold + 1) else ""
    return string.sub(text, 1, #text - hold)
end

--- Returns the bytes still waiting, as they are.
const function flush(s: Stream): string
    const rest = s.pending
    s.pending = ""
    return rest
end

return {
    new = new,
    encode = encode,
    decode = decode,
    stream = stream,
    push = push,
    flush = flush,
}
