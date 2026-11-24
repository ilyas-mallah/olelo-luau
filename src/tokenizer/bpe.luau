--!native
-- BPE: the vocabulary, the merge ranks and the encoding of one piece of text. Two forms: byte level (GPT-2, Qwen), where
-- the bytes of a piece are mapped to printable characters, and SentencePiece style (Gemma), where a space is the
-- character U+2581, a character outside the vocabulary becomes its byte tokens and the whole text is one piece.

const types = require("./types")

type Bpe = types.Bpe

const KEY = 262144
const CACHE_LIMIT = 20000
const CACHE_PIECE_BYTES = 48
--- Rank and position share a heap key: `rank * POSITIONS + position`.
const POSITIONS = 16777216
const SPACE_MARK = "\u{2581}"

--- The characters GPT-2 maps bytes to: printable bytes stand for themselves, the rest follow 255.
const function byteChars(): ({ string }, { [number]: number })
    const chars = table.create<<string>>(256)
    const inverse: { [number]: number } = {}
    local extra = 0
    for b = 0, 255 do
        local cp = b
        if not ((b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255)) then
            cp = 256 + extra
            extra += 1
        end
        chars[b + 1] = utf8.char(cp)
        inverse[cp] = b
    end
    return chars, inverse
end

--- The bytes a SentencePiece style token stands for: a byte token `<0xXX>` is that byte, the space mark is a space.
const function spmBytes(text: string): string
    const hex = string.match(text, "^<0x(%x%x)>$")
    if hex ~= nil then
        return string.char(tonumber(hex, 16) :: number)
    end
    return (string.gsub(text, SPACE_MARK, " "))
end

--- Builds the tables from the container's vocabulary and merge blobs. `specials` lists the special tokens, whose
--- bytes are their own text. With `spm` the vocabulary is SentencePiece style. With `ignoreMerges` a piece that is
--- a whole token is encoded as that token.
const function load(
    vocabBlob: buffer,
    mergesBlob: buffer,
    specials: { { id: number, content: string } },
    spm: boolean,
    ignoreMerges: boolean
): Bpe
    const chars, inverse = byteChars()
    const size = buffer.readu32(vocabBlob, 0)
    const vocab: { [string]: number } = {}
    const idToBytes = table.create<<string>>(size)
    local at = 4
    for id = 0, size - 1 do
        const length = buffer.readu16(vocabBlob, at)
        const text = buffer.readstring(vocabBlob, at + 2, length)
        at += 2 + length
        if length > 0 then
            vocab[text] = id
        end
        if spm then
            idToBytes[id + 1] = spmBytes(text)
        else
            const pieces: { string } = {}
            for _, cp in utf8.codes(text) do
                const b = inverse[cp]
                -- A character that is no byte stand in stands for itself.
                table.insert(pieces, if b == nil then utf8.char(cp) else string.char(b))
            end
            idToBytes[id + 1] = table.concat(pieces)
        end
    end
    for _, s in specials do
        idToBytes[s.id + 1] = if spm then spmBytes(s.content) else s.content
    end
    const byteId = table.create<<number>>(256)
    for b = 0, 255 do
        byteId[b + 1] = if spm then vocab[string.format("<0x%02X>", b)] else vocab[chars[b + 1]]
    end
    const rank: { [number]: number } = {}
    const merged: { [number]: number } = {}
    const count = buffer.readu32(mergesBlob, 0)
    for r = 0, count - 1 do
        const a = buffer.readu32(mergesBlob, 4 + r * 12)
        const b = buffer.readu32(mergesBlob, 8 + r * 12)
        const result = buffer.readu32(mergesBlob, 12 + r * 12)
        const key = a * KEY + b
        rank[key] = r
        merged[key] = result
    end
    return {
        vocab = vocab,
        idToBytes = idToBytes,
        byteId = byteId,
        rank = rank,
        merged = merged,
        cache = {},
        cacheSize = 0,
        size = size,
        spm = spm,
        ignoreMerges = ignoreMerges,
        byteChar = chars,
    }
end

--- Appends the ids of one pre-tokenized piece to `out`.
const function encodePiece(bpe: Bpe, piece: string, out: { number })
    const cached = bpe.cache[piece]
    if cached ~= nil then
        for _, id in cached do
            table.insert(out, id)
        end
        return
    end
    if bpe.ignoreMerges then
        const mapped = table.create<<string>>(#piece)
        for i = 1, #piece do
            mapped[i] = bpe.byteChar[string.byte(piece, i) + 1]
        end
        const whole = bpe.vocab[table.concat(mapped)]
        if whole ~= nil then
            table.insert(out, whole)
            return
        end
    end
    const ids = table.create<<number>>(#piece)
    const byteId = bpe.byteId
    for i = 1, #piece do
        ids[i] = byteId[string.byte(piece, i) + 1]
    end
    local n = #ids
    const rank, merged = bpe.rank, bpe.merged
    while n > 1 do
        local best = math.huge
        local bestKey = 0
        for i = 1, n - 1 do
            const key = ids[i] * KEY + ids[i + 1]
            const r = rank[key]
            if r ~= nil and r < best then
                best = r
                bestKey = key
            end
        end
        if best == math.huge then
            break
        end
        const a = bestKey // KEY
        const b = bestKey % KEY
        const result = merged[bestKey]
        local w = 1
        local i = 1
        while i <= n do
            if i < n and ids[i] == a and ids[i + 1] == b then
                ids[w] = result
                i += 2
            else
                ids[w] = ids[i]
                i += 1
            end
            w += 1
        end
        for k = n, w, -1 do
            table.remove(ids, k)
        end
        n = w - 1
    end
    if #piece <= CACHE_PIECE_BYTES then
        if bpe.cacheSize >= CACHE_LIMIT then
            bpe.cache = {}
            bpe.cacheSize = 0
        end
        bpe.cache[piece] = ids
        bpe.cacheSize += 1
    end
    for _, id in ids do
        table.insert(out, id)
    end
end

--- Pushes the heap entry `key` and returns the new size.
const function heapPush(heap: { number }, size: number, key: number): number
    size += 1
    local i = size
    while i > 1 do
        const parent = i // 2
        if heap[parent] <= key then
            break
        end
        heap[i] = heap[parent]
        i = parent
    end
    heap[i] = key
    return size
end

--- Removes the smallest entry and returns it with the new size. The heap holds at least one entry.
const function heapPop(heap: { number }, size: number): (number, number)
    const top = heap[1]
    const last = heap[size]
    size -= 1
    local i = 1
    while true do
        local child = i * 2
        if child > size then
            break
        end
        if child < size and heap[child + 1] < heap[child] then
            child += 1
        end
        if heap[child] >= last then
            break
        end
        heap[i] = heap[child]
        i = child
    end
    if size >= 1 then
        heap[i] = last
    end
    return top, size
end

--- Appends the ids of a SentencePiece style text to `out`: each character is a symbol, one outside the vocabulary
--- becomes its byte tokens, then the lowest ranked pair merges first and the leftmost among equals, as Hugging Face
--- does.
const function encodeSpm(bpe: Bpe, text: string, out: { number })
    const ids: { number } = {}
    local n = 0
    const vocab, byteId = bpe.vocab, bpe.byteId
    for _, cp in utf8.codes(text) do
        const ch = utf8.char(cp)
        const id = vocab[ch]
        if id ~= nil then
            n += 1
            ids[n] = id
        else
            for i = 1, #ch do
                n += 1
                ids[n] = byteId[string.byte(ch, i) + 1] or vocab["<unk>"] or 0
            end
        end
    end
    if n == 0 then
        return
    end
    const rank, merged = bpe.rank, bpe.merged
    -- Symbols form a linked list over the positions 1 to n, a merged away symbol has no id.
    const prev: { number } = table.create(n, 0)
    const nextOf: { number } = table.create(n, 0)
    for i = 1, n do
        prev[i] = i - 1
        nextOf[i] = if i < n then i + 1 else 0
    end
    const heap: { number } = {}
    local size = 0
    for i = 1, n - 1 do
        const r = rank[ids[i] * KEY + ids[i + 1]]
        if r ~= nil then
            size = heapPush(heap, size, r * POSITIONS + i)
        end
    end
    while size > 0 do
        local key: number
        key, size = heapPop(heap, size)
        const r = key // POSITIONS
        const pos = key % POSITIONS
        const second = nextOf[pos]
        if ids[pos] ~= 0 and second ~= 0 then
            const pair = ids[pos] * KEY + ids[second]
            if rank[pair] == r then
                ids[pos] = merged[pair]
                ids[second] = 0
                const after = nextOf[second]
                nextOf[pos] = after
                if after ~= 0 then
                    prev[after] = pos
                    const r2 = rank[ids[pos] * KEY + ids[after]]
                    if r2 ~= nil then
                        size = heapPush(heap, size, r2 * POSITIONS + pos)
                    end
                end
                const before = prev[pos]
                if before ~= 0 then
                    const r3 = rank[ids[before] * KEY + ids[pos]]
                    if r3 ~= nil then
                        size = heapPush(heap, size, r3 * POSITIONS + before)
                    end
                end
            end
        end
    end
    local i = 1
    while i ~= 0 do
        table.insert(out, ids[i])
        i = nextOf[i]
    end
end

--- Appends the ids of `text` as SentencePiece style text: spaces become the space mark first.
const function encodeSentence(bpe: Bpe, text: string, out: { number })
    encodeSpm(bpe, (string.gsub(text, " ", SPACE_MARK)), out)
end

return {
    load = load,
    encodePiece = encodePiece,
    encodeSentence = encodeSentence,
}
