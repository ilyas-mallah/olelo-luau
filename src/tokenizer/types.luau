--!native
-- Types shared by the modules of src/tokenizer.

export type Bpe = {
    --- Token text (bytes mapped to printable characters) to id.
    vocab: { [string]: number },
    --- The raw bytes a token stands for.
    idToBytes: { string },
    --- The id of the single byte token of each byte value, indexed by byte + 1.
    byteId: { number },
    --- Merge rank and result of the pair `a * KEY + b`.
    rank: { [number]: number },
    merged: { [number]: number },
    cache: { [string]: { number } },
    cacheSize: number,
    size: number,
    --- SentencePiece style.
    spm: boolean,
    --- A piece that is a whole token is that token and skips the merges (Llama 3).
    ignoreMerges: boolean,
    --- The character each byte maps to, indexed by byte + 1, for byte level vocabularies.
    byteChar: { string },
}

export type Special = {
    id: number,
    content: string,
}

export type Tokenizer = {
    bpe: Bpe,
    unicode: Tables,
    pretokenizer: string,
    splitDigits: boolean,
    specials: { Special },
    --- The specials whose text starts with a byte, longest first, and a pattern that finds the next such byte.
    specialsByByte: { [number]: { Special } },
    specialStart: string,
    --- The beginning of sequence token the model expects before a text, nil for none.
    bos: number?,
    --- Token ids that end generation.
    stopIds: { [number]: boolean },
    chat: Spec,
}

--- A conversation message.
export type Message = {
    role: string,
    content: string,
}

--- The template of the container header's `chat` entry.
export type Spec = {
    kind: string,
    --- The system message added when a conversation does not start with one, nil for none. ChatML and Llama 3.
    defaultSystem: string?,
    --- Text after the assistant header of a generation prompt, nil for none. Qwen3 with thinking off has an empty
    --- think block there.
    generationPrefix: string?,
    --- Text that opens every conversation, the model's beginning of sequence token. Nil for none.
    opening: string?,
}

--- Sorted, non overlapping ranges as two parallel arrays.
export type Ranges = {
    lo: { number },
    hi: { number },
}

export type Tables = {
    letters: Ranges,
    numbers: Ranges,
    spaces: Ranges,
    --- The class of every code point below 128.
    ascii: { number },
}

return {}
