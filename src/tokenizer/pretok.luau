--!native
-- Splits text into the pieces byte level BPE runs on. The GPT-2 split is the regular expression
--   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
-- done by a scanner over code point classes, since Luau patterns work on bytes. With `splitDigits` every number
-- character first becomes its own piece, and the text between them is split by the expression.
--
-- The Qwen split is
--   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
-- which `scanQwen` follows alternative by alternative. The Llama 3 split is the same with `\p{N}{1,3}` for `\p{N}`, so
-- a run of numbers breaks into pieces of up to three.

const unicode = require("./unicode")
const types = require("./types")

const NUMBER = unicode.NUMBER
const SPACE = unicode.SPACE
const LETTER = unicode.LETTER
const OTHER = unicode.OTHER

const SPACE_CP = 32
const APOSTROPHE = 39

--- Appends the pieces of the code points in [`first`, `stop`) to `out`. `starts` holds the byte offset of each code
--- point and one past the last.
const function scan(
    text: string,
    cps: { number },
    cls: { number },
    starts: { number },
    first: number,
    stop: number,
    out: { string }
)
    local i = first
    while i < stop do
        local count = 0
        const c = cps[i]
        if c == APOSTROPHE and i + 1 < stop then
            const n1 = cps[i + 1]
            if n1 == 115 or n1 == 116 or n1 == 109 or n1 == 100 then
                count = 2
            elseif i + 2 < stop then
                const n2 = cps[i + 2]
                if (n1 == 114 and n2 == 101) or (n1 == 118 and n2 == 101) or (n1 == 108 and n2 == 108) then
                    count = 3
                end
            end
        end
        if count == 0 then
            local j = i
            if c == SPACE_CP then
                j = i + 1
            end
            if j < stop and cls[j] ~= SPACE then
                const kind = cls[j]
                local k = j + 1
                while k < stop and cls[k] == kind do
                    k += 1
                end
                count = k - i
            end
        end
        if count == 0 then
            -- Whitespace: all of it, except the last character when a non space follows.
            local j = i
            while j < stop and cls[j] == SPACE do
                j += 1
            end
            if j == stop or j - 1 == i then
                count = j - i
            else
                count = j - 1 - i
            end
        end
        table.insert(out, string.sub(text, starts[i], starts[i + count] - 1))
        i += count
    end
end

const CR, LF = 13, 10

--- The ASCII lower case of a code point, other code points unchanged.
const function lowerAscii(c: number): number
    return if c >= 65 and c <= 90 then c + 32 else c
end

--- Appends the pieces of [`first`, `stop`) split by the Qwen expression, or the Llama 3 one when `digitRun` is 3: it is
--- the longest run of number characters one piece takes.
const function scanQwen(
    text: string,
    cps: { number },
    cls: { number },
    starts: { number },
    first: number,
    stop: number,
    out: { string },
    digitRun: number
)
    local i = first
    while i < stop do
        local count = 0
        const c = cps[i]
        -- 1. A contraction, in any case.
        if c == APOSTROPHE and i + 1 < stop then
            const n1 = lowerAscii(cps[i + 1])
            if n1 == 115 or n1 == 116 or n1 == 109 or n1 == 100 then
                count = 2
            elseif i + 2 < stop then
                const n2 = lowerAscii(cps[i + 2])
                if (n1 == 114 and n2 == 101) or (n1 == 118 and n2 == 101) or (n1 == 108 and n2 == 108) then
                    count = 3
                end
            end
        end
        -- 2. One optional character that is not a line break, letter or number, then letters.
        if count == 0 then
            local j = i
            if cls[i] ~= LETTER and cls[i] ~= NUMBER and c ~= CR and c ~= LF then
                j = i + 1
            end
            if j < stop and cls[j] == LETTER then
                local k = j + 1
                while k < stop and cls[k] == LETTER do
                    k += 1
                end
                count = k - i
            end
        end
        -- 3. A run of number characters, one for Qwen and up to three for Llama 3.
        if count == 0 and cls[i] == NUMBER then
            count = 1
            while count < digitRun and i + count < stop and cls[i + count] == NUMBER do
                count += 1
            end
        end
        -- 4. An optional space, a run of other characters, then line breaks.
        if count == 0 then
            local j = i
            if c == SPACE_CP then
                j = i + 1
            end
            if j < stop and cls[j] == OTHER then
                local k = j + 1
                while k < stop and cls[k] == OTHER do
                    k += 1
                end
                while k < stop and (cps[k] == CR or cps[k] == LF) do
                    k += 1
                end
                count = k - i
            end
        end
        if count == 0 then
            local j = i
            while j < stop and cls[j] == SPACE do
                j += 1
            end
            -- 5. Whitespace up to its last line break.
            local last = j - 1
            while last >= i and cps[last] ~= CR and cps[last] ~= LF do
                last -= 1
            end
            if last >= i then
                count = last - i + 1
            elseif j == stop or j - 1 == i then
                -- 7. A single whitespace, or 6. all but the last of a run when a non space follows.
                count = j - i
            else
                count = j - 1 - i
            end
        end
        table.insert(out, string.sub(text, starts[i], starts[i + count] - 1))
        i += count
    end
end

--- Splits `text`, which must be valid UTF-8, into pieces. `kind` is `gpt2`, `qwen2` or `llama3`.
const function split(t: types.Tables, text: string, kind: string, splitDigits: boolean): { string }
    const cps: { number } = {}
    const cls: { number } = {}
    const starts: { number } = {}
    local n = 0
    for position, cp in utf8.codes(text) do
        n += 1
        cps[n] = cp
        cls[n] = unicode.classify(t, cp)
        starts[n] = position
    end
    starts[n + 1] = #text + 1
    const out: { string } = {}
    if kind == "qwen2" or kind == "llama3" then
        scanQwen(text, cps, cls, starts, 1, n + 1, out, if kind == "llama3" then 3 else 1)
        return out
    end
    if not splitDigits then
        scan(text, cps, cls, starts, 1, n + 1, out)
        return out
    end
    local segment = 1
    for i = 1, n do
        if cls[i] == NUMBER then
            scan(text, cps, cls, starts, segment, i, out)
            table.insert(out, string.sub(text, starts[i], starts[i + 1] - 1))
            segment = i + 1
        end
    end
    scan(text, cps, cls, starts, segment, n + 1, out)
    return out
end

return {
    split = split,
}
