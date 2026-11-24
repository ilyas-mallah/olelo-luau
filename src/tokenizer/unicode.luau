--!native
-- Letter, number and whitespace classes of code points, from range tables the converter writes into the container.

const types = require("./types")

type Ranges = types.Ranges
type Tables = types.Tables

--- Class codes of `classify`.
const OTHER = 0
const LETTER = 1
const NUMBER = 2
const SPACE = 3

const function inRanges(r: Ranges, cp: number): boolean
    local lo, hi = 1, #r.lo
    while lo <= hi do
        const mid = (lo + hi) // 2
        if cp < r.lo[mid] then
            hi = mid - 1
        elseif cp > r.hi[mid] then
            lo = mid + 1
        else
            return true
        end
    end
    return false
end

const function readRanges(blob: buffer, at: number): (Ranges, number)
    const count = buffer.readu32(blob, at)
    const lo = table.create<<number>>(count)
    const hi = table.create<<number>>(count)
    for i = 1, count do
        lo[i] = buffer.readu32(blob, at + 4 + (i - 1) * 8)
        hi[i] = buffer.readu32(blob, at + 8 + (i - 1) * 8)
    end
    return { lo = lo, hi = hi }, at + 4 + count * 8
end

--- The class of `cp`: LETTER, NUMBER, SPACE or OTHER.
const function classify(t: Tables, cp: number): number
    if cp < 128 then
        return t.ascii[cp + 1]
    end
    if inRanges(t.letters, cp) then
        return LETTER
    elseif inRanges(t.numbers, cp) then
        return NUMBER
    elseif inRanges(t.spaces, cp) then
        return SPACE
    end
    return OTHER
end

--- Reads the tables from the container's `unicode` blob: letters, numbers, then whitespace.
const function load(blob: buffer): Tables
    const letters, afterLetters = readRanges(blob, 0)
    const numbers, afterNumbers = readRanges(blob, afterLetters)
    const spaces = readRanges(blob, afterNumbers)
    const tables: Tables = { letters = letters, numbers = numbers, spaces = spaces, ascii = {} }
    for cp = 0, 127 do
        tables.ascii[cp + 1] = OTHER
        if inRanges(letters, cp) then
            tables.ascii[cp + 1] = LETTER
        elseif inRanges(numbers, cp) then
            tables.ascii[cp + 1] = NUMBER
        elseif inRanges(spaces, cp) then
            tables.ascii[cp + 1] = SPACE
        end
    end
    return tables
end

return {
    OTHER = OTHER,
    LETTER = LETTER,
    NUMBER = NUMBER,
    SPACE = SPACE,
    classify = classify,
    load = load,
}
