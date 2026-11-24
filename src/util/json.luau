--!native
-- A JSON decoder for the container header. It needs no host library, so the core can read a header anywhere.

type State = {
    text: string,
    pos: number,
}

const ESCAPES: { [string]: string } = {
    ['"'] = '"',
    ["\\"] = "\\",
    ["/"] = "/",
    b = "\b",
    f = "\f",
    n = "\n",
    r = "\r",
    t = "\t",
}

const function fail(s: State, message: string): never
    error(`json: {message} at byte {s.pos}`, 0)
end

const function skipSpace(s: State)
    const _, last = string.find(s.text, "^[ \t\r\n]*", s.pos)
    s.pos = (last or s.pos - 1) + 1
end

const function parseString(s: State): string
    -- The opening quote is at s.pos.
    local pos = s.pos + 1
    const parts: { string } = {}
    while true do
        const stop = string.find(s.text, '["\\]', pos)
        if stop == nil then
            return fail(s, "unterminated string")
        end
        const at = stop
        table.insert(parts, string.sub(s.text, pos, at - 1))
        if string.byte(s.text, at) == 34 then
            s.pos = at + 1
            return table.concat(parts)
        end
        const code = string.sub(s.text, at + 1, at + 1)
        if code == "u" then
            local cp = tonumber(string.sub(s.text, at + 2, at + 5), 16)
            if cp == nil then
                return fail(s, "bad unicode escape")
            end
            local used = 6
            if cp >= 0xD800 and cp < 0xDC00 and string.sub(s.text, at + 6, at + 7) == "\\u" then
                const low = tonumber(string.sub(s.text, at + 8, at + 11), 16)
                if low ~= nil and low >= 0xDC00 and low < 0xE000 then
                    cp = 0x10000 + (cp - 0xD800) * 0x400 + (low - 0xDC00)
                    used = 12
                end
            end
            table.insert(parts, utf8.char(cp))
            pos = at + used
        else
            const replacement = ESCAPES[code]
            if replacement == nil then
                return fail(s, "bad escape")
            end
            table.insert(parts, replacement)
            pos = at + 2
        end
    end
end

const function parseValue(s: State): any
    skipSpace(s)
    const first = string.byte(s.text, s.pos)
    if first == 123 then
        s.pos += 1
        const object: { [string]: any } = {}
        skipSpace(s)
        if string.byte(s.text, s.pos) == 125 then
            s.pos += 1
            return object
        end
        while true do
            skipSpace(s)
            if string.byte(s.text, s.pos) ~= 34 then
                fail(s, "expected a key")
            end
            const key = parseString(s)
            skipSpace(s)
            if string.byte(s.text, s.pos) ~= 58 then
                fail(s, "expected a colon")
            end
            s.pos += 1
            object[key] = parseValue(s)
            skipSpace(s)
            const separator = string.byte(s.text, s.pos)
            s.pos += 1
            if separator == 125 then
                return object
            elseif separator ~= 44 then
                fail(s, "expected a comma or a closing brace")
            end
        end
    elseif first == 91 then
        s.pos += 1
        const array: { any } = {}
        skipSpace(s)
        if string.byte(s.text, s.pos) == 93 then
            s.pos += 1
            return array
        end
        while true do
            table.insert(array, parseValue(s))
            skipSpace(s)
            const separator = string.byte(s.text, s.pos)
            s.pos += 1
            if separator == 93 then
                return array
            elseif separator ~= 44 then
                fail(s, "expected a comma or a closing bracket")
            end
        end
    elseif first == 34 then
        return parseString(s)
    end
    if string.sub(s.text, s.pos, s.pos + 3) == "true" then
        s.pos += 4
        return true
    elseif string.sub(s.text, s.pos, s.pos + 4) == "false" then
        s.pos += 5
        return false
    elseif string.sub(s.text, s.pos, s.pos + 3) == "null" then
        s.pos += 4
        return nil
    end
    const number = string.match(s.text, "^-?%d+%.?%d*[eE]?[+-]?%d*", s.pos)
    if number == nil or number == "" then
        return fail(s, "unexpected character")
    end
    s.pos += #number
    return tonumber(number)
end

--- Decodes a JSON document. Raises an error naming the byte position when the text is not valid JSON.
const function decode(text: string): any
    const s: State = { text = text, pos = 1 }
    const value = parseValue(s)
    skipSpace(s)
    if s.pos <= #text then
        fail(s, "trailing characters")
    end
    return value
end

return {
    decode = decode,
}
