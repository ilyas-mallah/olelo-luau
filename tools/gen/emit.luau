-- A line emitter with indentation, and the helpers that write an expression with a constant offset.

--- Lines written so far and the current indentation depth.
export type Emitter = {
    lines: { string },
    depth: number,
}

const INDENT = "    "

--- An empty emitter.
const function new(): Emitter
    return { lines = {}, depth = 0 }
end

--- Appends one line at the current depth.
const function line(e: Emitter, text: string)
    table.insert(e.lines, string.rep(INDENT, e.depth) .. text)
end

--- Opens a block: writes `head` and indents what follows.
const function open(e: Emitter, head: string)
    line(e, head)
    e.depth += 1
end

--- Closes a block: dedents and writes `end`.
const function close(e: Emitter)
    e.depth -= 1
    line(e, "end")
end

--- The text of everything written, one line each.
const function text(e: Emitter): string
    return table.concat(e.lines, "\n") .. "\n"
end

--- `base` with `n` added, written as `base` alone when `n` is 0 so no `+ 0` reaches the output.
const function plus(base: string, n: number): string
    if n == 0 then
        return base
    end
    return `{base} + {n}`
end

--- Like `plus`, but parenthesized when it adds something, for use as a factor.
const function plusFactor(base: string, n: number): string
    if n == 0 then
        return base
    end
    return `({base} + {n})`
end

return {
    new = new,
    line = line,
    open = open,
    close = close,
    text = text,
    plus = plus,
    plusFactor = plusFactor,
}
