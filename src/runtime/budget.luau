--!native
-- A time budget for work that must not hold a frame. `check` yields through the host when the slice is used up.

const host = require("@src/host")
const types = require("./types")

type Budget = types.Budget

--- A budget that yields every `slice` seconds of work, or never when `slice` is 0.
const function new(h: host.Host, slice: number): Budget
    return { host = h, slice = slice, since = h.clock() }
end

--- Call between blocks of work.
const function check(b: Budget)
    if b.slice <= 0 then
        return
    end
    if b.host.clock() - b.since >= b.slice then
        b.host.yield()
        b.since = b.host.clock()
    end
end

return {
    new = new,
    check = check,
}
