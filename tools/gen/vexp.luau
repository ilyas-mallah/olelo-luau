-- The text both attention generators write after the header of a module: the vector `exp` and the edge store.

--- The constants, `vexp` and `storeEdge` as Luau source.
const TEXT = [[
const EXPLO = vector.create(-32, -32, -32)
const E2 = vector.create(1 / 2, 1 / 2, 1 / 2)
const E3 = vector.create(1 / 6, 1 / 6, 1 / 6)
const E4 = vector.create(1 / 24, 1 / 24, 1 / 24)
const E5 = vector.create(1 / 120, 1 / 120, 1 / 120)
const E6 = vector.create(1 / 720, 1 / 720, 1 / 720)
const E7 = vector.create(1 / 5040, 1 / 5040, 1 / 5040)
const E8 = vector.create(1 / 40320, 1 / 40320, 1 / 40320)
const ONE = vector.one

--- exp(x) of three lanes for x <= 0, clamped at -32: the Taylor polynomial of degree 8 at x / 64, then six squarings.
--- The relative error is about 5e-6.
const function vexp(x: vector): vector
    const t = vector.max(x, EXPLO) * 0.015625
    local p = E8 * t + E7
    p = p * t + E6
    p = p * t + E5
    p = p * t + E4
    p = p * t + E3
    p = p * t + E2
    p = p * t + ONE
    p = p * t + ONE
    p = p * p
    p = p * p
    p = p * p
    p = p * p
    p = p * p
    return p * p
end

--- Writes the lane at `at` of `out` with only the `left` values the head still has, 1 to 3.
const function storeEdge(out: buffer, v: vector, at: number, left: number)
    buffer.writef32(out, at, v.x)
    if left > 1 then
        buffer.writef32(out, at + 4, v.y)
    end
    if left > 2 then
        buffer.writef32(out, at + 8, v.z)
    end
end
]]

return {
    TEXT = TEXT,
}
