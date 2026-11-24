--!native
-- A small deterministic generator, PCG with a 32 bit state, done in doubles and bit32 so a seed gives the same stream
-- on every host.

const types = require("./types")

type Rng = types.Rng

const MULTIPLIER = 747796405
const INCREMENT = 2891336453

--- `a * b mod 2^32` for two unsigned 32 bit numbers, in pieces that stay exact in doubles.
const function mul32(a: number, b: number): number
    const high = bit32.rshift(a, 16)
    const low = bit32.band(a, 0xFFFF)
    return (bit32.lshift(bit32.band(high * b, 0xFFFF), 16) + low * b) % 4294967296
end

--- A generator seeded with `seed`.
const function new(seed: number): Rng
    const rng: Rng = { state = 0 }
    rng.state = (mul32(bit32.band(seed, 0xFFFFFFFF), MULTIPLIER) + INCREMENT) % 4294967296
    return rng
end

--- The next unsigned 32 bit value.
const function nextU32(rng: Rng): number
    const state = rng.state
    rng.state = (mul32(state, MULTIPLIER) + INCREMENT) % 4294967296
    const word = mul32(bit32.bxor(bit32.rshift(state, bit32.rshift(state, 28) + 4), state), 277803737)
    return bit32.bxor(bit32.rshift(word, 22), word)
end

--- A number in [0, 1) with 32 bits of resolution.
const function nextFloat(rng: Rng): number
    return nextU32(rng) / 4294967296
end

return {
    new = new,
    nextU32 = nextU32,
    nextFloat = nextFloat,
}
