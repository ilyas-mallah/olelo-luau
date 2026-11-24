--!native
-- Choosing the next token. The public face of `src/sampling`: its types and the modules a caller uses.

const types = require("@self/types")

export type Options = types.Options
export type Rng = types.Rng

return {
    rng = require("@self/rng"),
    sampler = require("@self/sampler"),
}
