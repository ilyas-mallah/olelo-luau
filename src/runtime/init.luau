--!native
-- What the engine asks of its host while it runs. The public face of `src/runtime`: its types and the modules a caller uses.

const types = require("@self/types")

export type Budget = types.Budget

return {
    budget = require("@self/budget"),
}
