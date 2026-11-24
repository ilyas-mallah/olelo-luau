--!native
-- The Llama family decoder: SmolLM2 and its relatives. Every other family states how it differs in the same fields.

const config = require("@src/model/config")
const types = require("./types")

type Arch = types.Arch

const llama: Arch = {
    name = "llama",
    qkvBias = false,
    qkNorm = false,
    qkNormUnit = false,
    normPlusOne = false,
    embedScale = function(_hidden: number): number
        return 1
    end,
    activation = "silu",
    window = function(_layer: number, _cfg: config.Config): number?
        return nil
    end,
    postNorms = false,
}

return llama
