--!native
-- Gemma 3: norms weighted by `1 + w` around both blocks, a norm on each head's q and k, the embeddings scaled by the
-- square root of the width, tanh GELU, and layers that alternate between local windows and the whole context.

const config = require("@src/model/config")
const types = require("./types")

const gemma3: types.Arch = {
    name = "gemma3",
    qkvBias = false,
    qkNorm = true,
    qkNormUnit = false,
    normPlusOne = true,
    embedScale = function(hidden: number): number
        return math.sqrt(hidden)
    end,
    activation = "gelu_tanh",
    window = function(layer: number, cfg: config.Config): number?
        const pattern = cfg.slidingPattern
        if pattern == nil or layer % pattern == 0 then
            return nil
        end
        return cfg.slidingWindow
    end,
    postNorms = true,
}

return gemma3
