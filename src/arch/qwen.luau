--!native
-- The Qwen families. Qwen2 and Qwen2.5 add a bias to the q, k and v projections, Qwen3 drops the bias and normalizes
-- each head's q and k before the rotation. Everything else is the Llama decoder.

const llama = require("./llama")
const types = require("./types")

const qwen2: types.Arch = {
    name = "qwen2",
    qkvBias = true,
    qkNorm = false,
    qkNormUnit = false,
    normPlusOne = llama.normPlusOne,
    embedScale = llama.embedScale,
    activation = llama.activation,
    window = llama.window,
    postNorms = false,
}

const qwen3: types.Arch = {
    name = "qwen3",
    qkvBias = false,
    qkNorm = true,
    qkNormUnit = false,
    normPlusOne = llama.normPlusOne,
    embedScale = llama.embedScale,
    activation = llama.activation,
    window = llama.window,
    postNorms = false,
}

return {
    qwen2 = qwen2,
    qwen3 = qwen3,
}
