--!native
-- LFM2 and LFM2.5: most layers are a gated short convolution instead of attention (`config.layerTypes` says which), and
-- the attention layers normalize each head's q and k before the rotation like Qwen3. The MLP is the Llama one.

const llama = require("./llama")
const types = require("./types")

const lfm2: types.Arch = {
    name = "lfm2",
    qkvBias = false,
    qkNorm = true,
    qkNormUnit = false,
    normPlusOne = llama.normPlusOne,
    embedScale = llama.embedScale,
    activation = llama.activation,
    window = llama.window,
    postNorms = false,
}

return lfm2
