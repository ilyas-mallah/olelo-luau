--!native
-- Llama 4 text with dense layers, as MobileLLM-R1.5 uses it: the Llama decoder with a parameter free norm on each head's
-- q and k. The converter lays out the q and k rows for the half split rotation, so the rotation is the Llama one.

const llama = require("./llama")
const types = require("./types")

const llama4: types.Arch = {
    name = "llama4",
    qkvBias = false,
    qkNorm = true,
    qkNormUnit = true,
    normPlusOne = llama.normPlusOne,
    embedScale = llama.embedScale,
    activation = llama.activation,
    window = llama.window,
    postNorms = false,
}

return llama4
