--!native
-- The model configuration the container header carries.

export type Config = {
    hidden: number,
    layers: number,
    heads: number,
    kvHeads: number,
    headDim: number,
    intermediate: number,
    vocab: number,
    ropeTheta: number,
    rmsEps: number,
    maxPositions: number,
    tieEmbeddings: boolean,
    activation: string,
    --- Gemma 3: positions a local layer looks back, every `slidingPattern`th layer is global and the local layers
    --- rotate with `ropeLocalTheta`. Nil for models whose layers are all alike.
    slidingWindow: number?,
    slidingPattern: number?,
    ropeLocalTheta: number?,
    --- The attention score divides by the square root of this, `headDim` when nil.
    attnScalar: number?,
    --- LFM2: the kind of each layer, "attention" or "conv" (a gated short convolution in place of attention). Nil for
    --- models whose layers are all attention.
    layerTypes: { string }?,
    --- LFM2: the taps of a short convolution, one more than the positions it looks back.
    convKernel: number?,
}

--- Every stored activation row is padded to a multiple of this, the least common multiple of every kernel's column
--- step and row tile, so one buffer serves as the output of one product and the input of the next.
const ACT_PAD = 24

--- Rounds `n` up to a multiple of `step`.
const function roundUp(n: number, step: number): number
    return (n + step - 1) // step * step
end

--- The stored length of an activation row of `n` values.
const function pad(n: number): number
    return roundUp(n, ACT_PAD)
end

--- True when layer `layer` (from 1) is a short convolution layer.
const function isConv(cfg: Config, layer: number): boolean
    const kinds = cfg.layerTypes
    return kinds ~= nil and kinds[layer] == "conv"
end

--- The stored length of one attention head in the KV cache, a multiple of 3 for the vector read.
const function headPad(headDim: number): number
    return roundUp(headDim, 3)
end

return {
    ACT_PAD = ACT_PAD,
    roundUp = roundUp,
    isConv = isConv,
    pad = pad,
    headPad = headPad,
}
