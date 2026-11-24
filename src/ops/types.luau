--!native
-- Types shared by the modules of src/ops.

--- What `attend` needs to know about the layer.
export type Shape = {
    --- Query heads and KV heads this call computes. A worker that holds a share of a model has fewer than the model.
    heads: number,
    kvHeads: number,
    --- Query heads per KV head of the whole model, and the numbers in the whole model of the first query head and of
    --- the first KV head here. Query head `h` reads the cache of KV head `(headBase + h) // group - kvBase`.
    group: number,
    headBase: number,
    kvBase: number,
    headDim: number,
    hdPad: number,
    --- Positions the cache can hold.
    context: number,
    --- 1 / sqrt(headDim), or the model's own scale.
    scale: number,
    --- Positions a query may look back, nil for all of them.
    window: number?,
    --- The scales of the layer's Q8 cache, nil for an f32 cache (see `kvquant`).
    keyScales: buffer?,
    valueScales: buffer?,
}

--- The activations `gate` knows how to apply.
export type Activation = "silu" | "gelu_tanh"

--- Buffers the prepared activations are written into, sized for the widest row a model multiplies.
export type Scratch = {
    tokens: number,
    --- Longest stored row, a multiple of 32.
    maxK: number,
    --- Int8 values, tokens x maxK.
    q: buffer,
    --- Packed f64 values, tokens x maxK, or the splat form of the vector kernel, tokens x groups x 408 bytes.
    x: buffer,
    --- The token scales of the int8 form, the group scales of the int8g form, per group the sum and the scale of
    --- the packed form, or per group the scale in 3 lanes for the splat form.
    xs: buffer,
}

--- Cosines and sines of every position, f32, `half` values per position.
export type Table = {
    cos: buffer,
    sin: buffer,
    half: number,
    positions: number,
}

return {}
