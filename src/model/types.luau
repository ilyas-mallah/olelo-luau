--!native
-- Types shared by the modules of src/model.

const archTypes = require("@src/arch/types")
const config = require("@src/model/config")
const kernelsTypes = require("@src/kernels/types")
const opsTypes = require("@src/ops/types")
const tensorTypes = require("@src/tensor/types")

--- The weights of a short convolution layer: the input projection to the gates and the value, the filter of every channel
--- and the output projection.
export type ShortConv = {
    inProj: tensorTypes.Tensor,
    --- `hidden` filters of `config.convKernel` f32 taps each.
    taps: buffer,
    outProj: tensorTypes.Tensor,
}

--- The weights of one decoder layer. A layer has the attention projections or, in a model with short convolution
--- layers, `conv`. `attnNorm` is the norm before whichever of the two the layer has.
export type Layer = {
    attnNorm: buffer,
    q: tensorTypes.Tensor?,
    k: tensorTypes.Tensor?,
    v: tensorTypes.Tensor?,
    o: tensorTypes.Tensor?,
    conv: ShortConv?,
    mlpNorm: buffer,
    --- The norms after each block, for architectures with `postNorms`.
    postAttnNorm: buffer?,
    postMlpNorm: buffer?,
    --- Present for architectures with biases on q, k and v, or with a norm on each head's q and k.
    qBias: buffer?,
    kBias: buffer?,
    vBias: buffer?,
    qNorm: buffer?,
    kNorm: buffer?,
    gate: tensorTypes.Tensor,
    up: tensorTypes.Tensor,
    down: tensorTypes.Tensor,
}

--- The part of the decoder this model computes: all of it, or the share of one worker. Counts are local, bases are the
--- numbers of the first local head and KV head in the whole model.
export type Dims = {
    heads: number,
    kvHeads: number,
    headBase: number,
    kvBase: number,
    --- Query heads per KV head of the whole model.
    group: number,
    --- Columns of the MLP.
    inter: number,
}

export type Model = {
    config: config.Config,
    arch: archTypes.Arch,
    dims: Dims,
    --- The embedding table, nil in a worker's share, which gets its activations from the main worker.
    embed: tensorTypes.Tensor?,
    --- The output matrix, the embedding itself when the model ties them. A share has the rows of its vocabulary part.
    head: tensorTypes.Tensor,
    finalNorm: buffer,
    layers: { Layer },
    --- Kernels of the layer matrices for one token and for several.
    decode: kernelsTypes.Entry,
    prefill: kernelsTypes.Entry,
    --- The kernel of the output matrix.
    headKernel: kernelsTypes.Entry,
    --- The longest stored row a kernel that prepares its activations reads, 0 when every kernel reads f32.
    prepK: number,
    attend: (
        s: opsTypes.Shape,
        q: buffer,
        tokens: number,
        keys: buffer,
        values: buffer,
        pos0: number,
        out: buffer,
        outStride: number,
        scores: buffer,
        acc: { number }
    ) -> (),
    rope: opsTypes.Table,
    --- The positions of the local layers when their base frequency differs.
    ropeLocal: opsTypes.Table?,
    --- Positions the model was loaded for.
    context: number,
    --- The KV cache format of its sessions, "f32" or "q8".
    kv: string,
    --- How a Q8 cache stores its rows, "i8" or "v3" (see `ops/kvquant`), "f32" for the f32 cache.
    kvLayout: string,
    hdPad: number,
}

--- Bytes by purpose.
export type Estimate = {
    weights: number,
    cache: number,
    workspace: number,
    total: number,
}

--- The state of one conversation in a runner: how many positions it holds and how many it may hold.
export type Sequence = {
    length: number,
    context: number,
}

export type Hook = (layer: number, h: buffer, stride: number, tokens: number) -> ()

export type Runner = {
    --- True when `forward` waits for workers through the host's scheduler. A caller that wraps it in its own coroutine
    --- would take the scheduler's yield for its own, so it must run it in the thread it is in.
    yields: boolean,
    --- Rows of the logits a pass returns.
    vocab: number,
    open: (self: Runner) -> Sequence,
    --- Runs `ids[first]` to `ids[first + count - 1]` at the positions after the sequence's length, up to a pass of
    --- `chunk` tokens. Returns the logits of the last token when `wantLogits`, otherwise nil. The buffer is the
    --- runner's and holds until the next pass.
    forward: (
        self: Runner,
        seq: Sequence,
        ids: { number },
        first: number,
        count: number,
        wantLogits: boolean,
        hook: Hook?
    ) -> buffer?,
    --- Keeps the first `length` positions of the sequence.
    rewind: (self: Runner, seq: Sequence, length: number) -> (),
    --- Keeps the first `length` positions of the sequence under `key`, for sequences that start with the same tokens.
    save: (self: Runner, seq: Sequence, length: number, key: number) -> (),
    --- Makes the sequence, which is empty, hold the positions saved under `key`. Returns their number.
    restore: (self: Runner, seq: Sequence, key: number) -> number,
    --- Forgets what was saved under `key`.
    drop: (self: Runner, key: number) -> (),
    --- Releases what the sequence holds.
    close: (self: Runner, seq: Sequence) -> (),
    --- Ends the runner and its workers.
    shutdown: (self: Runner) -> (),
}

--- Scratch buffers, rows of `tokens` tokens. Every row length is padded as `config.pad` says.
export type Workspace = {
    --- The most tokens one forward pass takes, rounded up to the prefill kernel's tile tokens.
    tokens: number,
    hs: number,
    qs: number,
    ks: number,
    is: number,
    --- Residual stream.
    h: buffer,
    --- Normalized input of a block.
    xn: buffer,
    q: buffer,
    k: buffer,
    v: buffer,
    --- Rotated queries, [token][head][hdPad].
    qr: buffer,
    --- Rotated keys of a pass before they are quantized into a Q8 cache, [token][kv head][hdPad]. Nil for f32.
    kr: buffer?,
    attn: buffer,
    --- Output of the o and down projections.
    o: buffer,
    gate: buffer,
    up: buffer,
    act: buffer,
    --- One row for the last token's norm and the head product.
    lastIn: buffer,
    lastNorm: buffer,
    logits: buffer,
    scores: buffer,
    acc: { number },
    --- What the attention call needs to know about the layer, its window set per layer.
    shape: opsTypes.Shape,
    --- Output rows of the input projection of a short convolution layer, B, C and x of `hidden` values each, `bcxs` floats
    --- apart. Nil for models without such layers.
    bcx: buffer?,
    bcxs: number,
    --- Where activations are prepared for the kernels that read another form, nil when every kernel reads f32.
    prepared: opsTypes.Scratch?,
}

--- Keys and values of every layer. An f32 cache is [kv head][position][hdPad] f32. A Q8 cache is [kv head][position]
--- [headDim] int8 with a f32 scale per head and position in `keyScales` and `valueScales` (see `ops/kvquant`).
export type Cache = {
    keys: { buffer },
    values: { buffer },
    keyScales: { buffer }?,
    valueScales: { buffer }?,
    --- Per layer, the centre of the keys: `kvHeads` times `headDim` f32, set by the sequence's first positions.
    keyCentres: { buffer }?,
    --- Short convolution layers: per layer, `B * x` of every position, [position][hidden] f32 (see `ops/shortconv`).
    --- Nil for models without such layers. The buffers of layers of the other kind are empty, as are the keys and
    --- values of a convolution layer.
    conv: { buffer }?,
}

export type Session = {
    model: Model,
    cache: Cache,
    ws: Workspace,
    --- Positions filled so far.
    length: number,
    --- The most positions the session holds.
    context: number,
}

--- The cache of the first `length` positions of a session, kept apart from it, for a prefix that other sessions start
--- from. The layout is the session's, one run of `length` positions per KV head.
export type Snapshot = {
    length: number,
    keys: { buffer },
    values: { buffer },
    keyScales: { buffer }?,
    valueScales: { buffer }?,
    keyCentres: { buffer }?,
    conv: { buffer }?,
}

return {}
