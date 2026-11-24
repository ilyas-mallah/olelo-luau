--!native
-- One forward pass of the decoder over a chunk of tokens: embed, then per layer attention and MLP with residuals, then
-- the final norm and the output product for the last token. The block is written once and the architecture plug-in
-- says how a family differs.

const config = require("./config")
const model = require("@src/model")
const embed = require("@src/ops/embed")
const kvquant = require("@src/ops/kvquant")
const linear = require("@src/ops/linear")
const mlp = require("@src/ops/mlp")
const norm = require("@src/ops/norm")
const prepare = require("@src/ops/prepare")
const rope = require("@src/ops/rope")
const shortconv = require("@src/ops/shortconv")
const kernelsTypes = require("@src/kernels/types")
const runtimeTypes = require("@src/runtime/types")
const tensorTypes = require("@src/tensor/types")
const types = require("./types")

--- Called after each layer with the residual stream, for tests and tools. Receives the layer number from 1, the
--- residual buffer, its row stride in floats and the number of tokens of the pass.
export type LayerHook = (layer: number, h: buffer, stride: number, tokens: number) -> ()

--- The kernel for a pass of `count` tokens and the tokens the pass computes, `count` rounded up to its tile.
const function passEntry(m: types.Model, count: number): (kernelsTypes.Entry, number)
    const entry = if count == 1 then m.decode else m.prefill
    return entry, config.roundUp(count, entry.S)
end

--- The activations a kernel reads, prepared from `tokens` rows of `cols` values at stride `stride`, for `t`.
const function prep(
    s: types.Session,
    e: kernelsTypes.Entry,
    src: buffer,
    stride: number,
    cols: number,
    t: tensorTypes.Tensor,
    tokens: number
): (buffer, buffer?)
    return prepare.run(e, s.ws.prepared, src, stride, cols, t.kpad, tokens)
end

--- The attention block of layer `l` for `count` tokens at the positions `pos0` and after, up to the output projection.
--- It reads the block's normalized input from `ws.xn`, writes the keys and values to the cache and leaves the product
--- with `o` in `ws.o`. A worker that holds a share of the model leaves its part of that sum, which the main worker adds
--- up.
const function attnBlock(s: types.Session, l: number, pos0: number, count: number, b: runtimeTypes.Budget?)
    const m = s.model
    const cfg = m.config
    const dims = m.dims
    const ws = s.ws
    const entry, processed = passEntry(m, count)
    const hs, qs, ks = ws.hs, ws.qs, ws.ks
    const hdPad = m.hdPad
    const plusOne = m.arch.normPlusOne
    const layer = m.layers[l]
    const wq, wk, wv, wo = layer.q, layer.k, layer.v, layer.o
    if wq == nil or wk == nil or wv == nil or wo == nil then
        error(`layer {l} has no attention projections`, 0)
    end
    const xa, xas = prep(s, entry, ws.xn, hs, cfg.hidden, wq, processed)
    linear.apply(entry, wq, xa, xas, ws.q, qs, processed, b)
    linear.apply(entry, wk, xa, xas, ws.k, ks, processed, b)
    linear.apply(entry, wv, xa, xas, ws.v, ks, processed, b)
    const qBias, kBias, vBias = layer.qBias, layer.kBias, layer.vBias
    if qBias ~= nil and kBias ~= nil and vBias ~= nil then
        norm.addBias(ws.q, qs, qBias, dims.heads * cfg.headDim, count)
        norm.addBias(ws.k, ks, kBias, dims.kvHeads * cfg.headDim, count)
        norm.addBias(ws.v, ks, vBias, dims.kvHeads * cfg.headDim, count)
    end
    const qNorm, kNorm = layer.qNorm, layer.kNorm
    if qNorm ~= nil and kNorm ~= nil then
        norm.rmsHeads(ws.q, qs, qNorm, dims.heads, cfg.headDim, count, cfg.rmsEps, plusOne)
        norm.rmsHeads(ws.k, ks, kNorm, dims.kvHeads, cfg.headDim, count, cfg.rmsEps, plusOne)
    end
    const window = m.arch.window(l, cfg)
    const ropeLocal = m.ropeLocal
    const ropeTable = if window ~= nil and ropeLocal ~= nil then ropeLocal else m.rope
    rope.apply(ropeTable, ws.q, qs, dims.heads, cfg.headDim, count, pos0, ws.qr, 0, dims.heads * hdPad * 4, hdPad * 4)
    const shape = ws.shape
    const keyScales = s.cache.keyScales
    if keyScales == nil then
        const stride = s.context * hdPad * 4
        rope.apply(
            ropeTable,
            ws.k,
            ks,
            dims.kvHeads,
            cfg.headDim,
            count,
            pos0,
            s.cache.keys[l],
            pos0 * hdPad * 4,
            hdPad * 4,
            stride
        )
        rope.copy(ws.v, ks, dims.kvHeads, cfg.headDim, count, s.cache.values[l], pos0 * hdPad * 4, hdPad * 4, stride)
    else
        -- A Q8 cache quantizes the rotated keys, less the centre of the first keys of the sequence, and the values
        -- row by row, one scale per head and position.
        const kr, valueScales, keyCentres = ws.kr, s.cache.valueScales, s.cache.keyCentres
        assert(kr ~= nil and valueScales ~= nil and keyCentres ~= nil, "a Q8 cache has its scratch, scales and centres")
        const kvHeadStride = dims.kvHeads * hdPad
        rope.apply(ropeTable, ws.k, ks, dims.kvHeads, cfg.headDim, count, pos0, kr, 0, kvHeadStride * 4, hdPad * 4)
        const centre = keyCentres[l]
        if pos0 == 0 then
            kvquant.centre(kr, kvHeadStride, hdPad, dims.kvHeads, cfg.headDim, count, centre)
        end
        kvquant.store(
            kr,
            kvHeadStride,
            hdPad,
            dims.kvHeads,
            cfg.headDim,
            count,
            s.cache.keys[l],
            keyScales[l],
            pos0,
            s.context,
            centre,
            m.kvLayout
        )
        kvquant.store(
            ws.v,
            ks,
            cfg.headDim,
            dims.kvHeads,
            cfg.headDim,
            count,
            s.cache.values[l],
            valueScales[l],
            pos0,
            s.context,
            nil,
            m.kvLayout
        )
        shape.keyScales = keyScales[l]
        shape.valueScales = valueScales[l]
    end
    shape.window = window
    m.attend(shape, ws.qr, count, s.cache.keys[l], s.cache.values[l], pos0, ws.attn, qs, ws.scores, ws.acc)
    const oa, oas = prep(s, entry, ws.attn, qs, wo.cols, wo, processed)
    linear.apply(entry, wo, oa, oas, ws.o, hs, processed, b)
end

--- The short convolution block of layer `l` for `count` tokens at the positions `pos0` and after, in place of the
--- attention block: it reads the normalized input from `ws.xn`, keeps `B * x` of the new positions in the layer's
--- history and leaves the product with the output projection in `ws.o`. The convolved row overwrites `ws.xn`, whose
--- input the projection has consumed by then.
const function convBlock(s: types.Session, l: number, pos0: number, count: number, b: runtimeTypes.Budget?)
    const m = s.model
    const cfg = m.config
    const ws = s.ws
    const entry, processed = passEntry(m, count)
    const hs = ws.hs
    const conv, history, bcx, kernel = m.layers[l].conv, s.cache.conv, ws.bcx, cfg.convKernel
    if conv == nil or history == nil or bcx == nil or kernel == nil then
        error(`layer {l} is not a short convolution layer`, 0)
    end
    const xa, xas = prep(s, entry, ws.xn, hs, cfg.hidden, conv.inProj, processed)
    linear.apply(entry, conv.inProj, xa, xas, bcx, ws.bcxs, processed, b)
    shortconv.run(bcx, ws.bcxs, cfg.hidden, conv.taps, kernel, history[l], pos0, count, ws.xn, hs)
    const oa, oas = prep(s, entry, ws.xn, hs, cfg.hidden, conv.outProj, processed)
    linear.apply(entry, conv.outProj, oa, oas, ws.o, hs, processed, b)
end

--- The MLP block of layer `l` for `count` tokens, from the normalized input in `ws.xn` to the product with `down` in
--- `ws.o`, or a worker's part of that sum.
const function mlpBlock(s: types.Session, l: number, count: number, b: runtimeTypes.Budget?)
    const m = s.model
    const ws = s.ws
    const entry, processed = passEntry(m, count)
    const hs, is = ws.hs, ws.is
    const layer = m.layers[l]
    const ga, gas = prep(s, entry, ws.xn, hs, m.config.hidden, layer.gate, processed)
    linear.apply(entry, layer.gate, ga, gas, ws.gate, is, processed, b)
    linear.apply(entry, layer.up, ga, gas, ws.up, is, processed, b)
    mlp.gated(m.arch.activation, ws.gate, is, ws.up, is, ws.act, is, m.dims.inter, count)
    const da, das = prep(s, entry, ws.act, is, m.dims.inter, layer.down, processed)
    linear.apply(entry, layer.down, da, das, ws.o, hs, processed, b)
end

--- The output product for the normalized last row in `ws.lastNorm`: the model's logits, or a worker's part of them,
--- in `ws.logits`.
const function headProduct(s: types.Session, b: runtimeTypes.Budget?): buffer
    const m = s.model
    const ws = s.ws
    const ha, has = prep(s, m.headKernel, ws.lastNorm, ws.hs, m.config.hidden, m.head, 1)
    linear.apply(m.headKernel, m.head, ha, has, ws.logits, m.head.rowsPadded, 1, b)
    return ws.logits
end

--- Runs `ids[first]` to `ids[first + count - 1]` through the model at the positions after the session's current length.
--- With `wantLogits` it returns the session's logits buffer holding the output row of the last token, otherwise nil.
--- `count` is at most the session's workspace tokens.
const function forward(
    s: types.Session,
    ids: { number },
    first: number,
    count: number,
    wantLogits: boolean,
    b: runtimeTypes.Budget?,
    hook: LayerHook?
): buffer?
    const m = s.model
    const cfg = m.config
    const ws = s.ws
    const pos0 = s.length
    assert(count >= 1 and count <= ws.tokens, "a forward pass takes between 1 and the workspace tokens")
    assert(pos0 + count <= s.context, `the context of {s.context} positions is full`)
    const hs = ws.hs
    const plusOne = m.arch.normPlusOne
    const embedding = m.embed
    assert(embedding ~= nil, "the main worker holds the embedding")

    const scale = m.arch.embedScale(cfg.hidden)
    for t = 0, count - 1 do
        embed.row(embedding, ids[first + t], ws.h, t * hs * 4)
    end
    if scale ~= 1 then
        for t = 0, count - 1 do
            for i = 0, cfg.hidden - 1 do
                const at = (t * hs + i) * 4
                buffer.writef32(ws.h, at, buffer.readf32(ws.h, at) * scale)
            end
        end
    end

    for l = 1, cfg.layers do
        const layer = m.layers[l]
        norm.rms(ws.h, hs, layer.attnNorm, ws.xn, hs, cfg.hidden, count, cfg.rmsEps, plusOne)
        if layer.conv ~= nil then
            convBlock(s, l, pos0, count, b)
        else
            attnBlock(s, l, pos0, count, b)
        end
        const postAttnNorm = layer.postAttnNorm
        if postAttnNorm ~= nil then
            norm.rms(ws.o, hs, postAttnNorm, ws.o, hs, cfg.hidden, count, cfg.rmsEps, plusOne)
        end
        norm.add(ws.h, hs, ws.o, hs, cfg.hidden, count)

        norm.rms(ws.h, hs, layer.mlpNorm, ws.xn, hs, cfg.hidden, count, cfg.rmsEps, plusOne)
        mlpBlock(s, l, count, b)
        const postMlpNorm = layer.postMlpNorm
        if postMlpNorm ~= nil then
            norm.rms(ws.o, hs, postMlpNorm, ws.o, hs, cfg.hidden, count, cfg.rmsEps, plusOne)
        end
        norm.add(ws.h, hs, ws.o, hs, cfg.hidden, count)
        if hook ~= nil then
            hook(l, ws.h, hs, count)
        end
    end
    s.length = pos0 + count
    if not wantLogits then
        return nil
    end
    buffer.copy(ws.lastIn, 0, ws.h, (count - 1) * hs * 4, hs * 4)
    norm.rms(ws.lastIn, hs, m.finalNorm, ws.lastNorm, hs, cfg.hidden, 1, cfg.rmsEps, plusOne)
    return headProduct(s, b)
end

return {
    forward = forward,
    attnBlock = attnBlock,
    convBlock = convBlock,
    mlpBlock = mlpBlock,
    headProduct = headProduct,
}
