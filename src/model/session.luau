--!native
-- The state of one conversation: its KV cache, the scratch buffers of a forward pass and how many positions it holds.

const config = require("./config")
const kvquant = require("@src/ops/kvquant")
const model = require("@src/model")
const prepare = require("@src/ops/prepare")
const types = require("./types")

type Cache = types.Cache
type Session = types.Session
type Snapshot = types.Snapshot
type Workspace = types.Workspace

--- Bytes of the cache of one attention layer, keys and values together, for `kv` the cache format ("f32" or "q8") and,
--- for "q8", `layout` the row layout (`ops/kvquant`, "i8" when nil).
const function layerCacheBytes(cfg: config.Config, context: number, kv: string, layout: string?): number
    if kv == "q8" then
        return 2 * cfg.kvHeads * context * (kvquant.rowBytes(cfg.headDim, layout or "i8") + 4)
            + cfg.kvHeads * cfg.headDim * 4
    end
    return 2 * cfg.kvHeads * context * config.headPad(cfg.headDim) * 4
end

--- Bytes of the cache of every layer of the model.
const function cacheBytes(cfg: config.Config, context: number, kv: string, layout: string?): number
    local total = 0
    for l = 1, cfg.layers do
        total += if config.isConv(cfg, l) then context * cfg.hidden * 4 else layerCacheBytes(cfg, context, kv, layout)
    end
    return total
end

--- Bytes of the scratch buffers for `tokens` tokens per pass.
const function workspaceBytes(cfg: config.Config, tokens: number, vocabRows: number, prepK: number, kv: string): number
    const hs, qs, ks, is =
        config.pad(cfg.hidden),
        config.pad(cfg.heads * cfg.headDim),
        config.pad(cfg.kvHeads * cfg.headDim),
        config.pad(cfg.intermediate)
    const floats = tokens * (3 * hs + 2 * qs + 2 * ks + cfg.heads * config.headPad(cfg.headDim) + 3 * is)
        + 3 * hs
        + vocabRows
        + (if kv == "q8" then tokens * cfg.kvHeads * config.headPad(cfg.headDim) else 0)
        + (if cfg.layerTypes ~= nil then tokens * config.pad(3 * cfg.hidden) else 0)
    return floats * 4 + (if prepK > 0 then prepare.bytes(tokens, prepK) else 0)
end

--- A session for up to `m.context` positions that takes `tokens` tokens per forward pass.
const function new(m: types.Model, tokens: number): Session
    const cfg = m.config
    const tp = config.roundUp(tokens, m.prefill.S)
    const dims = m.dims
    const hs, qs, ks, is =
        config.pad(cfg.hidden),
        config.pad(dims.heads * cfg.headDim),
        config.pad(dims.kvHeads * cfg.headDim),
        config.pad(dims.inter)
    const quantized = m.kv == "q8"
    const perLayer = if quantized
        then dims.kvHeads * m.context * kvquant.rowBytes(cfg.headDim, m.kvLayout)
        else dims.kvHeads * m.context * m.hdPad * 4
    const keys = table.create<<buffer>>(cfg.layers)
    const values = table.create<<buffer>>(cfg.layers)
    const keyScales = table.create<<buffer>>(cfg.layers)
    const valueScales = table.create<<buffer>>(cfg.layers)
    const keyCentres = table.create<<buffer>>(cfg.layers)
    const conv = table.create<<buffer>>(cfg.layers)
    for l = 1, cfg.layers do
        const isConv = config.isConv(cfg, l)
        keys[l] = buffer.create(if isConv then 0 else perLayer)
        values[l] = buffer.create(if isConv then 0 else perLayer)
        conv[l] = buffer.create(if isConv then m.context * cfg.hidden * 4 else 0)
        if quantized then
            keyScales[l] = buffer.create(if isConv then 0 else dims.kvHeads * m.context * 4)
            valueScales[l] = buffer.create(if isConv then 0 else dims.kvHeads * m.context * 4)
            keyCentres[l] = buffer.create(if isConv then 0 else dims.kvHeads * cfg.headDim * 4)
        end
    end
    const ws: Workspace = {
        tokens = tp,
        hs = hs,
        qs = qs,
        ks = ks,
        is = is,
        h = buffer.create(tp * hs * 4),
        xn = buffer.create(tp * hs * 4),
        q = buffer.create(tp * qs * 4),
        k = buffer.create(tp * ks * 4),
        v = buffer.create(tp * ks * 4),
        qr = buffer.create(tp * dims.heads * m.hdPad * 4),
        kr = if quantized then buffer.create(tp * dims.kvHeads * m.hdPad * 4) else nil,
        attn = buffer.create(tp * qs * 4),
        o = buffer.create(tp * hs * 4),
        gate = buffer.create(tp * is * 4),
        up = buffer.create(tp * is * 4),
        act = buffer.create(tp * is * 4),
        lastIn = buffer.create(hs * 4),
        lastNorm = buffer.create(hs * 4),
        logits = buffer.create(m.head.rowsPadded * 4),
        scores = buffer.create(m.context * 4),
        acc = table.create(cfg.headDim, 0),
        shape = {
            heads = dims.heads,
            kvHeads = dims.kvHeads,
            group = dims.group,
            headBase = dims.headBase,
            kvBase = dims.kvBase,
            headDim = cfg.headDim,
            hdPad = m.hdPad,
            context = m.context,
            scale = 1 / math.sqrt(cfg.attnScalar or cfg.headDim),
            window = nil,
        },
        bcx = if cfg.layerTypes ~= nil then buffer.create(tp * config.pad(3 * cfg.hidden) * 4) else nil,
        bcxs = config.pad(3 * cfg.hidden),
        prepared = if m.prepK > 0 then prepare.new(tp, m.prepK) else nil,
    }
    const cache: Cache = {
        keys = keys,
        values = values,
        keyScales = if quantized then keyScales else nil,
        valueScales = if quantized then valueScales else nil,
        keyCentres = if quantized then keyCentres else nil,
        conv = if cfg.layerTypes ~= nil then conv else nil,
    }
    return { model = m, cache = cache, ws = ws, length = 0, context = m.context }
end

--- Forgets the conversation, keeping the buffers.
const function reset(s: Session)
    s.length = 0
end

--- Bytes one position of one KV head takes in the keys of this cache, and in its scales (0 for an f32 cache).
const function rowBytes(s: Session): (number, number)
    return if s.cache.keyScales ~= nil
        then kvquant.rowBytes(s.model.config.headDim, s.model.kvLayout)
        else s.model.hdPad * 4,
        if s.cache.keyScales ~= nil then 4 else 0
end

--- Copies `length` positions of every KV head from `src` to `dst`, whose heads are `srcContext` and `dstContext`
--- positions long, `row` bytes a position.
const function copyHeads(
    dst: buffer,
    dstContext: number,
    src: buffer,
    srcContext: number,
    heads: number,
    length: number,
    row: number
)
    for g = 0, heads - 1 do
        buffer.copy(dst, g * dstContext * row, src, g * srcContext * row, length * row)
    end
end

--- Keeps the first `length` positions of `s`.
const function snapshot(s: Session, length: number): Snapshot
    assert(length >= 1 and length <= s.length, "a snapshot holds positions the session has")
    const heads = s.model.dims.kvHeads
    const row, scaleRow = rowBytes(s)
    const layers = s.model.config.layers
    const keyScales = table.create<<buffer>>(layers)
    const valueScales = table.create<<buffer>>(layers)
    const keyCentres = table.create<<buffer>>(layers)
    const conv = table.create<<buffer>>(layers)
    const out: Snapshot = {
        length = length,
        keys = {},
        values = {},
        conv = if s.cache.conv ~= nil then conv else nil,
        keyScales = if scaleRow > 0 then keyScales else nil,
        valueScales = if scaleRow > 0 then valueScales else nil,
        keyCentres = if scaleRow > 0 then keyCentres else nil,
    }
    const from = s.cache
    const fromConv = from.conv
    for l = 1, s.model.config.layers do
        if fromConv ~= nil and config.isConv(s.model.config, l) then
            const history = fromConv[l]
            const hidden = s.model.config.hidden
            out.keys[l] = buffer.create(0)
            out.values[l] = buffer.create(0)
            conv[l] = buffer.create(length * hidden * 4)
            buffer.copy(conv[l], 0, history, 0, length * hidden * 4)
            if scaleRow > 0 then
                keyScales[l] = buffer.create(0)
                valueScales[l] = buffer.create(0)
                keyCentres[l] = buffer.create(0)
            end
            continue
        end
        conv[l] = buffer.create(0)
        out.keys[l] = buffer.create(heads * length * row)
        out.values[l] = buffer.create(heads * length * row)
        copyHeads(out.keys[l], length, from.keys[l], s.context, heads, length, row)
        copyHeads(out.values[l], length, from.values[l], s.context, heads, length, row)
        const sourceKeyScales, sourceValueScales, sourceCentres = from.keyScales, from.valueScales, from.keyCentres
        if sourceKeyScales ~= nil and sourceValueScales ~= nil and sourceCentres ~= nil then
            const ks, vs = buffer.create(heads * length * 4), buffer.create(heads * length * 4)
            copyHeads(ks, length, sourceKeyScales[l], s.context, heads, length, 4)
            copyHeads(vs, length, sourceValueScales[l], s.context, heads, length, 4)
            keyScales[l] = ks
            valueScales[l] = vs
            const centre = sourceCentres[l]
            keyCentres[l] = buffer.create(buffer.len(centre))
            buffer.copy(keyCentres[l], 0, centre)
        end
    end
    return out
end

--- Makes `s` hold the positions of `snap`, which a session of the same model and context took.
const function restore(s: Session, snap: Snapshot)
    assert(snap.length <= s.context, "the snapshot fits the context")
    const heads = s.model.dims.kvHeads
    const row = rowBytes(s)
    const into = s.cache
    for l = 1, s.model.config.layers do
        const history, saved = snap.conv, into.conv
        if history ~= nil and saved ~= nil and config.isConv(s.model.config, l) then
            buffer.copy(saved[l], 0, history[l], 0, snap.length * s.model.config.hidden * 4)
            continue
        end
        copyHeads(into.keys[l], s.context, snap.keys[l], snap.length, heads, snap.length, row)
        copyHeads(into.values[l], s.context, snap.values[l], snap.length, heads, snap.length, row)
        const keyScales, valueScales, centres = into.keyScales, into.valueScales, into.keyCentres
        const savedKeyScales, savedValueScales, savedCentres = snap.keyScales, snap.valueScales, snap.keyCentres
        if
            keyScales ~= nil
            and valueScales ~= nil
            and centres ~= nil
            and savedKeyScales ~= nil
            and savedValueScales ~= nil
            and savedCentres ~= nil
        then
            copyHeads(keyScales[l], s.context, savedKeyScales[l], snap.length, heads, snap.length, 4)
            copyHeads(valueScales[l], s.context, savedValueScales[l], snap.length, heads, snap.length, 4)
            buffer.copy(centres[l], 0, savedCentres[l])
        end
    end
    s.length = snap.length
end

--- Bytes the snapshot holds.
const function snapshotBytes(snap: Snapshot): number
    local total = 0
    for l = 1, #snap.keys do
        total += buffer.len(snap.keys[l]) + buffer.len(snap.values[l])
        const keyScales, valueScales, centres = snap.keyScales, snap.valueScales, snap.keyCentres
        if keyScales ~= nil and valueScales ~= nil and centres ~= nil then
            total += buffer.len(keyScales[l]) + buffer.len(valueScales[l]) + buffer.len(centres[l])
        end
        const history = snap.conv
        if history ~= nil then
            total += buffer.len(history[l])
        end
    end
    return total
end

return {
    new = new,
    reset = reset,
    snapshot = snapshot,
    restore = restore,
    snapshotBytes = snapshotBytes,
    layerCacheBytes = layerCacheBytes,
    cacheBytes = cacheBytes,
    workspaceBytes = workspaceBytes,
}
