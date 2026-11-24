--!native
-- A loaded model: its weights, the kernels chosen for this host and the position table.

const arch = require("@src/arch")
const attention = require("@src/ops/attention")
const attentionUnrolled = require("@src/ops/generated")
const attentionQ8 = require("@src/ops/attention_q8")
const attentionVector = require("@src/ops/attention_vector")
const config = require("@self/config")
const container = require("@src/container")
const registry = require("@src/kernels/registry")
const rope = require("@src/ops/rope")
const split = require("@src/parallel/split")
const containerTypes = require("@src/container/types")
const kernelsTypes = require("@src/kernels/types")
const parallelTypes = require("@src/parallel/types")
const tensorTypes = require("@src/tensor/types")
const types = require("@self/types")

export type Cache = types.Cache
export type Dims = types.Dims
export type Estimate = types.Estimate
export type Hook = types.Hook
export type Layer = types.Layer
export type Model = types.Model
export type Runner = types.Runner
export type Sequence = types.Sequence
export type Session = types.Session
export type ShortConv = types.ShortConv
export type Snapshot = types.Snapshot
export type Workspace = types.Workspace

--- How a KV cache of format `kv` stores its rows for a model with `headDim` values a head: "f32" for the f32 cache, and for
--- the Q8 cache "v3" when the vector read is there and a generated kernel has this head size, else "i8".
const function kvLayoutFor(headDim: number, caps: kernelsTypes.Caps, kv: string?): string
    if kv ~= "q8" then
        return "f32"
    end
    return if caps.vectorRead and attentionUnrolled.q8ByLanes[config.headPad(headDim) // 3] ~= nil then "v3" else "i8"
end

--- Loads every weight of the container into buffers and picks the kernels `caps` allows. `context` is the number of
--- positions a conversation may use. With a `share` it loads only the weights of that share of the decoder: its heads of
--- q, k and v, the matching columns of o, its columns of gate and up and of down, its rows of the output matrix, and no
--- embedding table.
const function load(
    c: containerTypes.Container,
    caps: kernelsTypes.Caps,
    context: number,
    share: parallelTypes.Share?,
    kv: string?,
    layout: string?
): Model
    const cfg: config.Config = c.header.config
    const kvLayout = layout or kvLayoutFor(cfg.headDim, caps, kv)
    if layout ~= nil and (kv ~= "q8" or (layout ~= "i8" and layout ~= "v3")) then
        error(`the KV layout {layout} does not exist for a {kv or "f32"} cache`, 0)
    end
    if
        kvLayout == "v3"
        and (not caps.vectorRead or attentionUnrolled.q8ByLanes[config.headPad(cfg.headDim) // 3] == nil)
    then
        error("the vector layout of the Q8 cache needs the vector read and a generated kernel for the head size", 0)
    end
    const plugin = arch.get(c.header.arch)
    const mine = share or split.whole(cfg)
    const sharded = share ~= nil
    if sharded and cfg.layerTypes ~= nil then
        error("a model with short convolution layers cannot be divided among workers", 0)
    end
    const hd = cfg.headDim
    const layers: { Layer } = {}
    local prepK = 0
    --- The weight of a norm that only scales to unit mean square, shared by every layer that has one.
    const unitNorm = buffer.create(hd * 4)
    for j = 0, hd - 1 do
        buffer.writef32(unitNorm, j * 4, 1)
    end
    --- Loads a matrix, or the rows and columns of it that a share holds, and notes the row length a prepared activation
    --- needs for it.
    const function loadMatrix(
        name: string,
        rowStart: number?,
        rowCount: number?,
        colStart: number?,
        colCount: number?
    ): tensorTypes.Tensor
        const info = c.tensors[name]
        const t = if sharded
            then container.loadSlice(
                c,
                name,
                rowStart or 0,
                rowCount or info.rows,
                colStart or 0,
                colCount or info.cols
            )
            else container.loadTensor(c, name)
        if t.format == "q8" or t.format == "q4" then
            prepK = math.max(prepK, t.kpad)
        end
        return t
    end
    for i = 0, cfg.layers - 1 do
        const function vector(name: string, start: number, count: number): buffer
            return if sharded
                then container.loadVectorSlice(c, `layers.{i}.{name}`, start, count)
                else container.loadVector(c, `layers.{i}.{name}`)
        end
        const qStart, qCount = mine.headStart * hd, mine.headCount * hd
        const kvStart, kvCount = mine.kvStart * hd, mine.kvCount * hd
        const isConv = config.isConv(cfg, i + 1)
        table.insert(layers, {
            attnNorm = container.loadVector(c, `layers.{i}.attn_norm`),
            q = if isConv then nil else loadMatrix(`layers.{i}.q`, qStart, qCount),
            k = if isConv then nil else loadMatrix(`layers.{i}.k`, kvStart, kvCount),
            v = if isConv then nil else loadMatrix(`layers.{i}.v`, kvStart, kvCount),
            o = if isConv then nil else loadMatrix(`layers.{i}.o`, nil, nil, qStart, qCount),
            conv = if isConv
                then {
                    inProj = loadMatrix(`layers.{i}.conv_in`),
                    taps = container.loadVector(c, `layers.{i}.conv_taps`),
                    outProj = loadMatrix(`layers.{i}.conv_out`),
                }
                else nil,
            mlpNorm = container.loadVector(c, `layers.{i}.mlp_norm`),
            postAttnNorm = if plugin.postNorms then container.loadVector(c, `layers.{i}.post_attn_norm`) else nil,
            postMlpNorm = if plugin.postNorms then container.loadVector(c, `layers.{i}.post_mlp_norm`) else nil,
            qBias = if plugin.qkvBias then vector("q_bias", qStart, qCount) else nil,
            kBias = if plugin.qkvBias then vector("k_bias", kvStart, kvCount) else nil,
            vBias = if plugin.qkvBias then vector("v_bias", kvStart, kvCount) else nil,
            qNorm = if not plugin.qkNorm or isConv
                then nil
                elseif plugin.qkNormUnit then unitNorm
                else container.loadVector(c, `layers.{i}.q_norm`),
            kNorm = if not plugin.qkNorm or isConv
                then nil
                elseif plugin.qkNormUnit then unitNorm
                else container.loadVector(c, `layers.{i}.k_norm`),
            gate = loadMatrix(`layers.{i}.gate`, mine.interStart, mine.interCount),
            up = loadMatrix(`layers.{i}.up`, mine.interStart, mine.interCount),
            down = loadMatrix(`layers.{i}.down`, nil, nil, mine.interStart, mine.interCount),
        })
    end
    local embed: tensorTypes.Tensor? = nil
    local head: tensorTypes.Tensor
    if sharded then
        head = loadMatrix(if cfg.tieEmbeddings then "embed" else "head", mine.vocabStart, mine.vocabCount)
    else
        const embedding = loadMatrix("embed")
        embed = embedding
        head = if cfg.tieEmbeddings then embedding else loadMatrix("head")
    end
    const format, layout = layers[1].gate.format, layers[1].gate.layout
    for _, layer in layers do
        const matrices = { layer.gate, layer.up, layer.down }
        for _, t in { layer.q, layer.k, layer.v, layer.o } do
            table.insert(matrices, t)
        end
        const conv = layer.conv
        if conv ~= nil then
            table.insert(matrices, conv.inProj)
            table.insert(matrices, conv.outProj)
        end
        for _, t in matrices do
            if t.format ~= format or t.layout ~= layout then
                error("the layer matrices of one model share a format and a layout", 0)
            end
        end
    end
    return {
        config = cfg,
        arch = plugin,
        dims = {
            heads = mine.headCount,
            kvHeads = mine.kvCount,
            headBase = mine.headStart,
            kvBase = mine.kvStart,
            group = cfg.heads // cfg.kvHeads,
            inter = mine.interCount,
        },
        embed = embed,
        head = head,
        finalNorm = container.loadVector(c, "final_norm"),
        layers = layers,
        decode = registry.select(format, caps, false, layout),
        prefill = registry.select(format, caps, true, layout),
        headKernel = registry.select(head.format, caps, false, head.layout),
        prepK = prepK,
        attend = if kvLayout == "v3"
            then attentionUnrolled.q8ByLanes[config.headPad(cfg.headDim) // 3]
            elseif kv == "q8" then attentionQ8.attend
            elseif caps.vectorRead then attentionUnrolled.byLanes[config.headPad(cfg.headDim) // 3]
                or attentionVector.attend
            else attention.attend,
        rope = rope.new(cfg.headDim, cfg.ropeTheta, context),
        ropeLocal = if cfg.ropeLocalTheta ~= nil
            then rope.new(cfg.headDim, cfg.ropeLocalTheta :: number, context)
            else nil,
        context = context,
        kv = kv or "f32",
        kvLayout = kvLayout,
        hdPad = config.headPad(cfg.headDim),
    }
end

return {
    load = load,
    kvLayoutFor = kvLayoutFor,
}
