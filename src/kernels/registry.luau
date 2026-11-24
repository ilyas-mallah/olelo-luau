--!native
-- Lists every generated kernel with the format, layout, tile and capability it needs, pairs it with its reference, and
-- selects the kernel a host should run.

const bf16Scalar = require("@src/kernels/generated/bf16_scalar")
const bf16Vector = require("@src/kernels/generated/bf16_vector")
const f32Buffer = require("@src/kernels/generated/f32_buffer")
const f32Scalar = require("@src/kernels/generated/f32_scalar")
const f32Vector = require("@src/kernels/generated/f32_vector")
const float = require("@src/kernels/reference/float")
const formats = require("@src/tensor/formats")
const q4Byte = require("@src/kernels/generated/q4_byte")
const q4P4 = require("@src/kernels/generated/q4_p4")
const q8Il4e = require("@src/kernels/generated/q8_il4e")
const q8P8 = require("@src/kernels/generated/q8_p8")
const q8V12 = require("@src/kernels/generated/q8_v12")
const quant = require("@src/kernels/reference/quant")
const tensorTypes = require("@src/tensor/types")
const types = require("./types")

type Activation = types.Activation
type Caps = types.Caps
type Entry = types.Entry
type Kind = types.Kind

const MODULES: { { name: string, module: types.Module, reference: types.Kernel } } = {
    { name = "f32_vector", module = f32Vector, reference = float.f32 },
    { name = "f32_buffer", module = f32Buffer, reference = float.f32 },
    { name = "f32_scalar", module = f32Scalar, reference = float.f32 },
    { name = "bf16_vector", module = bf16Vector, reference = float.bf16 },
    { name = "bf16_scalar", module = bf16Scalar, reference = float.bf16 },
    { name = "q8_il4e", module = q8Il4e, reference = quant.q8 },
    { name = "q8_p8", module = q8P8, reference = quant.q8p },
    { name = "q8_v12", module = q8V12, reference = quant.q8v },
    { name = "q4_byte", module = q4Byte, reference = quant.q4 },
    { name = "q4_p4", module = q4P4, reference = quant.q4p },
}

--- Kernel ids in order of preference for decode (one token at a time), best first. The first one a host can run wins.
const DECODE: { [tensorTypes.FormatName]: { string } } = {
    f32 = { "f32_vector/R8S1U8", "f32_buffer/R8S1U8", "f32_scalar/R8S1U8" },
    bf16 = { "bf16_vector/R4S1U8", "bf16_scalar/R8S1U8" },
    q8 = { "q8_p8/R8S1U32" },
    q4 = { "q4_p4/R8S1U32" },
}

--- Kernel ids in order of preference for prefill (several tokens at once). Formats without a multi token tile fall
--- back to their decode list.
const PREFILL: { [tensorTypes.FormatName]: { string } } = {
    f32 = { "f32_vector/R4S2U2" },
    q8 = { "q8_p8/R4S2U32" },
    q4 = { "q4_p4/R4S2U32" },
}

--- The same lists for the Q8 matrices stored in the `v12` layout, which only the vector kernels read.
const DECODE_V12: { string } = { "q8_v12/R12S1U32" }
const PREFILL_V12: { string } = { "q8_v12/R6S2U32" }

const entries: { Entry } = {}
const byId: { [string]: Entry } = {}

for _, m in MODULES do
    for _, t in m.module.tiles do
        const kind: Kind = if t.src == "rv" then "vector" elseif t.src == "buf" then "buffer" else "scalar"
        const entry: Entry = {
            id = `{m.name}/{t.name}`,
            module = m.name,
            name = t.name,
            format = m.module.format :: tensorTypes.FormatName,
            layout = m.module.layout :: tensorTypes.LayoutName,
            activation = m.module.activation :: Activation,
            kind = kind,
            R = t.R,
            S = t.S,
            U = t.U,
            rowTile = if m.module.layout == "p8" or m.module.layout == "p4"
                then 8
                elseif m.module.layout == "v12" then formats.V12_TILE_ROWS
                else t.R,
            kstep = t.kstep,
            needsVectorRead = t.src == "rv",
            fn = t.fn,
            reference = m.reference,
        }
        table.insert(entries, entry)
        byId[entry.id] = entry
    end
end

--- Every generated kernel.
const function all(): { Entry }
    return entries
end

--- The kernel with this id, which must exist.
const function find(id: string): Entry
    const entry = byId[id]
    assert(entry ~= nil, `no kernel {id}`)
    return entry
end

--- The best kernel of `format` the host can run. `prefill` asks for a multi token tile where one exists. `layout` is
--- the layout the matrix is stored in, the format's own when nil.
const function select(
    format: tensorTypes.FormatName,
    caps: Caps,
    prefill: boolean?,
    layout: tensorTypes.LayoutName?
): Entry
    const lists: { { string } } = {}
    if layout == "v12" then
        assert(format == "q8", "only Q8 has the v12 layout")
        if not caps.vectorRead then
            error(
                "this model stores its Q8 matrices in the v12 layout, which needs the fast buffer.readvectorf32x3, convert it with the default layout to run it here",
                0
            )
        end
        if prefill then
            table.insert(lists, PREFILL_V12)
        end
        table.insert(lists, DECODE_V12)
    else
        if prefill and PREFILL[format] then
            table.insert(lists, PREFILL[format])
        end
        table.insert(lists, DECODE[format])
    end
    for _, list in lists do
        for _, id in list do
            const entry = byId[id]
            assert(entry ~= nil, `unknown kernel {id} in the preference list`)
            if caps.vectorRead or not entry.needsVectorRead then
                return entry
            end
        end
    end
    error(`no kernel of format {format} runs on this host`)
end

return {
    all = all,
    find = find,
    select = select,
}
