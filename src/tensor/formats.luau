--!native
-- Weight formats and the address arithmetic of their layouts. Pure Luau, no host.

const types = require("./types")

type FormatName = types.FormatName
type LayoutName = types.LayoutName

--- What a format stores per weight.
export type Format = {
    name: FormatName,
    --- Layout the shipped kernels of this format read.
    layout: LayoutName,
    --- Weights per scale, 0 when the format has no scales.
    group: number,
    --- Bits of one stored weight.
    bits: number,
}

--- The largest buffer Luau creates, in bytes, on every host.
const MAX_BUFFER_BYTES = 1073741824

--- Weights per scale of the quantized formats.
const GROUP = 32

--- Bytes of one f32 scale.
const SCALE_BYTES = 4

--- The `v12` layout: a tile of 12 rows is 4 row vectors of 3 rows, a group of 32 columns is 11 steps of one lane per
--- row, a lane holding the weights of columns j, j + 11 and j + 22.
const V12_TILE_ROWS = 12
const V12_STEPS = 11
const V12_GROUP_BYTES = 44

const FORMATS: { [FormatName]: Format } = {
    f32 = { name = "f32", layout = "rowmajor", group = 0, bits = 32 },
    bf16 = { name = "bf16", layout = "rowmajor", group = 0, bits = 16 },
    q8 = { name = "q8", layout = "p8", group = GROUP, bits = 8 },
    q4 = { name = "q4", layout = "p4", group = GROUP, bits = 4 },
}

const scratch = buffer.create(4)

--- The format record for `name`.
const function get(name: FormatName): Format
    return FORMATS[name]
end

--- Bytes of one weight row of `kpad` stored weights, without scales. The `v12` layout stores 11 f32 lanes of 3 weights
--- per group of 32 and 3 rows, 44 bytes per row and group.
const function rowBytes(name: FormatName, kpad: number, layout: LayoutName?): number
    if layout == "v12" then
        return kpad // GROUP * V12_GROUP_BYTES
    end
    return kpad * FORMATS[name].bits // 8
end

--- Bytes of the scales of one row of `kpad` stored weights, 0 for formats without scales.
const function scaleRowBytes(name: FormatName, kpad: number): number
    const group = FORMATS[name].group
    if group == 0 then
        return 0
    end
    return kpad // group * SCALE_BYTES
end

--- Byte offset of the scale of `group` in `row`, counted from the start of the part's scale region. The `p8` and `p4`
--- layouts keep the 8 scales of a tile's rows together for each group.
const function scaleOffset(row: number, group: number, kpad: number, layout: LayoutName?): number
    const groups = kpad // GROUP
    if layout == "p8" or layout == "p4" then
        return ((row // 8) * 8 * groups + group * 8 + row % 8) * SCALE_BYTES
    elseif layout == "v12" then
        return ((row // 12) * 12 * groups + group * 12 + row % 12) * SCALE_BYTES
    end
    return (row * groups + group) * SCALE_BYTES
end

--- Byte offset of weight (`row`, `k`) in a part, row counted inside the part, and for Q4 the nibble that holds it
--- (0 low, 1 high). Rows 0 to 3 of a part start a 4 row group, which the part cut on a tile boundary guarantees, and
--- for `p8` rows 0 to 7 start a tile. `layout` is the format's own layout when nil.
const function weightOffset(
    name: FormatName,
    row: number,
    k: number,
    kpad: number,
    layout: LayoutName?
): (number, number)
    if name == "f32" then
        return (row * kpad + k) * 4, 0
    elseif name == "bf16" then
        return (row * kpad + k) * 2, 0
    elseif name == "q8" then
        const q8Layout = layout or FORMATS.q8.layout
        if q8Layout == "p8" then
            return (row // 8) * 8 * kpad + k * 8 + row % 8, 0
        elseif q8Layout == "v12" then
            -- The byte of the f32 lane and which of its three weights (0 low, 1 middle, 2 high) is the one.
            const group = k // GROUP
            const j = k % GROUP
            const step = j % V12_STEPS
            const tile = (row // 12) * (kpad // GROUP) + group
            return tile * V12_GROUP_BYTES * 12 + step * 48 + (row % 12) // 3 * 12 + row % 3 * 4, j // V12_STEPS
        end
        return (row // 4) * 4 * kpad + k * 4 + row % 4, 0
    else
        const group = k // GROUP
        const j = k % GROUP
        if (layout or FORMATS.q4.layout) == "p4" then
            -- Rows 2m and 2m + 1 of a tile share pair m: the even row is nibble m of the column's word, the odd one m + 4.
            const nibble = (row % 2) * 4 + row % 8 // 2
            return (row // 8) * 4 * kpad + group * 128 + j * 4 + nibble // 2, nibble % 2
        end
        return row * (kpad // 2) + group * 16 + j % 16, j // 16
    end
end

--- The f32 lane that holds the weights `a`, `b`, `c` (offset binary bytes, low to high) of the `v12` layout.
const function laneOf(a: number, b: number, c: number): number
    return (a + b * 256 + c * 65536) / 65536
end

--- The three bytes of a `v12` lane.
const function bytesOf(lane: number): (number, number, number)
    const c = math.floor(lane)
    const m = lane * 256
    const f = math.floor(m)
    return (m - f) * 256, f - c * 256, c
end

--- The BF16 bits of an f32 value, rounded to nearest even. NaN and infinity are not special cased.
const function bf16FromF32(value: number): number
    buffer.writef32(scratch, 0, value)
    const bits = buffer.readu32(scratch, 0)
    const round = bit32.band(bit32.rshift(bits, 16), 1) + 0x7FFF
    return bit32.rshift((bits + round) % 4294967296, 16)
end

--- The f32 value of BF16 bits, exact.
const function bf16ToF32(bits: number): number
    buffer.writeu32(scratch, 0, bit32.lshift(bits, 16))
    return buffer.readf32(scratch, 0)
end

--- The f32 value nearest to `value`, which is what a stored weight or output holds.
const function roundF32(value: number): number
    buffer.writef32(scratch, 0, value)
    return buffer.readf32(scratch, 0)
end

--- The half precision value nearest to `value` (ties to even), as a number. Q8_0 stores its scales in this precision.
const function roundF16(value: number): number
    if value == 0 then
        return 0
    end
    const _, exponent = math.frexp(math.abs(value))
    -- 11 significant bits, and below 2^-14 the spacing stays 2^-24.
    const quantum = 2 ^ math.max(exponent - 11, -24)
    const scaled = math.abs(value) / quantum
    local whole = math.floor(scaled)
    const rest = scaled - whole
    if rest > 0.5 or (rest == 0.5 and whole % 2 == 1) then
        whole += 1
    end
    return math.sign(value) * whole * quantum
end

return {
    MAX_BUFFER_BYTES = MAX_BUFFER_BYTES,
    GROUP = GROUP,
    SCALE_BYTES = SCALE_BYTES,
    V12_TILE_ROWS = V12_TILE_ROWS,
    V12_STEPS = V12_STEPS,
    V12_GROUP_BYTES = V12_GROUP_BYTES,
    get = get,
    rowBytes = rowBytes,
    scaleRowBytes = scaleRowBytes,
    scaleOffset = scaleOffset,
    weightOffset = weightOffset,
    laneOf = laneOf,
    bytesOf = bytesOf,
    bf16FromF32 = bf16FromF32,
    bf16ToF32 = bf16ToF32,
    roundF32 = roundF32,
    roundF16 = roundF16,
}
