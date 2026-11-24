--!native
-- A matrix of weights in one format and layout, stored in parts of at most one buffer limit each.

const formats = require("@self/formats")
const store = require("@self/store")
const types = require("@self/types")

export type FormatName = types.FormatName
export type LayoutName = types.LayoutName
export type Part = types.Part
export type Spec = types.Spec
export type Tensor = types.Tensor

--- Rounds `n` up to a multiple of `step`.
const function roundUp(n: number, step: number): number
    return (n + step - 1) // step * step
end

--- Fills a `v12` buffer with lanes of the weight 0, the offset binary byte 128 in each of the three fields.
const function fillZeroLanes(b: buffer)
    const zero = formats.laneOf(128, 128, 128)
    for at = 0, buffer.len(b) - 4, 4 do
        buffer.writef32(b, at, zero)
    end
end

--- Builds a zero filled tensor, split into parts on tile row boundaries.
const function new(spec: Spec): Tensor
    const layout = spec.layout or formats.get(spec.format).layout
    if spec.format == "q8" or spec.format == "q4" then
        assert(spec.kstep % formats.GROUP == 0, "quantized kernels step over whole groups")
    end
    if layout == "il4e" then
        assert(spec.rowTile % 4 == 0, "the interleaved layout needs a multiple of 4 rows")
    elseif layout == "p8" or layout == "p4" then
        assert(spec.rowTile % 8 == 0, "the packed layouts need a multiple of 8 rows")
    elseif layout == "v12" then
        assert(spec.rowTile % formats.V12_TILE_ROWS == 0, "the vector layout needs a multiple of 12 rows")
    end
    const kpad = roundUp(spec.cols, spec.kstep)
    const rowsPadded = roundUp(spec.rows, spec.rowTile)
    const plans = store.plan(
        rowsPadded,
        formats.rowBytes(spec.format, kpad, layout),
        spec.rowTile,
        spec.maxPartBytes or formats.MAX_BUFFER_BYTES
    )
    const parts = store.create(plans, spec.format, kpad, layout)
    if layout == "v12" then
        for _, part in parts do
            fillZeroLanes(part.buf)
        end
    elseif spec.format == "q4" or layout == "p8" then
        -- A stored nibble of 8 and an offset binary byte of 128 are the weight 0, so untouched padding stays zero.
        for _, part in parts do
            buffer.fill(part.buf, 0, if spec.format == "q4" then 0x88 else 0x80)
        end
    end
    return {
        format = spec.format,
        layout = layout,
        rows = spec.rows,
        rowsPadded = rowsPadded,
        cols = spec.cols,
        kpad = kpad,
        rowTile = spec.rowTile,
        parts = parts,
    }
end

--- The part that holds `row` and the row's number inside it.
const function locate(t: Tensor, row: number): (types.Part, number)
    const part = t.parts[store.partIndex(t.parts, row)]
    return part, row - part.rowStart
end

--- Writes an f32 or BF16 weight, rounding as the format does.
const function setFloat(t: Tensor, row: number, k: number, value: number)
    const part, localRow = locate(t, row)
    const offset = formats.weightOffset(t.format, localRow, k, t.kpad, t.layout)
    if t.format == "f32" then
        buffer.writef32(part.buf, part.offset + offset, value)
    else
        buffer.writeu16(part.buf, part.offset + offset, formats.bf16FromF32(value))
    end
end

--- The f32 or BF16 weight at (`row`, `k`) as a number.
const function getFloat(t: Tensor, row: number, k: number): number
    const part, localRow = locate(t, row)
    const offset = formats.weightOffset(t.format, localRow, k, t.kpad, t.layout)
    if t.format == "f32" then
        return buffer.readf32(part.buf, part.offset + offset)
    end
    return formats.bf16ToF32(buffer.readu16(part.buf, part.offset + offset))
end

--- Writes a quantized weight: -128 to 127 for Q8 whatever the layout stores, 0 to 15 (the stored nibble, value minus 8)
--- for Q4.
const function setQuant(t: Tensor, row: number, k: number, q: number)
    const part, localRow = locate(t, row)
    const offset, nibble = formats.weightOffset(t.format, localRow, k, t.kpad, t.layout)
    if t.format == "q8" then
        if t.layout == "v12" then
            const at = part.offset + offset
            const bytes = { formats.bytesOf(buffer.readf32(part.buf, at)) }
            bytes[nibble + 1] = q + 128
            buffer.writef32(part.buf, at, formats.laneOf(bytes[1], bytes[2], bytes[3]))
        elseif t.layout == "p8" then
            buffer.writeu8(part.buf, part.offset + offset, q + 128)
        else
            buffer.writei8(part.buf, part.offset + offset, q)
        end
        return
    end
    const at = part.offset + offset
    local byte = buffer.readu8(part.buf, at)
    if nibble == 0 then
        byte = byte - byte % 16 + q
    else
        byte = byte % 16 + q * 16
    end
    buffer.writeu8(part.buf, at, byte)
end

--- The quantized weight at (`row`, `k`) as setQuant takes it.
const function getQuant(t: Tensor, row: number, k: number): number
    const part, localRow = locate(t, row)
    const offset, nibble = formats.weightOffset(t.format, localRow, k, t.kpad, t.layout)
    if t.format == "q8" then
        if t.layout == "v12" then
            const bytes = { formats.bytesOf(buffer.readf32(part.buf, part.offset + offset)) }
            return bytes[nibble + 1] - 128
        elseif t.layout == "p8" then
            return buffer.readu8(part.buf, part.offset + offset) - 128
        end
        return buffer.readi8(part.buf, part.offset + offset)
    end
    local byte = buffer.readu8(part.buf, part.offset + offset)
    if nibble == 0 then
        return byte % 16
    end
    return byte // 16
end

--- Writes the f32 scale of `group` (32 columns) in `row` of a quantized tensor.
const function setScale(t: Tensor, row: number, group: number, scale: number)
    const part, localRow = locate(t, row)
    const scales = part.scales :: buffer
    buffer.writef32(scales, part.scaleOffset + formats.scaleOffset(localRow, group, t.kpad, t.layout), scale)
end

--- The f32 scale of `group` in `row` of a quantized tensor.
const function getScale(t: Tensor, row: number, group: number): number
    const part, localRow = locate(t, row)
    const scales = part.scales :: buffer
    return buffer.readf32(scales, part.scaleOffset + formats.scaleOffset(localRow, group, t.kpad, t.layout))
end

return {
    new = new,
    locate = locate,
    setFloat = setFloat,
    getFloat = getFloat,
    setQuant = setQuant,
    getQuant = getQuant,
    setScale = setScale,
    getScale = getScale,
}
