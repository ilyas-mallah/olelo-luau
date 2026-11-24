--!native
-- Reads the `.olelo` container: the header, the tensor table and the blobs. Format in docs/container.md.

const formats = require("@src/tensor/formats")
const host = require("@src/host")
const json = require("@src/util/json")
const tensor = require("@src/tensor")
const tensorTypes = require("@src/tensor/types")
const types = require("@self/types")

export type Container = types.Container
export type PartInfo = types.PartInfo
export type TensorInfo = types.TensorInfo

const MAGIC = "OLELO\0\0\0"
const VERSION = 1
const ALIGN = 64

--- Opens a container and decodes its header. Weight bytes are read later by `loadTensor` and `blob`.
const function open(source: host.WeightSource): Container
    const prefix = buffer.create(16)
    source.read(0, 16, prefix, 0)
    if buffer.readstring(prefix, 0, 8) ~= MAGIC then
        error("not an Olelo container: wrong magic", 0)
    end
    const version = buffer.readu32(prefix, 8)
    if version ~= VERSION then
        error(
            `the container has version {version}, this runtime reads version {VERSION}, convert again with the matching converter`,
            0
        )
    end
    const headerBytes = buffer.readu32(prefix, 12)
    const text = buffer.create(headerBytes)
    source.read(16, headerBytes, text, 0)
    const header = json.decode(buffer.readstring(text, 0, headerBytes))
    const tensors: { [string]: TensorInfo } = {}
    for _, info in header.tensors do
        tensors[info.name] = info
    end
    return {
        header = header,
        source = source,
        tensors = tensors,
        dataStart = (16 + headerBytes + ALIGN - 1) // ALIGN * ALIGN,
        blobs = header.blobs,
    }
end

--- The bytes of a blob in a new buffer.
const function blob(c: Container, name: string): buffer
    const info = c.blobs[name]
    if info == nil then
        error(`the container has no blob {name}`, 0)
    end
    const out = buffer.create(info.bytes)
    c.source.read(c.dataStart + info.offset, info.bytes, out, 0)
    return out
end

--- The table entry of a matrix this runtime reads, with its format and layout.
const function matrixInfo(c: Container, name: string): (TensorInfo, tensorTypes.FormatName, tensorTypes.LayoutName)
    const info = c.tensors[name]
    if info == nil then
        error(`the container has no tensor {name}`, 0)
    end
    if info.format ~= "f32" and info.format ~= "bf16" and info.format ~= "q8" and info.format ~= "q4" then
        error(`tensor {name} has format {info.format}, which this runtime build does not read`, 0)
    end
    const format = info.format :: tensorTypes.FormatName
    -- Q8 has two layouts a kernel reads, the packed one and the vector one.
    const layout = if format == "q8" and info.layout == "v12"
        then "v12" :: tensorTypes.LayoutName
        else formats.get(format).layout
    if info.layout ~= layout then
        error(`tensor {name} has layout {info.layout}, this runtime reads {layout} for {format}`, 0)
    end
    return info, format, layout
end

--- Loads a matrix into parts of buffers.
const function loadTensor(c: Container, name: string): tensorTypes.Tensor
    const info, format, layout = matrixInfo(c, name)
    const parts: { tensorTypes.Part } = {}
    for _, p in info.parts do
        const buf = buffer.create(p.bytes)
        c.source.read(c.dataStart + p.offset, p.bytes, buf, 0)
        local scales: buffer? = nil
        if format == "q8" or format == "q4" then
            const scaleBytes = p.scaleBytes
            const scaleOffset = p.scaleOffset
            if scaleBytes == nil or scaleOffset == nil then
                error(`tensor {name} is {format} and a part has no scales`, 0)
            end
            const partScales = buffer.create(scaleBytes)
            scales = partScales
            c.source.read(c.dataStart + scaleOffset, scaleBytes, partScales, 0)
        end
        table.insert(parts, {
            buf = buf,
            offset = 0,
            scales = scales,
            scaleOffset = 0,
            rowStart = p.rowStart,
            rowCount = p.rows,
        })
    end
    return {
        format = format,
        layout = layout,
        rows = info.rows,
        rowsPadded = (info.rows + info.rowTile - 1) // info.rowTile * info.rowTile,
        cols = info.cols,
        kpad = info.kpad,
        rowTile = info.rowTile,
        parts = parts,
    }
end

--- Loads a vector tensor (a norm weight or a bias) as an f32 buffer of `cols` floats.
const function loadVector(c: Container, name: string): buffer
    const info = c.tensors[name]
    if info == nil then
        error(`the container has no tensor {name}`, 0)
    end
    const part = info.parts[1]
    const out = buffer.create(part.bytes)
    c.source.read(c.dataStart + part.offset, part.bytes, out, 0)
    return out
end

--- Loads `count` values of a vector tensor from `start` as an f32 buffer.
const function loadVectorSlice(c: Container, name: string, start: number, count: number): buffer
    const info = c.tensors[name]
    if info == nil then
        error(`the container has no tensor {name}`, 0)
    end
    const part = info.parts[1]
    if start < 0 or (start + count) * 4 > part.bytes then
        error(`the slice {start}, {count} is outside the vector {name}`, 0)
    end
    const out = buffer.create(count * 4)
    c.source.read(c.dataStart + part.offset + start * 4, count * 4, out, 0)
    return out
end

--- The part of a matrix's table entry that holds `row`.
const function sourcePart(info: TensorInfo, row: number): PartInfo
    for _, p in info.parts do
        if row >= p.rowStart and row < p.rowStart + p.rows then
            return p
        end
    end
    error("row outside the tensor", 0)
end

--- Loads rows `rowStart` to `rowStart + rowCount - 1` and columns `colStart` to `colStart + colCount - 1` of a matrix as
--- a matrix of its own, padded as `tensor.new` pads. Only those bytes are read. The weights are the same bytes as the
--- whole matrix holds, so a product over the slice equals the matching part of the product over the whole. A quantized
--- matrix starts its columns on a group of 32, the packed Q8 layout starts its rows on a tile of 8 and takes a row count
--- that is a whole number of tiles unless the slice reaches the last row.
const function loadSlice(
    c: Container,
    name: string,
    rowStart: number,
    rowCount: number,
    colStart: number,
    colCount: number
): tensorTypes.Tensor
    const info, format, layout = matrixInfo(c, name)
    const quant = format == "q8" or format == "q4"
    if rowStart < 0 or rowCount < 1 or rowStart + rowCount > info.rows then
        error(`the rows {rowStart}, {rowCount} are outside {name}, which has {info.rows}`, 0)
    end
    if colStart < 0 or colCount < 1 or colStart + colCount > info.cols then
        error(`the columns {colStart}, {colCount} are outside {name}, which has {info.cols}`, 0)
    end
    if quant and colStart % formats.GROUP ~= 0 then
        error(`the columns of a quantized slice start on a group of {formats.GROUP}, not at {colStart}`, 0)
    end
    const vector = layout == "v12"
    const packed = layout == "p8" or layout == "p4" or vector
    const tileRows = if vector then formats.V12_TILE_ROWS else 8
    const aligned = rowStart % tileRows == 0 and (rowCount % tileRows == 0 or rowStart + rowCount == info.rows)
    if packed and not vector and not aligned then
        error("the rows of a packed slice are whole tiles of 8", 0)
    end
    const dst = tensor.new({
        format = format,
        rows = rowCount,
        cols = colCount,
        rowTile = info.rowTile,
        kstep = if quant then formats.GROUP else 24,
        layout = layout,
    })
    const base = c.dataStart
    const read = c.source.read
    const srcKpad, dstKpad = info.kpad, dst.kpad
    const srcGroups, dstGroups = srcKpad // formats.GROUP, dstKpad // formats.GROUP
    const firstGroup = colStart // formats.GROUP
    const groupCount = (colCount + formats.GROUP - 1) // formats.GROUP
    if packed and aligned then
        -- Bytes of one group of a tile: 256 for p8, 128 for p4, 528 for v12.
        const groupBytes = if layout == "p8" then 256 elseif layout == "p4" then 128 else formats.V12_GROUP_BYTES * 12
        const srcRowsPadded = (info.rows + tileRows - 1) // tileRows * tileRows
        for row = 0, dst.rowsPadded - 1, tileRows do
            if rowStart + row >= srcRowsPadded then
                break
            end
            const sp = sourcePart(info, rowStart + row)
            const srcTile = (rowStart + row - sp.rowStart) // tileRows
            const dp, dstRow = tensor.locate(dst, row)
            const dstTile = dstRow // tileRows
            read(
                base + sp.offset + (srcTile * srcGroups + firstGroup) * groupBytes,
                groupCount * groupBytes,
                dp.buf,
                dp.offset + dstTile * dstGroups * groupBytes
            )
            read(
                base
                    + (sp.scaleOffset :: number)
                    + (srcTile * tileRows * srcGroups + firstGroup * tileRows) * formats.SCALE_BYTES,
                groupCount * tileRows * formats.SCALE_BYTES,
                dp.scales :: buffer,
                dp.scaleOffset + dstTile * tileRows * dstGroups * formats.SCALE_BYTES
            )
        end
        return dst
    end
    if vector then
        -- The rows do not start on a tile of the source: read each source tile the slice touches, and move the f32
        -- lanes of its rows to where the slice's own tiles keep them.
        const groupBytes = formats.V12_GROUP_BYTES * 12
        const tileBytes = groupCount * groupBytes
        const tileScaleBytes = groupCount * 12 * formats.SCALE_BYTES
        const weights = buffer.create(tileBytes)
        const scales = buffer.create(tileScaleBytes)
        local sr = rowStart
        while sr < rowStart + rowCount do
            const sp = sourcePart(info, sr)
            const srcTile = (sr - sp.rowStart) // 12
            read(base + sp.offset + (srcTile * srcGroups + firstGroup) * groupBytes, tileBytes, weights, 0)
            read(
                base + (sp.scaleOffset :: number) + (srcTile * 12 * srcGroups + firstGroup * 12) * formats.SCALE_BYTES,
                tileScaleBytes,
                scales,
                0
            )
            const tileEnd = math.min(rowStart + rowCount, sp.rowStart + (srcTile + 1) * 12)
            while sr < tileEnd do
                const inTile = (sr - sp.rowStart) % 12
                const dp, dstRow = tensor.locate(dst, sr - rowStart)
                const lane = inTile // 3 * 12 + inTile % 3 * 4
                const dstLane = dstRow % 12 // 3 * 12 + dstRow % 3 * 4
                const dstTile = dstRow // 12
                for g = 0, groupCount - 1 do
                    const from = g * groupBytes + lane
                    const to = dp.offset + (dstTile * dstGroups + g) * groupBytes + dstLane
                    for step = 0, formats.V12_STEPS - 1 do
                        buffer.writeu32(dp.buf, to + step * 48, buffer.readu32(weights, from + step * 48))
                    end
                    buffer.writef32(
                        dp.scales :: buffer,
                        dp.scaleOffset + (dstTile * 12 * dstGroups + g * 12 + dstRow % 12) * formats.SCALE_BYTES,
                        buffer.readf32(scales, (g * 12 + inTile) * formats.SCALE_BYTES)
                    )
                end
                sr += 1
            end
        end
        return dst
    end
    const rowBytes = formats.rowBytes(format, srcKpad)
    const dstRowBytes = formats.rowBytes(format, dstKpad)
    const colBytes = if quant then groupCount * formats.GROUP * 4 // 8 else colCount * formats.get(format).bits // 8
    const colOffset = colStart * formats.get(format).bits // 8
    const whole = colStart == 0 and colCount == info.cols and srcKpad == dstKpad
    local row = 0
    while row < rowCount do
        const sp = sourcePart(info, rowStart + row)
        const srcLocal = rowStart + row - sp.rowStart
        const dp, dstLocal = tensor.locate(dst, row)
        -- A full width slice reads a run of rows at once, otherwise one row at a time.
        const run = if whole then math.min(sp.rows - srcLocal, dp.rowCount - dstLocal, rowCount - row) else 1
        if whole then
            read(base + sp.offset + srcLocal * rowBytes, run * rowBytes, dp.buf, dp.offset + dstLocal * dstRowBytes)
        else
            read(
                base + sp.offset + srcLocal * rowBytes + colOffset,
                colBytes,
                dp.buf,
                dp.offset + dstLocal * dstRowBytes
            )
        end
        if quant then
            const scaleSrc = (sp.scaleOffset :: number) + (srcLocal * srcGroups + firstGroup) * formats.SCALE_BYTES
            const scaleDst = dp.scaleOffset + dstLocal * dstGroups * formats.SCALE_BYTES
            read(
                base + scaleSrc,
                if whole then run * srcGroups * formats.SCALE_BYTES else groupCount * formats.SCALE_BYTES,
                dp.scales :: buffer,
                scaleDst
            )
        end
        row += run
    end
    return dst
end

return {
    open = open,
    blob = blob,
    loadTensor = loadTensor,
    loadVector = loadVector,
    loadVectorSlice = loadVectorSlice,
    loadSlice = loadSlice,
}
