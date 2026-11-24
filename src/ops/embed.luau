--!native
-- Embedding lookup: copies the row of a token out of the embedding matrix.

const formats = require("@src/tensor/formats")
const tensor = require("@src/tensor")
const tensorTypes = require("@src/tensor/types")

--- Writes row `rowIndex` of a matrix of any format, `cols` values as f32, into `dst` at byte offset `dstOffset`.
const function row(t: tensorTypes.Tensor, rowIndex: number, dst: buffer, dstOffset: number)
    assert(rowIndex >= 0 and rowIndex < t.rows, "token id outside the vocabulary")
    const part, localRow = tensor.locate(t, rowIndex)
    if t.format == "f32" then
        buffer.copy(dst, dstOffset, part.buf, part.offset + localRow * t.kpad * 4, t.cols * 4)
        return
    end
    if t.format == "bf16" then
        for k = 0, t.cols - 1 do
            const bits = buffer.readu16(part.buf, part.offset + (localRow * t.kpad + k) * 2)
            buffer.writef32(dst, dstOffset + k * 4, formats.bf16ToF32(bits))
        end
        return
    end
    const scales = part.scales :: buffer
    for k = 0, t.cols - 1 do
        const offset, nibble = formats.weightOffset(t.format, localRow, k, t.kpad, t.layout)
        const scaleAt = part.scaleOffset + formats.scaleOffset(localRow, k // formats.GROUP, t.kpad, t.layout)
        local weight: number
        if t.format == "q8" and t.layout == "v12" then
            const bytes = { formats.bytesOf(buffer.readf32(part.buf, part.offset + offset)) }
            weight = bytes[nibble + 1] - 128
        elseif t.format == "q8" then
            weight = buffer.readu8(part.buf, part.offset + offset) - 128
        else
            const byte = buffer.readu8(part.buf, part.offset + offset)
            weight = (if nibble == 0 then byte % 16 else byte // 16) - 8
        end
        buffer.writef32(dst, dstOffset + k * 4, weight * buffer.readf32(scales, scaleAt))
    end
end

return {
    row = row,
}
