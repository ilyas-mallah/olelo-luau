--!native
-- Plain Luau reference for the f32 and BF16 kernels. One dot product per output with a double accumulator, in
-- column order, no tiles, no unrolling.

const formats = require("@src/tensor/formats")

--- The f32 reference kernel.
const function f32(
    w: buffer,
    wOff: number,
    _ws: buffer?,
    _wsOff: number,
    x: buffer,
    _xs: buffer?,
    y: buffer,
    yBase: number,
    yStride: number,
    row0: number,
    rows: number,
    K: number,
    T: number
)
    for t = 0, T - 1 do
        for i = row0, row0 + rows - 1 do
            local acc = 0
            for k = 0, K - 1 do
                acc += buffer.readf32(w, wOff + (i * K + k) * 4) * buffer.readf32(x, (t * K + k) * 4)
            end
            buffer.writef32(y, yBase + (t * yStride + i) * 4, acc)
        end
    end
end

--- The BF16 reference kernel. A BF16 weight is the high half of the f32 with the same value.
const function bf16(
    w: buffer,
    wOff: number,
    _ws: buffer?,
    _wsOff: number,
    x: buffer,
    _xs: buffer?,
    y: buffer,
    yBase: number,
    yStride: number,
    row0: number,
    rows: number,
    K: number,
    T: number
)
    for t = 0, T - 1 do
        for i = row0, row0 + rows - 1 do
            local acc = 0
            for k = 0, K - 1 do
                const weight = formats.bf16ToF32(buffer.readu16(w, wOff + (i * K + k) * 2))
                acc += weight * buffer.readf32(x, (t * K + k) * 4)
            end
            buffer.writef32(y, yBase + (t * yStride + i) * 4, acc)
        end
    end
end

return {
    f32 = f32,
    bf16 = bf16,
}
