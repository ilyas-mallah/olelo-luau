--!native
-- Plain Luau references for the Q8 and Q4 kernels. Integer products are summed per group of 32 in doubles, which is
-- exact, then scaled and accumulated in group order, so the result matches a tiled kernel exactly.

const formats = require("@src/tensor/formats")
const kernelsTypes = require("@src/kernels/types")
const tensorTypes = require("@src/tensor/types")

const GROUP = formats.GROUP

--- The Q8 reference kernel for `layout`, `il4e` or `p8`. Both take int8 activations: `il4e` with one scale per token,
--- `p8` with one per group of 32, whatever form the kernel under test reads.
const function q8For(layout: tensorTypes.LayoutName): kernelsTypes.Kernel
    return function(
        w: buffer,
        wOff: number,
        ws: buffer?,
        wsOff: number,
        x: buffer,
        xs: buffer?,
        y: buffer,
        yBase: number,
        yStride: number,
        row0: number,
        rows: number,
        K: number,
        T: number
    )
        const scales = ws :: buffer
        const activationScales = xs :: buffer
        const packed = layout == "p8"
        const groups = K // GROUP
        for t = 0, T - 1 do
            for i = row0, row0 + rows - 1 do
                local acc = 0
                for g = 0, K // GROUP - 1 do
                    local sum = 0
                    for j = 0, GROUP - 1 do
                        const k = g * GROUP + j
                        const at = wOff + formats.weightOffset("q8", i, k, K, layout)
                        const weight = if packed then buffer.readu8(w, at) - 128 else buffer.readi8(w, at)
                        sum += weight * buffer.readi8(x, t * K + k)
                    end
                    const weightScale = buffer.readf32(scales, wsOff + formats.scaleOffset(i, g, K, layout))
                    if packed then
                        acc += weightScale * buffer.readf32(activationScales, (t * groups + g) * 4) * sum
                    else
                        acc += weightScale * sum
                    end
                end
                buffer.writef32(
                    y,
                    yBase + (t * yStride + i) * 4,
                    if packed then acc else acc * buffer.readf32(activationScales, t * 4)
                )
            end
        end
    end
end

--- The Q8 reference kernel for the `v12` layout: the weights come out of their f32 lanes, the activations are int8 with a
--- scale per group of 32, whatever form the kernel under test reads.
const function q8v(
    w: buffer,
    wOff: number,
    ws: buffer?,
    wsOff: number,
    x: buffer,
    xs: buffer?,
    y: buffer,
    yBase: number,
    yStride: number,
    row0: number,
    rows: number,
    K: number,
    T: number
)
    const scales = ws :: buffer
    const activationScales = xs :: buffer
    const groups = K // GROUP
    for t = 0, T - 1 do
        for i = row0, row0 + rows - 1 do
            local acc = 0
            for g = 0, groups - 1 do
                local sum = 0
                for j = 0, GROUP - 1 do
                    const k = g * GROUP + j
                    const at, field = formats.weightOffset("q8", i, k, K, "v12")
                    const bytes = { formats.bytesOf(buffer.readf32(w, wOff + at)) }
                    sum += (bytes[field + 1] - 128) * buffer.readi8(x, t * K + k)
                end
                acc += buffer.readf32(scales, wsOff + formats.scaleOffset(i, g, K, "v12")) * buffer.readf32(
                    activationScales,
                    (t * groups + g) * 4
                ) * sum
            end
            buffer.writef32(y, yBase + (t * yStride + i) * 4, acc)
        end
    end
end

--- The Q4 reference kernel for `layout`, `rowmajor` or `p4`. The weight is the stored nibble minus 8, the activations
--- int8 with a scale per group.
const function q4For(layout: tensorTypes.LayoutName): kernelsTypes.Kernel
    return function(
        w: buffer,
        wOff: number,
        ws: buffer?,
        wsOff: number,
        x: buffer,
        xs: buffer?,
        y: buffer,
        yBase: number,
        yStride: number,
        row0: number,
        rows: number,
        K: number,
        T: number
    )
        const scales = ws :: buffer
        const activationScales = xs :: buffer
        const groups = K // GROUP
        for t = 0, T - 1 do
            for i = row0, row0 + rows - 1 do
                local acc = 0
                for g = 0, K // GROUP - 1 do
                    local sum = 0
                    for j = 0, GROUP - 1 do
                        const k = g * GROUP + j
                        const offset, nibble = formats.weightOffset("q4", i, k, K, layout)
                        const byte = buffer.readu8(w, wOff + offset)
                        const q = if nibble == 0 then byte % 16 else byte // 16
                        sum += (q - 8) * buffer.readi8(x, t * K + k)
                    end
                    acc += buffer.readf32(scales, wsOff + formats.scaleOffset(i, g, K, layout)) * buffer.readf32(
                        activationScales,
                        (t * groups + g) * 4
                    ) * sum
                end
                buffer.writef32(y, yBase + (t * yStride + i) * 4, acc)
            end
        end
    end
end

return {
    q8 = q8For("il4e"),
    q8p = q8For("p8"),
    q8v = q8v,
    q4 = q4For("rowmajor"),
    q4p = q4For("p4"),
}
