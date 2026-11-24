--!native
-- Rotary position embedding in the half split form: element i of a head pairs with element i + headDim / 2.

const types = require("./types")

type Table = types.Table

--- Builds the table for `positions` positions of heads of `headDim` values with the given base frequency.
const function new(headDim: number, theta: number, positions: number): Table
    const half = headDim // 2
    const cos = buffer.create(positions * half * 4)
    const sin = buffer.create(positions * half * 4)
    for i = 0, half - 1 do
        const frequency = theta ^ (-2 * i / headDim)
        for p = 0, positions - 1 do
            const angle = p * frequency
            buffer.writef32(cos, (p * half + i) * 4, math.cos(angle))
            buffer.writef32(sin, (p * half + i) * 4, math.sin(angle))
        end
    end
    return { cos = cos, sin = sin, half = half, positions = positions }
end

--- Rotates the heads of `tokens` rows of `src` into `dst`. Token t sits at position `pos0 + t`. A head is `headDim`
--- values starting at `h * headDim` in the source row, and lands at `dstBase + t * tokenStride + h * headStride` bytes.
const function apply(
    r: Table,
    src: buffer,
    srcStride: number,
    heads: number,
    headDim: number,
    tokens: number,
    pos0: number,
    dst: buffer,
    dstBase: number,
    tokenStride: number,
    headStride: number
)
    const half = r.half
    for t = 0, tokens - 1 do
        const tableBase = (pos0 + t) * half * 4
        for h = 0, heads - 1 do
            const s = (t * srcStride + h * headDim) * 4
            const d = dstBase + t * tokenStride + h * headStride
            for i = 0, half - 1 do
                const c = buffer.readf32(r.cos, tableBase + i * 4)
                const sn = buffer.readf32(r.sin, tableBase + i * 4)
                const x1 = buffer.readf32(src, s + i * 4)
                const x2 = buffer.readf32(src, s + (i + half) * 4)
                buffer.writef32(dst, d + i * 4, x1 * c - x2 * sn)
                buffer.writef32(dst, d + (i + half) * 4, x2 * c + x1 * sn)
            end
        end
    end
end

--- Copies the heads of `tokens` rows of `src` without rotating, to the same destination layout as `apply`.
const function copy(
    src: buffer,
    srcStride: number,
    heads: number,
    headDim: number,
    tokens: number,
    dst: buffer,
    dstBase: number,
    tokenStride: number,
    headStride: number
)
    for t = 0, tokens - 1 do
        for h = 0, heads - 1 do
            buffer.copy(
                dst,
                dstBase + t * tokenStride + h * headStride,
                src,
                (t * srcStride + h * headDim) * 4,
                headDim * 4
            )
        end
    end
end

return {
    new = new,
    apply = apply,
    copy = copy,
}
