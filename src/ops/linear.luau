--!native
-- `y = W x` for a matrix tensor, one kernel call per part and per block of rows.

const budget = require("@src/runtime/budget")
const kernelsTypes = require("@src/kernels/types")
const runtimeTypes = require("@src/runtime/types")
const tensorTypes = require("@src/tensor/types")

--- Rows per kernel call when a budget is given, a multiple of every tile height (4, 8 and 12 rows).
const BLOCK_ROWS = 2040

--- Multiplies `tokens` activation rows of `t.kpad` values in `x` with `t`, as `prepare.run` returns them with `xs`. Row i of token k lands at float
--- `k * yStride + i` of `y`. `tokens` must be a multiple of the kernel's tile tokens and `yStride` at least
--- `t.rowsPadded`. A budget is checked between blocks of rows.
const function apply(
    entry: kernelsTypes.Entry,
    t: tensorTypes.Tensor,
    x: buffer,
    xs: buffer?,
    y: buffer,
    yStride: number,
    tokens: number,
    b: runtimeTypes.Budget?
)
    assert(t.rowTile % entry.R == 0 and t.kpad % entry.kstep == 0, `kernel {entry.id} does not fit the stored tensor`)
    assert(tokens % entry.S == 0, "tokens must be a multiple of the kernel's tile tokens")
    assert(yStride >= t.rowsPadded, "the output rows are shorter than the stored rows")
    for _, part in t.parts do
        const yBase = part.rowStart * 4
        if b == nil then
            entry.fn(
                part.buf,
                part.offset,
                part.scales,
                part.scaleOffset,
                x,
                xs,
                y,
                yBase,
                yStride,
                0,
                part.rowCount,
                t.kpad,
                tokens
            )
        else
            local row = 0
            while row < part.rowCount do
                const rows = math.min(BLOCK_ROWS, part.rowCount - row)
                entry.fn(
                    part.buf,
                    part.offset,
                    part.scales,
                    part.scaleOffset,
                    x,
                    xs,
                    y,
                    yBase,
                    yStride,
                    row,
                    rows,
                    t.kpad,
                    tokens
                )
                row += rows
                budget.check(b)
            end
        end
    end
end

return {
    apply = apply,
}
