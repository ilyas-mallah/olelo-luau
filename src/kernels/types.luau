--!native
-- Types shared by the modules of src/kernels.

const tensorTypes = require("@src/tensor/types")

--- Computes `y = W x` for the rows `row0` to `row0 + rows - 1` of one weight part and `T` tokens.
---
--- `w` holds the part's weights from byte `wOff`, rows numbered from 0 inside the part, each `K` stored weights long.
--- `ws` and `wsOff` are the part's scales, nil for f32 and BF16. `x` is `T` activation rows of `K` values, f32 for the
--- float formats and int8 for Q8 and Q4, and `xs` holds one f32 scale per token for the int8 ones. The packed Q8 tile
--- kernel reads `x` as f64 and `xs` as the token scales followed by the group sums, see `quant.packActivations`. Output for row `i`
--- of token `t` is the f32 at byte `yBase + (t * yStride + i) * 4` of `y`, so `yBase` places the part among the
--- rows of the whole matrix. `rows` is a multiple of the kernel's tile rows and `T` of its tile tokens.
export type Kernel = (
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
) -> ()

--- One kernel of a generated module with the tile it was generated from.
export type Tile = {
    name: string,
    R: number,
    S: number,
    U: number,
    --- Columns the stored row length is a multiple of.
    kstep: number,
    order: string,
    fmt: string?,
    --- `rv` for the vector read, `buf` for three `readf32` calls, nil for the scalar kernels.
    src: string?,
    fn: Kernel,
}

--- What a generated module returns.
export type Module = {
    --- Generator version that wrote the module.
    version: number,
    family: string,
    format: string,
    layout: string,
    --- Form of the activations the kernels read: `f32`, `int8`, `int8g` or `packed`.
    activation: string,
    tiles: { Tile },
}

--- How a kernel reads its weights.
export type Kind = "vector" | "buffer" | "scalar"

--- The form of the activations a kernel reads.
export type Activation = "f32" | "int8" | "int8g" | "packed" | "vsplat"

--- What the host can do, as the probe measures it.
export type Caps = {
    --- `buffer.readvectorf32x3` exists and its fast call makes it quick.
    vectorRead: boolean,
}

--- One generated kernel.
export type Entry = {
    --- `<module>/<tile>`, for example `f32_vector/R8S1U8`.
    id: string,
    module: string,
    name: string,
    format: tensorTypes.FormatName,
    layout: tensorTypes.LayoutName,
    --- The form of the activations the kernel reads: `f32`, `int8` (a scale per token), `int8g` (a scale per group of
    --- 32) or `packed` (`quant.packActivations`).
    activation: Activation,
    kind: Kind,
    --- Rows, tokens and unroll of the tile.
    R: number,
    S: number,
    U: number,
    --- Rows a stored tensor is padded to: , or a whole tile of the layout when that is larger.
    rowTile: number,
    --- Columns the stored row length is a multiple of.
    kstep: number,
    --- True when the kernel calls `buffer.readvectorf32x3`.
    needsVectorRead: boolean,
    fn: Kernel,
    --- The plain Luau kernel with the same signature and results.
    reference: Kernel,
}

return {}
