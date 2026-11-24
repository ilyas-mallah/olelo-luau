--!native
-- Types shared by the modules of src/tensor.

--- A weight format name.
export type FormatName = "f32" | "bf16" | "q8" | "q4"

--- A weight layout name. `rowmajor` is one row after another, `il4e` interleaves 4 rows element by element, `p8` is
--- the packed Q8 tile: 8 rows interleaved by column as offset binary bytes (weight + 128), with the scales of a tile
--- interleaved the same way. `p4` is the packed Q4 tile: a u32 per column holding the 8 rows of the tile as nibbles,
--- with the scales laid out as `p8` lays them. `v12` is the vector Q8 tile of 12 rows, see `docs/kernels.md`.
export type LayoutName = "rowmajor" | "il4e" | "p8" | "p4" | "v12"

--- A weight matrix.
export type Tensor = {
    format: FormatName,
    layout: LayoutName,
    --- Rows the model has.
    rows: number,
    --- Rows stored, `rows` rounded up to a multiple of `rowTile`. The extra rows are zero.
    rowsPadded: number,
    --- Columns the model has.
    cols: number,
    --- Stored weights per row, `cols` rounded up to a multiple of `kstep`. The extra columns are zero.
    kpad: number,
    --- Rows the kernel computes per pass.
    rowTile: number,
    parts: { Part },
}

--- What `new` needs to build a tensor.
export type Spec = {
    format: FormatName,
    rows: number,
    cols: number,
    --- Rows per kernel pass, a multiple of 4 for Q8.
    rowTile: number,
    --- Columns per kernel loop pass, from the kernel that reads the tensor.
    kstep: number,
    --- The layout, the format's own when nil. Q8 has two, `p8` for the shipped kernel and `il4e` for the reference one.
    layout: LayoutName?,
    --- Largest buffer in bytes, formats.MAX_BUFFER_BYTES when nil. Tests pass a small value to force parts.
    maxPartBytes: number?,
}

--- A run of rows held in one buffer. Row numbers inside a part start at 0.
export type Part = {
    --- Weights of the part, rows laid out as the tensor's layout says, starting at `offset`.
    buf: buffer,
    --- Byte offset of the part's first weight in `buf`.
    offset: number,
    --- Scales of the part, nil for formats without scales, starting at `scaleOffset`.
    scales: buffer?,
    --- Byte offset of the part's first scale in `scales`.
    scaleOffset: number,
    --- Row of the whole matrix that is row 0 of the part.
    rowStart: number,
    --- Rows in the part, a multiple of the tile rows.
    rowCount: number,
}

return {}
