-- The description of one kernel tile and the signature every generated kernel has.

--- One kernel to generate. `R` rows, `S` tokens and `U` unroll, `order` is when weights load against the multiplies.
--- `fmt`, `src`, `layout` and `conv` are read by the families they apply to.
export type Tile = {
    name: string,
    R: number,
    S: number,
    U: number,
    --- `all` loads a row's weights then multiplies, `elem` loads each right before its multiply, `gN` loads N at a time.
    order: string,
    --- `f32` or `bf16` for the float families.
    fmt: string?,
    --- `rv` reads vectors with `buffer.readvectorf32x3`, `buf` builds them from three `readf32`.
    src: string?,
    --- Q8 weight layout, `rm` or `il4e`.
    layout: string?,
}

--- One generated module: a file, a family that renders its tiles and the tiles.
export type ModuleSpec = {
    --- File name under `src/kernels/generated/`, without `.luau`.
    file: string,
    family: string,
    --- Format name the registry files the kernels under.
    format: string,
    layout: string,
    --- The form of the activations the kernels read, `f32`, `int8`, `int8g` or `packed`.
    activation: string,
    --- `strict` (the language mode of `.config.luau`, no header), or `nonstrict` where the module calls
    --- `buffer.readvectorf32x3`, which the type checker does not know.
    strictness: string,
    --- Short description written into the module header.
    summary: string,
    tiles: { Tile },
}

--- The parameters of every generated kernel. `ws` and `xs` are nil for formats without scales. `row0` and `rows` pick
--- rows inside the part, `yBase` is the byte offset in `y` of part row 0 of token 0 and `yStride` the floats per token.
const SIGNATURE = "w: buffer, wOff: number, ws: buffer?, wsOff: number, x: buffer, xs: buffer?, y: buffer, "
    .. "yBase: number, yStride: number, row0: number, rows: number, K: number, T: number"

return {
    SIGNATURE = SIGNATURE,
}
