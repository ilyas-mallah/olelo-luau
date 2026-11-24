-- The generated modules and the tiles in each, as data. A kernel is one tile: rows R per pass, tokens S, unroll U and
-- the order weights load in. Rates are in `docs/kernels.md`.

const tile = require("./tile")

const modules: { tile.ModuleSpec } = {
    {
        file = "f32_vector",
        family = "float_vector",
        format = "f32",
        layout = "rowmajor",
        activation = "f32",
        strictness = "nonstrict",
        summary = "f32 weights, 3 lane vectors read with buffer.readvectorf32x3 (needs the fast call)",
        tiles = {
            -- The primary decode kernel. Each vector loads right before its multiply, so the live set stays small.
            { name = "R8S1U8", fmt = "f32", src = "rv", R = 8, S = 1, U = 8, order = "elem" },
            -- Two tokens at once for prefill.
            { name = "R4S2U2", fmt = "f32", src = "rv", R = 4, S = 2, U = 2, order = "all" },
        },
    },
    {
        file = "f32_buffer",
        family = "float_vector",
        format = "f32",
        layout = "rowmajor",
        activation = "f32",
        strictness = "strict",
        summary = "f32 weights, 3 lane vectors built from three readf32 calls (the fallback without the fast call)",
        tiles = {
            { name = "R8S1U8", fmt = "f32", src = "buf", R = 8, S = 1, U = 8, order = "all" },
        },
    },
    {
        file = "f32_scalar",
        family = "float_scalar",
        format = "f32",
        layout = "rowmajor",
        activation = "f32",
        strictness = "strict",
        summary = "f32 weights, scalar double math",
        tiles = {
            { name = "R8S1U8", fmt = "f32", R = 8, S = 1, U = 8, order = "all" },
        },
    },
    {
        file = "bf16_vector",
        family = "float_vector",
        format = "bf16",
        layout = "rowmajor",
        activation = "f32",
        strictness = "nonstrict",
        summary = "BF16 weights converted pairwise through a scratch buffer, 3 lane vectors (needs the fast call)",
        tiles = {
            -- Steps 24 columns, the stored row length every activation buffer shares, so a model can use it.
            { name = "R4S1U8", fmt = "bf16", src = "rv", R = 4, S = 1, U = 8, order = "elem" },
            { name = "R4S1U12", fmt = "bf16", src = "rv", R = 4, S = 1, U = 12, order = "elem" },
        },
    },
    {
        file = "bf16_scalar",
        family = "float_scalar",
        format = "bf16",
        layout = "rowmajor",
        activation = "f32",
        strictness = "strict",
        summary = "BF16 weights converted pairwise through a scratch buffer, scalar double math",
        tiles = {
            { name = "R8S1U8", fmt = "bf16", R = 8, S = 1, U = 8, order = "all" },
        },
    },
    {
        file = "q8_il4e",
        family = "q8",
        format = "q8",
        layout = "il4e",
        activation = "int8",
        strictness = "strict",
        summary = "int8 weights interleaved 4 rows by element, a scale per 32, int8 activations",
        tiles = {
            { name = "R8S1U8", layout = "il4e", R = 8, S = 1, U = 8, order = "all" },
        },
    },
    {
        file = "q8_p8",
        family = "q8p",
        format = "q8",
        layout = "p8",
        activation = "packed",
        strictness = "strict",
        summary = "offset binary int8 weights, 4 rows per word in tiles of 8 rows, a scale per 32, f64 activations",
        tiles = {
            { name = "R8S1U32", layout = "p8", R = 8, S = 1, U = 32, order = "c2" },
            -- Two tokens at once for prefill, one band of 4 rows, so the field extraction serves both tokens.
            { name = "R4S2U32", layout = "p8", R = 4, S = 2, U = 32, order = "c2" },
        },
    },
    {
        file = "q8_v12",
        family = "q8v",
        format = "q8",
        layout = "v12",
        activation = "vsplat",
        strictness = "nonstrict",
        summary = "int8 weights, 3 per f32 lane, 12 rows per tile, a scale per 32, splat vector activations (needs the fast call)",
        tiles = {
            { name = "R12S1U32", layout = "v12", fmt = "q8", src = "rv", R = 12, S = 1, U = 32, order = "step" },
            -- Two tokens at once for prefill, 6 rows a pass, which the Windows x64 spill limit allows (12 rows by 2 tokens does not).
            { name = "R6S2U32", layout = "v12", fmt = "q8", src = "rv", R = 6, S = 2, U = 32, order = "step" },
        },
    },
    {
        file = "q4_p4",
        family = "q4p",
        format = "q4",
        layout = "p4",
        activation = "packed",
        strictness = "strict",
        summary = "4 bit weights, 8 rows of a column in one word of nibbles, a scale per 32, f64 activations",
        tiles = {
            { name = "R8S1U32", layout = "p4", R = 8, S = 1, U = 32, order = "c2" },
            -- Two tokens at once for prefill, a band of 4 rows, so the nibble extraction serves both tokens.
            { name = "R4S2U32", layout = "p4", R = 4, S = 2, U = 32, order = "c2" },
        },
    },
    {
        file = "q4_byte",
        family = "q4",
        format = "q4",
        layout = "rowmajor",
        activation = "int8g",
        strictness = "strict",
        summary = "4 bit weights unpacked per byte, a scale per 32, int8 activations with a scale per 32",
        tiles = {
            { name = "R8S1U4", R = 8, S = 1, U = 4, order = "all" },
        },
    },
}

return {
    modules = modules,
}
