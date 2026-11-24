--!native
-- The matrix kernels: which one a host runs, what it can run, and the plain references they are tested against. The public face of `src/kernels`: its types and the modules a caller uses.

const types = require("@self/types")

export type Activation = types.Activation
export type Caps = types.Caps
export type Entry = types.Entry
export type Kernel = types.Kernel
export type Kind = types.Kind
export type Module = types.Module
export type Tile = types.Tile

return {
    probe = require("@self/probe"),
    reference = require("@self/reference"),
    registry = require("@self/registry"),
}
