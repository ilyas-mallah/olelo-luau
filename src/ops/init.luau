--!native
-- The operations of a decoder layer. The public face of `src/ops`: its types and the modules a caller uses.

const types = require("@self/types")

export type Activation = types.Activation
export type Scratch = types.Scratch
export type Shape = types.Shape
export type Table = types.Table

return {
    attention = require("@self/attention"),
    attentionQ8 = require("@self/attention_q8"),
    attentionVector = require("@self/attention_vector"),
    embed = require("@self/embed"),
    kvquant = require("@self/kvquant"),
    linear = require("@self/linear"),
    mlp = require("@self/mlp"),
    norm = require("@self/norm"),
    prepare = require("@self/prepare"),
    rope = require("@self/rope"),
    shortconv = require("@self/shortconv"),
}
