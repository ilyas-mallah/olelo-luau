--!native
-- Tensor parallel inference: the split of a model, the coordinator and the workers it calls. The public face of `src/parallel`: its types and the modules a caller uses.

const types = require("@self/types")

export type Share = types.Share
export type Stats = types.Stats
export type Worker = types.Worker

return {
    coordinator = require("@self/coordinator"),
    split = require("@self/split"),
    transport = require("@self/transport"),
    worker = require("@self/worker"),
}
