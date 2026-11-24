--!native
-- Types shared by the modules of src/parallel.

const modelTypes = require("@src/model/types")

--- The share of one worker. Counts and starts are in heads, MLP columns and vocabulary rows.
export type Share = {
    --- 1 to the number of workers.
    index: number,
    workers: number,
    headStart: number,
    headCount: number,
    --- The KV heads the worker's query heads read. A KV head whose query group spans two workers is in both shares.
    kvStart: number,
    kvCount: number,
    interStart: number,
    interCount: number,
    vocabStart: number,
    vocabCount: number,
}

--- What a worker counts, returned by `stats`.
export type Stats = {
    --- Seconds spent in each op and how many calls.
    attnSeconds: number,
    attnCalls: number,
    mlpSeconds: number,
    mlpCalls: number,
    logitsSeconds: number,
    logitsCalls: number,
    --- Bytes of requests received and of replies sent.
    bytesIn: number,
    bytesOut: number,
    --- Bytes of weights, scales included.
    weightBytes: number,
    --- Whether the worker's kernels read vectors.
    vectorRead: boolean,
}

export type Worker = {
    share: Share,
    model: modelTypes.Model,
    --- Weight bytes held.
    weightBytes: number,
    --- Partial sum of the attention block's output projection for `count` tokens of sequence `seq` at positions
    --- `pos0` and after, from `xn`, the `count` normalized rows at stride `hs`.
    attn: (self: Worker, seq: number, layer: number, pos0: number, count: number, xn: buffer) -> buffer,
    --- Partial sum of the MLP block's down projection.
    mlp: (self: Worker, seq: number, layer: number, count: number, xn: buffer) -> buffer,
    --- The worker's part of the logits of the normalized row `xn`, `share.vocabCount` floats.
    logits: (self: Worker, seq: number, xn: buffer) -> buffer,
    --- Forgets the positions of `seq` from `length` on.
    trim: (self: Worker, seq: number, length: number) -> (),
    --- Releases the cache of `seq`.
    free: (self: Worker, seq: number) -> (),
    --- Keeps the first `length` positions of `seq` under `key`, restores them into an empty sequence, forgets them.
    save: (self: Worker, seq: number, length: number, key: number) -> (),
    restore: (self: Worker, seq: number, key: number) -> (),
    drop: (self: Worker, key: number) -> (),
    --- The positions saved under `key`.
    length: (self: Worker, key: number) -> number,
    stats: (self: Worker) -> Stats,
}

return {}
