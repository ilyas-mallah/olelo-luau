--!native
-- Types shared by the modules of src/container.

const host = require("@src/host")

--- One part of a tensor in the container.
export type PartInfo = {
    offset: number,
    rowStart: number,
    rows: number,
    bytes: number,
    --- Quantized tensors only: where the part's scales are and how long.
    scaleOffset: number?,
    scaleBytes: number?,
}

--- One tensor in the table.
export type TensorInfo = {
    name: string,
    format: string,
    layout: string,
    rows: number,
    cols: number,
    kpad: number,
    rowTile: number,
    parts: { PartInfo },
}

--- An open container.
export type Container = {
    --- The decoded header: config, tokenizer, chat.
    header: any,
    source: host.WeightSource,
    tensors: { [string]: TensorInfo },
    dataStart: number,
    blobs: { [string]: { offset: number, bytes: number } },
}

return {}
