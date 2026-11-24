--!native
-- The tables a backend builds so the core never imports a host library.

--- Where weight bytes come from. Offsets are in the logical container, a file or a run of chunks.
export type WeightSource = {
    --- Total bytes.
    size: () -> number,
    --- Fills `dst` from `dstOffset` with `length` bytes starting at `offset`. Errors on a short read.
    read: (offset: number, length: number, dst: buffer, dstOffset: number) -> (),
}

--- The reply to a request, filled in by the backend when the worker has answered.
export type Future = {
    done: boolean,
    --- The worker's return value.
    value: any,
    --- The worker's error message when the request failed.
    failure: string?,
}

--- A line to one worker. A request is a function call on the worker, `op` names the function. Buffers in the arguments
--- and in the reply are copies, on every host, so the sender may reuse its buffer as soon as `call` returns.
export type Transport = {
    --- Sends the request and returns at once, without waiting for the reply.
    call: (op: string, ...any) -> Future,
    --- Waits until every future is done and returns their values in order. Errors with a worker's message when a
    --- request failed.
    join: ({ Future }) -> { any },
    close: () -> (),
}

--- What the core needs from the host it runs on.
export type Host = {
    name: string,
    --- Seconds, monotonic.
    clock: () -> number,
    --- Gives the frame back on a host that shares one, does nothing elsewhere.
    yield: () -> (),
    log: (level: string, message: string) -> (),
}

return {}
