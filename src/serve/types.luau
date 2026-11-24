--!native
-- Types shared by the modules of src/serve.

const apiTypes = require("@src/api/types")
const host = require("@src/host")
const tokenizerTypes = require("@src/tokenizer/types")

--- The host the pool runs on.
export type Machine = {
    --- Threads the pool may use, one per replica VM or worker VM.
    threads: number,
    memoryBytes: number,
    --- Memory left to the rest of the system, 1 GiB when nil.
    reserveBytes: number?,
}

--- What a replica of the model costs, from `unit.Info`.
export type Model = {
    weightBytes: number,
    --- The cache of one conversation.
    sessionBytes: number,
}

export type Layout = {
    --- Lanes of the scheduler. A lane is a replica when `workers` is 1, otherwise a group of that many workers.
    lanes: number,
    workers: number,
    --- Conversations a lane holds at once, the others wait or are dropped and rebuilt.
    slots: number,
    --- Estimated bytes of the whole pool.
    bytes: number,
    --- Whether `bytes` fit the memory that is usable. False when not even one replica does.
    fits: boolean,
    --- Why, in a sentence.
    reason: string,
}

--- One lane of the pool.
export type LaneSpec = {
    transport: host.Transport,
    --- Workers the lane uses, 1 for a replica.
    workers: number?,
    --- Conversations it may hold at once, 1 when nil.
    slots: number?,
}

--- Settings of a request on top of the reply options. `onText` receives each piece as it arrives.
export type RequestOptions = ReplyOptions & {
    onDone: ((reply: Reply) -> ())?,
    queueTimeout: number?,
    timeout: number?,
}

--- How a request ended. `finish` is "stop", "length", "context" or "text" as in `apiTypes.Stats`, or "cancelled",
--- "timeout" (the reply ran too long), "queue_timeout", "rejected" (too many requests), "closed" (its user left) or
--- "error" with `error` set.
export type Reply = {
    id: number,
    user: string,
    text: string,
    finish: string,
    error: string?,
    promptTokens: number,
    newTokens: number,
    --- Prompt tokens whose cache came from a prefix shared with other conversations.
    cachedTokens: number,
    --- Seconds from submit to the first token, and to the end.
    firstTokenSeconds: number,
    totalSeconds: number,
    --- Seconds spent waiting for a lane, rebuilding a dropped cache included.
    waitSeconds: number,
    decodeRate: number,
}

export type Request = {
    id: number,
    user: string,
    --- "waiting", "running" or "done".
    state: string,
    reply: Reply?,
    --- Ends the request: at once when it has not started, at its next token when it runs.
    cancel: (self: Request) -> (),
}

export type LaneStats = {
    workers: number,
    slots: number,
    resident: number,
    steps: number,
    busySeconds: number,
    tokens: number,
}

export type UserStats = {
    tokens: number,
    seconds: number,
    vtime: number,
}

export type Stats = {
    lanes: { LaneStats },
    users: { [string]: UserStats },
    submitted: number,
    finished: number,
    refused: number,
    waiting: number,
}

export type Scheduler = {
    --- Sets what a user's conversation starts with. Call before the user's first request.
    configure: (self: Scheduler, userId: string, system: string?, seed: number?, weight: number?) -> (),
    --- Queues the reply to `text` in the conversation of `userId`, which is created by its first request.
    submit: (self: Scheduler, userId: string, text: string, options: RequestOptions?) -> Request,
    --- Ends the user's requests and forgets the conversation.
    close: (self: Scheduler, userId: string) -> (),
    --- One round of work. Returns true while any request is unfinished or any lane is busy.
    pump: (self: Scheduler) -> boolean,
    --- Pumps until nothing is left, calling `wait` between rounds.
    run: (self: Scheduler, wait: () -> ()) -> (),
    --- The messages a user's conversation holds (those of finished requests).
    messages: (self: Scheduler, userId: string) -> { tokenizerTypes.Message },
    stats: (self: Scheduler) -> Stats,
}

--- What `step` returns.
export type StepOut = {
    --- The text of the step, the pieces joined.
    text: string,
    done: boolean,
    --- "prefill" or "decode".
    kind: string,
    tokens: number,
    --- Seconds the step took on this unit's clock.
    seconds: number,
    --- Set with `done` of a reply: why it ended, and the numbers of `apiTypes.Stats`.
    finish: string?,
    --- The conversation's `apiTypes.State` after the reply, which `open` takes to rebuild it.
    history: { number }?,
    pending: boolean?,
    promptTokens: number?,
    newTokens: number?,
    --- Prompt tokens whose cache came from a shared prefix.
    cachedTokens: number?,
    prefillRate: number?,
    decodeRate: number?,
}

--- What a conversation needs to start.
export type OpenOptions = {
    system: string?,
    seed: number?,
}

--- What `info` returns.
export type Info = {
    --- Conversations held, and how many may be.
    sessions: number,
    --- The engine's memory estimate, whole model, and the cache of one conversation.
    weightBytes: number,
    sessionBytes: number,
    kv: string,
    context: number,
}

--- The reply options a request carries across a transport: sampling, `maxTokens` and `stop`. `onText` is not carried.
export type ReplyOptions = apiTypes.SendOptions

export type Unit = {
    --- Starts conversation `sid`. With a `state` (from a finished reply) its cache is rebuilt in steps, which the caller
    --- steps to the end before the next reply.
    open: (self: Unit, sid: number, options: OpenOptions?, state: apiTypes.State?) -> (),
    --- Starts the reply to `text` in conversation `sid`. Errors when the prompt does not fit.
    begin: (self: Unit, sid: number, text: string, options: ReplyOptions?) -> (),
    --- Runs the job of `sid` one step: a prefill chunk, or up to `maxTokens` tokens.
    step: (self: Unit, sid: number, maxTokens: number?) -> StepOut,
    --- Ends the job of `sid` at its next step.
    cancel: (self: Unit, sid: number) -> (),
    --- Forgets conversation `sid` and frees its cache.
    close: (self: Unit, sid: number) -> (),
    info: (self: Unit) -> Info,
}

return {}
