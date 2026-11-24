--!native
-- Types shared by the modules of src/api.

const config = require("@src/model/config")
const host = require("@src/host")
const kernelsTypes = require("@src/kernels/types")
const modelTypes = require("@src/model/types")
const runtimeTypes = require("@src/runtime/types")
const samplingTypes = require("@src/sampling/types")
const tokenizerTypes = require("@src/tokenizer/types")

--- Settings of a conversation.
export type ChatOptions = {
    --- The system message, the model's own default when nil.
    system: string?,
    --- Seed of this conversation's sampling stream, the engine's seed when nil.
    seed: number?,
}

--- Settings of one `send`, on top of the sampling options.
export type SendOptions = samplingTypes.Options & {
    --- Most new tokens, 256 when nil.
    maxTokens: number?,
    --- Text that ends the reply and is not shown.
    stop: { string }?,
    --- Goes on past the model's end of turn tokens until `maxTokens`, to time a fixed number of tokens.
    ignoreStop: boolean?,
    --- Receives each piece of text instead of `send` returning an iterator.
    onText: ((piece: string) -> ())?,
}

--- Numbers of the last `send`.
export type Stats = {
    promptTokens: number,
    newTokens: number,
    --- Seconds from `send` to the first piece of text, prefill included.
    timeToFirstToken: number,
    --- Prompt tokens per second of prefill, the ones copied from a shared prefix not counted.
    prefillRate: number,
    --- Prompt tokens whose cache came from a shared prefix.
    cachedTokens: number,
    --- New tokens per second after the first.
    decodeRate: number,
    --- "stop", "length", "context", "cancelled" or "text" when a stop string ended the reply.
    finish: string,
}

--- What one step of a turn did.
export type StepResult = {
    --- Text the step produced, in the order to show it. Empty for a prefill step.
    pieces: { string },
    --- True when the reply ended. `stats().finish` says why.
    done: boolean,
    --- "prefill" or "decode".
    kind: string,
    --- Prompt tokens prefilled or new tokens sampled by the step.
    tokens: number,
}

--- One reply in progress. Its steps run the model, so call them one after the other from one thread.
export type Turn = {
    --- Prefills one chunk of the prompt, or samples up to `maxTokens` tokens (1 when nil) once the prompt is in.
    step: (self: Turn, maxTokens: number?) -> StepResult,
    --- Ends the reply at its next step.
    cancel: (self: Turn) -> (),
}

--- Everything a conversation needs to go on after its cache was dropped: its tokens, and whether the last of them
--- was sampled but is not in the cache yet.
export type State = {
    tokens: { number },
    pending: boolean,
}

export type Chat = {
    --- Starts a reply to a user message without running any of it. Errors when the prompt does not fit.
    begin: (self: Chat, text: string, options: SendOptions?) -> Turn,
    --- Sends a user message and returns an iterator of text pieces. With `onText` it runs to the end and the iterator
    --- is empty.
    send: (self: Chat, text: string, options: SendOptions?) -> () -> string?,
    --- Ends the reply in progress at the next token.
    cancel: (self: Chat) -> (),
    stats: (self: Chat) -> Stats,
    --- Positions of the context in use.
    used: (self: Chat) -> number,
    --- Every token of the conversation so far, prompts and replies.
    tokens: (self: Chat) -> { number },
    --- Makes the conversation `messages`, which end with a reply (none forgets it), as the chat template renders
    --- them. The positions of the cache that already hold the same tokens are kept, the rest is prefilled. Returns
    --- the number of kept positions.
    load: (self: Chat, messages: { tokenizerTypes.Message }) -> number,
    --- `load` in steps: returns the turn whose steps prefill one chunk each, and the number of kept positions. The
    --- chat is not usable before the last step. Cancelling it leaves the cache short of the messages, so `load` again.
    restore: (self: Chat, messages: { tokenizerTypes.Message }) -> (Turn, number),
    --- The state to rebuild this conversation from with `resume`.
    snapshot: (self: Chat) -> State,
    --- Rebuilds the cache of a conversation from a `snapshot`, in steps like `restore`. The tokens are the ones the
    --- conversation had, so a reply goes on exactly as it would have.
    resume: (self: Chat, state: State) -> Turn,
    --- Forgets the conversation.
    reset: (self: Chat) -> (),
    --- Releases what the conversation holds in the runner. Do not use the chat afterwards.
    close: (self: Chat) -> (),
}

export type Core = {
    host: host.Host,
    config: config.Config,
    --- The whole model when it runs in this VM, nil when workers hold it.
    model: modelTypes.Model?,
    runner: modelTypes.Runner,
    tokenizer: tokenizerTypes.Tokenizer,
    budget: runtimeTypes.Budget,
    --- Tokens per prefill pass.
    chunk: number,
    --- Seeds the samplers of new chats.
    seed: number,
    --- Prefixes that conversations share, nil when the engine keeps none.
    prefixes: Store?,
}

--- What `new` takes.
export type Options = {
    host: host.Host,
    source: host.WeightSource,
    --- Positions a conversation may use, 2048 when nil.
    context: number?,
    --- Seed of the sampling streams, 1 when nil.
    seed: number?,
    --- Tokens per prefill pass, 32 when nil.
    chunk: number?,
    --- How many system prompt prefixes the engine keeps the cache of, so conversations that start with the same system
    --- prompt prefill it once. 8 when nil, 0 for none.
    prefixes: number?,
    --- Seconds of work between yields to the host, 0 for never, which is the default.
    slice: number?,
    --- Host memory limit in bytes. The load fails with the numbers when the model does not fit.
    memoryLimit: number?,
    --- The KV cache format of conversations, "f32" or "q8" (int8 with a scale per head and position, a quarter of the
    --- memory). "f32" when nil.
    kv: string?,
    --- How a Q8 cache stores its rows, "v3" (the vector layout, 1.5 times the bytes, needs the vector read) or "i8".
    --- Chosen from the kernel capabilities when nil. For benchmarks that compare the two.
    kvLayout: string?,
    --- Kernel capabilities, measured by the probe when nil.
    caps: kernelsTypes.Caps?,
    --- Lines to workers that hold the layers between them, loaded with the same container, context and chunk. When
    --- given this VM loads the embedding and the norms only, and every pass is divided among the workers.
    transports: { host.Transport }?,
}

export type Engine = {
    core: Core,
    caps: kernelsTypes.Caps,
    --- The KV cache format and the positions of a conversation.
    kv: string,
    context: number,
    --- A new conversation.
    chat: (self: Engine, options: ChatOptions?) -> Chat,
    tokenize: (self: Engine, text: string) -> { number },
    detokenize: (self: Engine, ids: { number }) -> string,
    --- The greedy or sampled continuation of `ids` as ids, in a fresh session. For tests and tools.
    generate: (self: Engine, ids: { number }, maxTokens: number, options: samplingTypes.Options?) -> { number },
    --- A session for forward passes of up to `tokens` tokens. Errors when workers hold the model.
    session: (self: Engine, tokens: number?) -> modelTypes.Session,
    --- Ends the workers, if there are any.
    close: (self: Engine) -> (),
    --- The memory the loaded model uses, estimated.
    memory: (self: Engine) -> modelTypes.Estimate,
}

export type Entry = {
    tokens: { number },
    key: number,
    --- A counter value that grows with each use, for dropping the least recently used entry.
    used: number,
}

export type Store = {
    runner: modelTypes.Runner,
    entries: { Entry },
    capacity: number,
    nextKey: number,
    tick: number,
    hits: number,
    misses: number,
}

return {}
