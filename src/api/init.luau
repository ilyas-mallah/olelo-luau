--!native
-- The API layer: a model behind an engine, and the chats that run on it. The public face of `src/api`: its types and the modules a caller uses.

const types = require("@self/types")

export type Chat = types.Chat
export type ChatOptions = types.ChatOptions
export type Core = types.Core
export type Engine = types.Engine
export type Entry = types.Entry
export type Options = types.Options
export type SendOptions = types.SendOptions
export type State = types.State
export type Stats = types.Stats
export type StepResult = types.StepResult
export type Store = types.Store
export type Turn = types.Turn

return {
    chat = require("@self/chat"),
    engine = require("@self/engine"),
    prefix = require("@self/prefix"),
}
