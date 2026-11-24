--!native
-- Many users on a pool of lanes. The public face of `src/serve`: its types and the modules a caller uses.

const types = require("@self/types")

export type Info = types.Info
export type LaneSpec = types.LaneSpec
export type LaneStats = types.LaneStats
export type Layout = types.Layout
export type Machine = types.Machine
export type Model = types.Model
export type OpenOptions = types.OpenOptions
export type Reply = types.Reply
export type ReplyOptions = types.ReplyOptions
export type Request = types.Request
export type RequestOptions = types.RequestOptions
export type Scheduler = types.Scheduler
export type Stats = types.Stats
export type StepOut = types.StepOut
export type Unit = types.Unit
export type UserStats = types.UserStats

return {
    layout = require("@self/layout"),
    scheduler = require("@self/scheduler"),
    unit = require("@self/unit"),
}
