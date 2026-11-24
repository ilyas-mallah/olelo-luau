--!native
-- Types shared by the modules of src/runtime.

const host = require("@src/host")

export type Budget = {
    host: host.Host,
    --- Seconds of work between yields, 0 for never.
    slice: number,
    since: number,
}

return {}
