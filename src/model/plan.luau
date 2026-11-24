--!native
-- The memory a model needs before it is loaded, so a host that cannot hold it is told the numbers.

const config = require("./config")
const session = require("./session")
const containerTypes = require("@src/container/types")
const types = require("./types")

type Estimate = types.Estimate

--- The bytes the container's tensors take, plus the cache for `context` positions and the scratch for `tokens` tokens
--- per forward pass.
const function estimate(
    c: containerTypes.Container,
    context: number,
    tokens: number,
    kv: string?,
    layout: string?
): Estimate
    const cfg: config.Config = c.header.config
    local weights = 0
    local prepK = 0
    for _, info in c.tensors do
        if info.format ~= "f32" then
            prepK = math.max(prepK, info.kpad)
        end
        for _, part in info.parts do
            weights += part.bytes + (part.scaleBytes or 0)
        end
    end
    const format = kv or "f32"
    const cache = session.cacheBytes(cfg, context, format, layout)
    const workspace = session.workspaceBytes(cfg, tokens, config.roundUp(cfg.vocab, 24), prepK, format)
    return { weights = weights, cache = cache, workspace = workspace, total = weights + cache + workspace }
end

--- Errors with the numbers when the estimate does not fit `limit` bytes.
const function check(e: Estimate, limit: number)
    if e.total > limit then
        error(
            string.format(
                "the model needs %.0f MiB (weights %.0f, cache %.0f, scratch %.0f) and the limit is %.0f MiB",
                e.total / 1048576,
                e.weights / 1048576,
                e.cache / 1048576,
                e.workspace / 1048576,
                limit / 1048576
            ),
            0
        )
    end
end

return {
    estimate = estimate,
    check = check,
}
