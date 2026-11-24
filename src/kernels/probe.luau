--!native
-- The startup capability probe. It decides whether the vector kernels run: `buffer.readvectorf32x3` must exist and
-- its fast call must make the vector kernel clearly faster than the one built from three `readf32` calls. Without the
-- fast call the vector read is an ordinary call and the vector kernel runs about 14 times slower than the fallback.

const registry = require("@src/kernels/registry")
const tensor = require("@src/tensor")
const types = require("./types")

--- What the probe found.
export type Result = {
    --- The vector kernels may be selected.
    vectorRead: boolean,
    --- Why, in one sentence, for logs and tests.
    reason: string,
    --- MACs per second of the vector kernel, nil when it could not run.
    vectorRate: number?,
    --- MACs per second of the 3 x readf32 kernel, nil when the vector read is missing.
    bufferRate: number?,
}

--- The vector kernel must beat the fallback by this factor to be chosen.
const MIN_RATIO = 1.5

--- Matrix shape of the probe, one decode sized layer of SmolLM2-135M's MLP.
const ROWS = 1536
const COLS = 576

--- Seconds each kernel is timed for.
const BUDGET = 0.01

--- Timings before the probe says the vector read is slow. CPU load can slow the vector kernel for one timing, and a
--- model stored in the vector layout cannot run without the answer, so a failed probe is repeated and the best ratio
--- counts.
const ATTEMPTS = 3

--- Times `entry` on a zero free matrix and returns MACs per second.
const function rate(entry: types.Entry, clock: () -> number, budget: number): number
    const t = tensor.new({ format = "f32", rows = ROWS, cols = COLS, rowTile = entry.R, kstep = entry.kstep })
    const part = t.parts[1]
    -- Every byte 0x3F reads back as a normal f32, so no kernel meets zeros or denormals.
    buffer.fill(part.buf, 0, 0x3F)
    const x = buffer.create(t.kpad * 4)
    buffer.fill(x, 0, 0x3F)
    const y = buffer.create(t.rowsPadded * 4)
    const run = entry.fn
    run(part.buf, part.offset, nil, 0, x, nil, y, 0, t.rowsPadded, 0, t.rowsPadded, t.kpad, 1)
    local calls = 0
    const start = clock()
    local now = start
    repeat
        run(part.buf, part.offset, nil, 0, x, nil, y, 0, t.rowsPadded, 0, t.rowsPadded, t.kpad, 1)
        calls += 1
        now = clock()
    until now - start >= budget
    return ROWS * COLS * calls / (now - start)
end

--- Runs the probe. `clock` is a monotonic clock in seconds, `os.clock` when nil, and `budget` the seconds each kernel
--- is timed for.
const function run(clock: (() -> number)?, budget: number?): Result
    if (buffer :: any).readvectorf32x3 == nil then
        return { vectorRead = false, reason = "buffer.readvectorf32x3 does not exist on this host" }
    end
    local now = clock or os.clock
    const seconds = budget or BUDGET
    local bestVector, bestBuffer, bestRatio = 0, 1, -1
    for _ = 1, ATTEMPTS do
        const bufferRate = rate(registry.find("f32_buffer/R8S1U8"), now, seconds)
        const vectorRate = rate(registry.find("f32_vector/R8S1U8"), now, seconds)
        if vectorRate / bufferRate > bestRatio then
            bestVector, bestBuffer, bestRatio = vectorRate, bufferRate, vectorRate / bufferRate
        end
        if bestRatio >= MIN_RATIO then
            break
        end
    end
    if bestRatio >= MIN_RATIO then
        return {
            vectorRead = true,
            reason = `the vector kernel is {string.format("%.1f", bestRatio)} times the 3 x readf32 kernel`,
            vectorRate = bestVector,
            bufferRate = bestBuffer,
        }
    end
    return {
        vectorRead = false,
        reason = `the vector read exists but is not fast, the vector kernel is {string.format("%.2f", bestRatio)} times the 3 x readf32 kernel`,
        vectorRate = bestVector,
        bufferRate = bestBuffer,
    }
end

return {
    MIN_RATIO = MIN_RATIO,
    run = run,
}
