-- Measures what tensor parallel inference in Lute depends on, with worker VMs from `parallel_worker.luau`.
--
--   lute run tools/bench/parallel.luau exchange <workers> <bytes> [blocks=9]
--       the time of one exchange: a request of `bytes` bytes to every worker and the join of the replies (L2)
--   lute run tools/bench/parallel.luau kernel <workers> <kernel id> <rows> <cols> <tokens> [copies=8] [repeats=7]
--       every worker runs the kernel over a matrix of its own at the same time, and the rate of the whole (L3, L4)
--   lute run tools/bench/parallel.luau vector <workers>
--       whether the probe finds the vector read fast inside a worker VM, with the main VM's answer (L6)
--
-- Each prints one `EXCH`, `KERN` or `VEC` line. Run it under `taskset -c 0-13`, several processes, see docs/benchmarks.md.
const lute = require("@backends/lute")
const probe = require("@src/kernels/probe")
const process = require("@std/process")
const vm = require("@lute/vm")

const mode = process.args[2]
const workers = tonumber(process.args[3]) or 1
if mode == nil then
    print("usage: lute run tools/bench/parallel.luau exchange|kernel|vector <workers> ...")
    process.exit(2)
end

const handles: { { [any]: any } } = {}
const transports = {}
for i = 1, workers do
    handles[i] = vm.create("./parallel_worker")
    transports[i] = lute.vmTransport(handles[i])
end
const join = transports[1].join

--- Sends `op` with `...` to every worker and waits. Returns the replies and the seconds it took.
const function fan(op: string, ...: any): ({ any }, number)
    const start = os.clock()
    const futures = {}
    for i = 1, workers do
        futures[i] = transports[i].call(op, ...)
    end
    const replies = join(futures)
    return replies, os.clock() - start
end

const function median(values: { number }): number
    table.sort(values)
    return values[(#values + 1) // 2]
end

if mode == "exchange" then
    const bytes = tonumber(process.args[4]) or 2304
    const blocks = tonumber(process.args[5]) or 9
    const message = buffer.create(bytes)
    for _ = 1, 100 do
        fan("echo", message)
    end
    const times: { number } = {}
    for b = 1, blocks do
        const n = 300
        local total = 0
        for _ = 1, n do
            const _, seconds = fan("echo", message)
            total += seconds
        end
        times[b] = total / n
    end
    print(
        string.format(
            "EXCH workers=%d bytes=%d exchange_us_med=%.1f min=%.1f max=%.1f",
            workers,
            bytes,
            median(table.clone(times)) * 1e6,
            math.min(table.unpack(times)) * 1e6,
            math.max(table.unpack(times)) * 1e6
        )
    )
elseif mode == "kernel" then
    const id = process.args[4] :: string
    const rows = tonumber(process.args[5]) or 2048
    const cols = tonumber(process.args[6]) or 2048
    const tokens = tonumber(process.args[7]) or 1
    const copies = tonumber(process.args[8]) or 8
    const repeats = tonumber(process.args[9]) or 7
    const sizes = fan("setup", id, rows, cols, copies, tokens)
    const bytes = sizes[1] :: number
    -- Calls that take about 0.25 s on one worker, from one worker running alone.
    const single = (handles[1].run :: any)(2) / 2
    const calls = math.max(2, math.floor(0.25 / math.max(single, 1e-6) + 0.5))
    fan("run", 2)
    const walls, innerMin, innerMax = {}, {}, {}
    for r = 1, repeats do
        const times, wall = fan("run", calls)
        walls[r] = wall
        innerMin[r] = math.min(table.unpack(times))
        innerMax[r] = math.max(table.unpack(times))
    end
    const wall = median(table.clone(walls))
    print(
        string.format(
            "KERN %s workers=%d M=%d K=%d T=%d bytes_per_worker=%d calls=%d wall_med=%.4f inner_min_med=%.4f inner_max_med=%.4f macs_per_s=%.4e gb_per_s=%.2f single_call_s=%.6f",
            id,
            workers,
            rows,
            cols,
            tokens,
            bytes,
            calls,
            wall,
            median(innerMin),
            median(innerMax),
            workers * rows * cols * tokens * calls / wall,
            workers * bytes * calls / wall / 1e9,
            single
        )
    )
elseif mode == "vector" then
    const replies = fan("vector")
    local inside = 0
    for _, v in replies do
        if v then
            inside += 1
        end
    end
    print(
        string.format(
            "VEC workers=%d main=%s workers_with_vector=%d",
            workers,
            tostring(probe.run(os.clock).vectorRead),
            inside
        )
    )
else
    print(`unknown mode {mode}`)
    process.exit(2)
end
