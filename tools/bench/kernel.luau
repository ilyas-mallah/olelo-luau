-- Times one kernel in one process: `lute run tools/bench/kernel.luau <kernel id> [rows] [cols] [tokens] [repeats]`.
-- It checks the first output against the reference, then times repeats of enough calls to take 0.2 s each, cycling
-- through copies of the weights so the matrix streams from memory like a model. Prints one `RES` line.
const data = require("@tests/data")
const process = require("@std/process")
const quant = require("@src/tensor/quant")
const registry = require("@src/kernels/registry")
const tensor = require("@src/tensor")

const WINDOW = 0.2
const STREAM_BYTES = 96 * 1024 * 1024

const id = process.args[2]
if id == nil then
    print("usage: lute run tools/bench/kernel.luau <kernel id> [rows] [cols] [tokens] [repeats]")
    process.exit(2)
end
const rows = tonumber(process.args[3]) or 1024
const cols = tonumber(process.args[4]) or 4096
const tokens = tonumber(process.args[5]) or 1
const repeats = tonumber(process.args[6]) or 15

const entry = registry.find(id)
const rng = data.newRng(20261122)
const quantized = entry.format == "q8" or entry.format == "q4"
const base = tensor.new({
    format = entry.format,
    rows = rows,
    cols = cols,
    rowTile = entry.rowTile,
    kstep = entry.kstep,
    layout = entry.layout,
})
assert(#base.parts == 1, "the benchmark matrix is one part")
local x: buffer
local xs: buffer? = nil
-- The reference reads int8 activations, a packed kernel reads their packed form.
local kernelX: buffer
local kernelXs: buffer? = nil
if quantized then
    data.fillQuant(base, rng, false)
    if entry.activation == "packed" then
        x, xs = data.activationsG(rng, tokens, cols, base.kpad, 1)
        kernelX, kernelXs = quant.packActivations(x, xs :: buffer, base.kpad, tokens)
    elseif entry.activation == "vsplat" then
        x, xs = data.activationsG(rng, tokens, cols, base.kpad, 1)
        kernelX, kernelXs = quant.splatActivations(x, xs :: buffer, base.kpad, tokens)
    elseif entry.activation == "int8g" then
        x, xs = data.activationsG(rng, tokens, cols, base.kpad, 1)
        kernelX, kernelXs = x, xs
    else
        x, xs = data.activationsQ(rng, tokens, cols, base.kpad, 1)
        kernelX, kernelXs = x, xs
    end
else
    data.fillFloat(base, rng, 1)
    x = data.activationsF32(rng, tokens, cols, base.kpad, 1)
    kernelX = x
end

-- Copies of the weights, so one call's matrix is cold when it comes round again.
const bytes = buffer.len(base.parts[1].buf)
    + (if base.parts[1].scales then buffer.len(base.parts[1].scales :: buffer) else 0)
const copies = math.max(1, math.min(16, (STREAM_BYTES + bytes - 1) // bytes))
const weights: { buffer } = { base.parts[1].buf }
const scales: { buffer? } = { base.parts[1].scales }
for c = 2, copies do
    const w = buffer.create(buffer.len(weights[1]))
    buffer.copy(w, 0, weights[1])
    weights[c] = w
    local s: buffer? = nil
    if scales[1] then
        s = buffer.create(buffer.len(scales[1] :: buffer))
        buffer.copy(s :: buffer, 0, scales[1] :: buffer)
    end
    scales[c] = s
end

const padded = base.rowsPadded
const y = buffer.create(tokens * padded * 4)
const yRef = buffer.create(tokens * padded * 4)
const kpad = base.kpad

--- One kernel call on copy `c` of the weights.
const function call(c: number)
    entry.fn(weights[c], 0, scales[c], 0, kernelX, kernelXs, y, 0, padded, 0, padded, kpad, tokens)
end

--- The checksum of the output buffer, as the C and Python benchmarks compute it.
const function checksum(out: buffer): number
    local sum = 0
    for n = 0, tokens * padded - 1 do
        sum += buffer.readf32(out, n * 4) * (n % 13 + 1)
    end
    return sum
end

call(1)
entry.reference(weights[1], 0, scales[1], 0, x, xs, yRef, 0, padded, 0, padded, kpad, tokens)
const csum = checksum(y)
const refSum = checksum(yRef)
const relative = math.abs(csum - refSum) / math.max(math.abs(refSum), 1e-30)

--- Seconds per call over `iterations` calls.
const function timed(iterations: number): number
    const start = os.clock()
    for n = 1, iterations do
        call(n % copies + 1)
    end
    return (os.clock() - start) / iterations
end

timed(1)
const single = timed(1)
const iterations = math.max(1, math.floor(WINDOW / math.max(single, 1e-6) + 0.5))
if iterations > 1 then
    timed(1)
end
const times: { number } = {}
for r = 1, repeats do
    times[r] = timed(iterations)
end
table.sort(times)
const median = times[(repeats + 1) // 2]
const macs = rows * cols * tokens
print(
    string.format(
        "RES %s M=%d K=%d T=%d csum=%.6g refdiff=%.2e med=%.6f min=%.6f max=%.6f iters=%d reps=%d copies=%d macs_per_s_med=%.4e macs_per_s_max=%.4e",
        id,
        rows,
        cols,
        tokens,
        csum,
        relative,
        median,
        times[1],
        times[repeats],
        iterations,
        repeats,
        copies,
        macs / median,
        macs / times[1]
    )
)
