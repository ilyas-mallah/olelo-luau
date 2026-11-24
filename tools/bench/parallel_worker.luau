--!native
-- The worker VM of `parallel.luau`: answers messages of a given size, and runs a kernel over weights of its own, which
-- is what one worker of tensor parallel inference does. Every function is a call from the main VM.
const data = require("@tests/data")
const probe = require("@src/kernels/probe")
const quant = require("@src/tensor/quant")
const registry = require("@src/kernels/registry")
const tensor = require("@src/tensor")
const kernelsTypes = require("@src/kernels/types")

local entry: kernelsTypes.Entry? = nil
local weights: { buffer } = {}
local scales: { buffer? } = {}
local x: buffer = buffer.create(0)
local xs: buffer? = nil
local y: buffer = buffer.create(0)
local rowsPadded = 0
local kpad = 0
local tokens = 1

return {
    nop = function(): number
        return 0
    end,
    echo = function(b: buffer): buffer
        return b
    end,
    --- Whether this VM's probe finds the vector read fast, and the rate of the f32 vector kernel there in MAC/s.
    vector = function(): boolean
        return probe.run(os.clock).vectorRead
    end,
    --- Builds `copies` copies of a random `rows` x `cols` matrix for the kernel `id`. Returns the bytes of one copy.
    setup = function(id: string, rows: number, cols: number, copyCount: number, tokenCount: number): number
        const e = registry.find(id)
        entry = e
        tokens = tokenCount
        const rng = data.newRng(20261123)
        const base = tensor.new({
            format = e.format,
            rows = rows,
            cols = cols,
            rowTile = e.rowTile,
            kstep = e.kstep,
            layout = e.layout,
        })
        if e.format == "q8" or e.format == "q4" then
            data.fillQuant(base, rng, false)
            if e.activation == "packed" then
                const a, s = data.activationsG(rng, tokens, cols, base.kpad, 1)
                x, xs = quant.packActivations(a, s, base.kpad, tokens)
            elseif e.activation == "int8g" then
                x, xs = data.activationsG(rng, tokens, cols, base.kpad, 1)
            else
                x, xs = data.activationsQ(rng, tokens, cols, base.kpad, 1)
            end
        else
            data.fillFloat(base, rng, 1)
            x = data.activationsF32(rng, tokens, cols, base.kpad, 1)
        end
        const part = base.parts[1]
        weights, scales = { part.buf }, { part.scales }
        for c = 2, copyCount do
            const w = buffer.create(buffer.len(part.buf))
            buffer.copy(w, 0, part.buf)
            weights[c] = w
            if part.scales ~= nil then
                const s = buffer.create(buffer.len(part.scales :: buffer))
                buffer.copy(s, 0, part.scales :: buffer)
                scales[c] = s
            end
        end
        rowsPadded, kpad = base.rowsPadded, base.kpad
        y = buffer.create(tokens * rowsPadded * 4)
        return buffer.len(part.buf) + (if part.scales ~= nil then buffer.len(part.scales :: buffer) else 0)
    end,
    --- Runs `calls` kernel calls over the copies in turn and returns the seconds they took in this VM.
    run = function(calls: number): number
        const e = entry :: kernelsTypes.Entry
        const copies = #weights
        const start = os.clock()
        for n = 1, calls do
            const c = n % copies + 1
            e.fn(weights[c], 0, scales[c], 0, x, xs, y, 0, rowsPadded, 0, rowsPadded, kpad, tokens)
        end
        return os.clock() - start
    end,
}
