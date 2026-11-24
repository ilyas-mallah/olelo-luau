-- Shared kernel checks: runs a generated kernel and its reference on the same random tensor and compares them.
-- Float kernels must agree within K x 2^-24 x the sum of |w x| per output. The packed Q8 and Q4 kernels must agree
-- exactly, because their integer sums are exact in doubles and the scales are applied in the same order. The vector Q8
-- kernel sums its groups in f32, so it agrees within 3 x groups x 2^-24 x the sum of |scales x integer products|, and
-- exactly where the scales are 1.

const data = require("@tests/data")
const formats = require("@src/tensor/formats")
const harness = require("@tests/harness")
const quant = require("@src/tensor/quant")
const registry = require("@src/kernels/registry")
const tensor = require("@src/tensor")
const kernelsTypes = require("@src/kernels/types")
const tensorTypes = require("@src/tensor/types")

--- One shape and value range to check a kernel on.
export type Case = {
    rows: number,
    cols: number,
    --- Tokens, a multiple of the kernel's tile tokens.
    tokens: number,
    --- Magnitude of the f32 weights and activations.
    scale: number,
    --- Cut the tensor into parts of two tiles of rows, so the kernel runs once per part.
    parts: boolean?,
    --- Run only the tile rows from `first` for `count` tiles of the first part, the rest must stay untouched.
    range: { first: number, count: number }?,
    --- Quantized formats only: every scale is 1 and the output is compared with an independent integer sum.
    unit: boolean?,
    --- Packed kernels only, implies `unit`: weights of all 127, all -128 and alternating, activations of all 127, all -127 and
    --- alternating, the worst case of the packed kernel's fields.
    extreme: boolean?,
}

const SENTINEL = 0xAA
const Y_BASE = 16
const EPS = 2 ^ -24

--- The cases a kernel of this tile runs, around its padding step and across value ranges.
const function casesFor(entry: kernelsTypes.Entry): { Case }
    const R, S, k = entry.R, entry.S, entry.kstep
    const quantized = entry.format == "q8" or entry.format == "q4"
    const cases: { Case } = {
        { rows = 1, cols = 1, tokens = S, scale = 1 },
        { rows = R + 1, cols = k - 1, tokens = S, scale = 1 },
        { rows = 37, cols = k, tokens = S, scale = 1 },
        { rows = R, cols = k + 1, tokens = 3 * S, scale = 1 },
        { rows = 3 * R - 2, cols = 2 * k + 1, tokens = S, scale = 1 },
        { rows = 64, cols = 100, tokens = 2 * S, scale = 1, parts = true },
        { rows = 5 * R, cols = 3 * k, tokens = S, scale = 1, range = { first = 1, count = 2 } },
        { rows = 2 * R, cols = 5 * k - 2, tokens = S, scale = 0 },
    }
    if quantized then
        table.insert(cases, { rows = 3 * R, cols = 3 * k, tokens = 2 * S, scale = 1, unit = true })
        table.insert(cases, { rows = 2 * R, cols = 2 * k, tokens = S, scale = 1e6 })
        if entry.activation == "packed" or entry.activation == "vsplat" then
            table.insert(cases, { rows = 2 * R, cols = 4 * k, tokens = 3 * S, scale = 1, unit = true, extreme = true })
        end
        if entry.activation == "packed" then
            table.insert(
                cases,
                { rows = 3 * R, cols = 128 * k, tokens = 4 * S, scale = 1, unit = true, extreme = true }
            )
        end
    else
        table.insert(cases, { rows = 2 * R, cols = 2 * k + 5, tokens = S, scale = 1e-6 })
        table.insert(cases, { rows = 2 * R, cols = 2 * k + 5, tokens = S, scale = 1e6 })
    end
    return cases
end

--- The slot of a float in the output buffer, as bytes.
const function slot(row: number, token: number, stride: number): number
    return Y_BASE + (token * stride + row) * 4
end

--- Runs `entry` and its reference on `case` and returns a description of the first mismatch, nil when they agree.
const function runCase(entry: kernelsTypes.Entry, case: Case, seed: number): string?
    const rng = data.newRng(seed)
    const quantized = entry.format == "q8" or entry.format == "q4"
    const spec: tensorTypes.Spec = {
        format = entry.format,
        rows = case.rows,
        cols = case.cols,
        rowTile = entry.rowTile,
        kstep = entry.kstep,
        layout = entry.layout,
        maxPartBytes = nil,
    }
    if case.parts then
        spec.maxPartBytes = 2
            * entry.rowTile
            * formats.rowBytes(entry.format, (case.cols + entry.kstep - 1) // entry.kstep * entry.kstep)
    end
    const t = tensor.new(spec)
    if case.parts and #t.parts < 2 then
        return "the multi part case made one part"
    end

    local x: buffer
    local xs: buffer? = nil
    if quantized then
        if case.extreme then
            data.fillExtremeQuant(t, rng)
        else
            data.fillQuant(t, rng, case.unit == true)
        end
        -- The packed and the Q4 kernel read activations with a scale per group, the others with one per token.
        const perGroup = entry.activation == "packed" or entry.activation == "int8g" or entry.activation == "vsplat"
        if case.extreme then
            x, xs = data.activationsExtreme(rng, case.tokens, case.cols, t.kpad, perGroup)
        elseif case.unit then
            x, xs = data.activationsUnit(rng, case.tokens, case.cols, t.kpad, perGroup)
        elseif perGroup then
            x, xs = data.activationsG(rng, case.tokens, case.cols, t.kpad, case.scale)
        else
            x, xs = data.activationsQ(rng, case.tokens, case.cols, t.kpad, case.scale)
        end
    else
        data.fillFloat(t, rng, case.scale)
        x = data.activationsF32(rng, case.tokens, case.cols, t.kpad, case.scale)
    end

    -- The reference reads int8 activations, a packed kernel reads their packed form.
    local kernelX, kernelXs = x, xs
    if entry.activation == "packed" then
        kernelX, kernelXs = quant.packActivations(x, xs :: buffer, t.kpad, case.tokens)
    elseif entry.activation == "vsplat" then
        kernelX, kernelXs = quant.splatActivations(x, xs :: buffer, t.kpad, case.tokens)
    end

    const stride = t.rowsPadded + 3
    const size = Y_BASE + case.tokens * stride * 4 + 16
    const yKernel = buffer.create(size)
    const yReference = buffer.create(size)
    buffer.fill(yKernel, 0, SENTINEL)
    buffer.fill(yReference, 0, SENTINEL)

    const written: { [number]: boolean } = {}
    for _, part in t.parts do
        local first, count = 0, part.rowCount
        if case.range and part == t.parts[1] then
            first = case.range.first * entry.R
            count = case.range.count * entry.R
        end
        const yBase = Y_BASE + part.rowStart * 4
        entry.fn(
            part.buf,
            part.offset,
            part.scales,
            part.scaleOffset,
            kernelX,
            kernelXs,
            yKernel,
            yBase,
            stride,
            first,
            count,
            t.kpad,
            case.tokens
        )
        entry.reference(
            part.buf,
            part.offset,
            part.scales,
            part.scaleOffset,
            x,
            xs,
            yReference,
            yBase,
            stride,
            first,
            count,
            t.kpad,
            case.tokens
        )
        for token = 0, case.tokens - 1 do
            for row = first, first + count - 1 do
                written[slot(part.rowStart + row, token, stride)] = true
            end
        end
    end

    for token = 0, case.tokens - 1 do
        for row = 0, t.rowsPadded - 1 do
            const at = slot(row, token, stride)
            const got = buffer.readf32(yKernel, at)
            const want = buffer.readf32(yReference, at)
            if not written[at] then
                if buffer.readu32(yKernel, at) ~= 0xAAAAAAAA or buffer.readu32(yReference, at) ~= 0xAAAAAAAA then
                    return `row {row} token {token} was written outside the requested rows`
                end
            else
                local tolerance = 0
                if not quantized then
                    local bound = 0
                    for k = 0, case.cols - 1 do
                        bound += math.abs(tensor.getFloat(t, row, k) * buffer.readf32(x, (token * t.kpad + k) * 4))
                    end
                    tolerance = (case.cols + 8) * EPS * bound
                elseif entry.activation == "vsplat" and not case.unit then
                    const groups = t.kpad // formats.GROUP
                    local bound = 0
                    for g = 0, groups - 1 do
                        local magnitude = 0
                        for j = 0, formats.GROUP - 1 do
                            const k = g * formats.GROUP + j
                            if k < case.cols then
                                magnitude += math.abs(tensor.getQuant(t, row, k)) * math.abs(
                                    buffer.readi8(x, token * t.kpad + k)
                                )
                            end
                        end
                        bound += math.abs(
                            tensor.getScale(t, row, g) * buffer.readf32(xs :: buffer, (token * groups + g) * 4)
                        ) * magnitude
                    end
                    tolerance = (3 * groups + 8) * EPS * bound
                end
                if not (math.abs(got - want) <= tolerance) then
                    return `row {row} token {token}: kernel {got}, reference {want}, tolerance {tolerance}`
                end
            end
        end
    end
    -- The checks above compare against the reference. A unit scale case also checks the integer sum itself.
    if case.unit then
        for token = 0, case.tokens - 1 do
            for row = 0, t.rows - 1 do
                local sum = 0
                for k = 0, case.cols - 1 do
                    local w = tensor.getQuant(t, row, k)
                    if entry.format == "q4" then
                        w -= 8
                    end
                    sum += w * buffer.readi8(x, token * t.kpad + k)
                end
                const got = buffer.readf32(yKernel, slot(row, token, stride))
                if got ~= sum then
                    return `unit scale row {row} token {token}: kernel {got}, integer sum {sum}`
                end
            end
        end
    end
    return nil
end

--- Registers one case per shape for `entry` on the harness.
const function checkEntry(h: harness.Harness, entry: kernelsTypes.Entry)
    for i, case in casesFor(entry) do
        const name = `{entry.id} rows {case.rows} cols {case.cols} tokens {case.tokens} scale {case.scale}`
            .. (if case.parts then " parts" else "")
            .. (if case.range then " range" else "")
            .. (if case.unit then " unit" else "")
        h.case(name, function()
            const problem = runCase(entry, case, 1000 + i)
            if problem then
                error(problem)
            end
        end)
    end
end

--- A suite that checks every kernel of the generated modules in `modules`. Vector kernels are skipped, with the
--- reason, on a host that has no `buffer.readvectorf32x3`.
const function suiteFor(modules: { string }): harness.Suite
    return function(h: harness.Harness)
        const vectorAvailable = (buffer :: any).readvectorf32x3 ~= nil
        local found = 0
        for _, entry in registry.all() do
            if table.find(modules, entry.module) then
                found += 1
                if entry.needsVectorRead and not vectorAvailable then
                    h.skip(entry.id, "buffer.readvectorf32x3 does not exist in this mode")
                else
                    checkEntry(h, entry)
                end
            end
        end
        h.case("the registry lists the modules", function()
            h.truthy(found >= #modules, "a module has no kernel in the registry")
        end)
    end
end

return {
    casesFor = casesFor,
    runCase = runCase,
    checkEntry = checkEntry,
    suiteFor = suiteFor,
}
