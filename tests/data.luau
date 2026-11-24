-- Deterministic test data: a generator and builders for filled tensors and activations.

const probe = require("@src/kernels/probe")
const quant = require("@src/tensor/quant")
const tensor = require("@src/tensor")
const tensorTypes = require("@src/tensor/types")

local vectorFastResult: boolean? = nil

--- True when the probe finds the fast vector read, which a model with Q8 matrices in the vector layout needs.
const function vectorFast(): boolean
    if vectorFastResult == nil then
        vectorFastResult = probe.run(os.clock).vectorRead
    end
    return vectorFastResult :: boolean
end

--- A linear congruential generator, the same one the kernel benchmarks use.
export type Rng = {
    state: number,
}

--- A generator started at `seed`.
const function newRng(seed: number): Rng
    return { state = seed }
end

--- An integer from 0 to `n - 1`.
const function nextInt(rng: Rng, n: number): number
    rng.state = (rng.state * 16807) % 2147483647
    return (rng.state // 65536) % n
end

--- A number in [-scale, scale] on a grid of 1/1000 of it.
const function nextFloat(rng: Rng, scale: number): number
    return (nextInt(rng, 2001) - 1000) / 1000 * scale
end

--- Fills the rows and columns the model has of an f32 or BF16 tensor, leaving the padding zero.
const function fillFloat(t: tensorTypes.Tensor, rng: Rng, scale: number)
    for row = 0, t.rows - 1 do
        for k = 0, t.cols - 1 do
            tensor.setFloat(t, row, k, nextFloat(rng, scale))
        end
    end
end

--- Fills a Q8 or Q4 tensor with random quantized weights and scales. With `unit` every scale is 1.
const function fillQuant(t: tensorTypes.Tensor, rng: Rng, unit: boolean)
    for row = 0, t.rowsPadded - 1 do
        for group = 0, t.kpad // 32 - 1 do
            tensor.setScale(t, row, group, if unit then 1 else (1 + nextInt(rng, 8)) / 64)
        end
    end
    for row = 0, t.rows - 1 do
        for k = 0, t.cols - 1 do
            tensor.setQuant(t, row, k, if t.format == "q8" then nextInt(rng, 256) - 128 else nextInt(rng, 16))
        end
    end
end

--- `tokens` rows of `kpad` f32 activations, the first `cols` of each random and the rest zero.
const function activationsF32(rng: Rng, tokens: number, cols: number, kpad: number, scale: number): buffer
    const x = buffer.create(tokens * kpad * 4)
    for t = 0, tokens - 1 do
        for k = 0, cols - 1 do
            buffer.writef32(x, (t * kpad + k) * 4, nextFloat(rng, scale))
        end
    end
    return x
end

--- Int8 activations quantized from random f32 ones, with their per token scales.
const function activationsQ(rng: Rng, tokens: number, cols: number, kpad: number, scale: number): (buffer, buffer)
    return quant.quantizeRows(activationsF32(rng, tokens, cols, kpad, scale), kpad, tokens)
end

--- Int8 activations quantized per group of 32 from random f32 ones, with their group scales.
const function activationsG(rng: Rng, tokens: number, cols: number, kpad: number, scale: number): (buffer, buffer)
    return quant.quantizeGroups(activationsF32(rng, tokens, cols, kpad, scale), kpad, tokens)
end

--- A buffer of `tokens` scales of 1, or `tokens x kpad / 32` when `perGroup`.
const function unitScales(tokens: number, kpad: number, perGroup: boolean?): buffer
    const count = if perGroup then tokens * (kpad // 32) else tokens
    const scales = buffer.create(count * 4)
    for i = 0, count - 1 do
        buffer.writef32(scales, i * 4, 1)
    end
    return scales
end

--- Int8 activations drawn directly in -127 to 127, with every scale 1, one per token or one per group.
const function activationsUnit(
    rng: Rng,
    tokens: number,
    cols: number,
    kpad: number,
    perGroup: boolean?
): (buffer, buffer)
    const x = buffer.create(tokens * kpad)
    const scales = unitScales(tokens, kpad, perGroup)
    for t = 0, tokens - 1 do
        for k = 0, cols - 1 do
            buffer.writei8(x, t * kpad + k, nextInt(rng, 255) - 127)
        end
    end
    return x, scales
end

--- Fills a Q8 or Q4 tensor with the weights that stress the packed kernel: rows of the largest weight, of the smallest,
--- alternating and random ones, every scale 1.
const function fillExtremeQuant(t: tensorTypes.Tensor, rng: Rng)
    for row = 0, t.rowsPadded - 1 do
        for group = 0, t.kpad // 32 - 1 do
            tensor.setScale(t, row, group, 1)
        end
    end
    for row = 0, t.rows - 1 do
        for k = 0, t.cols - 1 do
            const kind = row % 4
            const wide = t.format == "q8"
            const high, low = if wide then 127 else 15, if wide then -128 else 0
            local q = low + nextInt(rng, high - low + 1)
            if kind == 0 then
                q = high
            elseif kind == 1 then
                q = low
            elseif kind == 2 then
                q = if k % 2 == 0 then high else low
            end
            tensor.setQuant(t, row, k, q)
        end
    end
end

--- Int8 activations at the extremes with every scale 1: token 0 all 127, token 1 all -127, token 2 alternating,
--- later tokens random in -127 to 127.
const function activationsExtreme(
    rng: Rng,
    tokens: number,
    cols: number,
    kpad: number,
    perGroup: boolean?
): (buffer, buffer)
    const x = buffer.create(tokens * kpad)
    const scales = unitScales(tokens, kpad, perGroup)
    for t = 0, tokens - 1 do
        for k = 0, cols - 1 do
            local v = nextInt(rng, 255) - 127
            if t == 0 then
                v = 127
            elseif t == 1 then
                v = -127
            elseif t == 2 then
                v = if k % 2 == 0 then 127 else -127
            end
            buffer.writei8(x, t * kpad + k, v)
        end
    end
    return x, scales
end

return {
    vectorFast = vectorFast,
    newRng = newRng,
    nextInt = nextInt,
    nextFloat = nextFloat,
    fillFloat = fillFloat,
    fillQuant = fillQuant,
    activationsF32 = activationsF32,
    activationsQ = activationsQ,
    activationsG = activationsG,
    activationsUnit = activationsUnit,
    fillExtremeQuant = fillExtremeQuant,
    activationsExtreme = activationsExtreme,
}
