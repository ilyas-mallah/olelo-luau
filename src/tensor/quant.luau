--!native
-- Activation quantization for the Q8 and Q4 kernels. The int8 kernels read int8 values with one f32 scale per token. The
-- packed Q8 kernel reads int8 values with a scale per group of 32, packed as f64 with the group sums that cancel its
-- offset binary weights.

const formats = require("./formats")

const GROUP = 32

--- Quantizes `tokens` rows of `k` f32 values (row stride `k`) to int8 with a scale per row of amax / 127. A zero row
--- gets scale 1. Returns the int8 buffer (tokens x k) and the f32 scales (tokens).
const function quantizeRows(x: buffer, k: number, tokens: number): (buffer, buffer)
    const q = buffer.create(tokens * k)
    const scales = buffer.create(tokens * 4)
    for t = 0, tokens - 1 do
        local amax = 0
        for j = 0, k - 1 do
            amax = math.max(amax, math.abs(buffer.readf32(x, (t * k + j) * 4)))
        end
        const scale = if amax == 0 then 1 else amax / 127
        buffer.writef32(scales, t * 4, scale)
        -- The kernel multiplies by the f32 scale, so the values are rounded against that same scale.
        const stored = buffer.readf32(scales, t * 4)
        for j = 0, k - 1 do
            const v = buffer.readf32(x, (t * k + j) * 4) / stored
            const r = math.clamp(if v >= 0 then math.floor(v + 0.5) else -math.floor(-v + 0.5), -127, 127)
            buffer.writei8(q, t * k + j, r)
        end
    end
    return q, scales
end

--- Quantizes `tokens` rows of `k` f32 values (row stride `k`, `k` a multiple of 32) the way Q8_0 quantizes weights and
--- llama.cpp quantizes activations: per group of 32 a scale d = amax / 127 and values round(x / d) taken with the f32
--- reciprocal of d, the scale stored in half precision. Returns the int8 buffer (tokens x k) and the f32 scales
--- (tokens x groups), a zero group having scale 0.
const function quantizeGroups(x: buffer, k: number, tokens: number): (buffer, buffer)
    assert(k % GROUP == 0, "group scales need whole groups")
    const groups = k // GROUP
    const q = buffer.create(tokens * k)
    const scales = buffer.create(tokens * groups * 4)
    for t = 0, tokens - 1 do
        for g = 0, groups - 1 do
            const first = t * k + g * GROUP
            local amax = 0
            for j = first, first + GROUP - 1 do
                amax = math.max(amax, math.abs(buffer.readf32(x, j * 4)))
            end
            const d = formats.roundF32(amax / 127)
            const id = if d ~= 0 then formats.roundF32(1 / d) else 0
            buffer.writef32(scales, (t * groups + g) * 4, formats.roundF16(d))
            for j = first, first + GROUP - 1 do
                const v = formats.roundF32(buffer.readf32(x, j * 4) * id)
                buffer.writei8(q, j, if v >= 0 then math.floor(v + 0.5) else -math.floor(-v + 0.5))
            end
        end
    end
    return q, scales
end

--- The packed form of group quantized activations, for kernels whose `activation` is `packed`. `q` and `scales` are what
--- `quantizeGroups` returns for `tokens` rows of `k` values. Returns the values as f64 and a buffer of, per token and
--- group of 32, the f32 128 times the sum of the group's values, which is exact, and the group's scale.
const function packActivations(q: buffer, scales: buffer, k: number, tokens: number): (buffer, buffer)
    assert(k % GROUP == 0, "packed activations need whole groups")
    const groups = k // GROUP
    const x = buffer.create(tokens * k * 8)
    const xs = buffer.create(tokens * groups * 8)
    for t = 0, tokens - 1 do
        for g = 0, groups - 1 do
            local sum = 0
            for j = 0, GROUP - 1 do
                const v = buffer.readi8(q, t * k + g * GROUP + j)
                buffer.writef64(x, (t * k + g * GROUP + j) * 8, v)
                sum += v
            end
            buffer.writef32(xs, (t * groups + g) * 8, sum * 128)
            buffer.writef32(xs, (t * groups + g) * 8 + 4, buffer.readf32(scales, (t * groups + g) * 4))
        end
    end
    return x, xs
end

--- Bytes of one group of `splatActivations` per token: the 12 byte offset vector and 11 steps of 3 vectors.
const SPLAT_GROUP_BYTES = 12 + 11 * 36

--- The splat form of group quantized activations, for kernels whose `activation` is `vsplat`. `q` and `scales` are what
--- `quantizeGroups` returns. Returns, per token and group of 32 with x0..x31 the values, 12 bytes of three times
--- -128 sum(x), then for step j = 0 to 10 the three vectors (256 xj, xj+11, xj+22 - 256 xj+11) each holding one number in
--- all 3 lanes (x32 is 0), and a buffer of the group scale as 3 equal lanes.
const function splatActivations(q: buffer, scales: buffer, k: number, tokens: number): (buffer, buffer)
    assert(k % GROUP == 0, "splat activations need whole groups")
    const groups = k // GROUP
    const x = buffer.create(tokens * groups * SPLAT_GROUP_BYTES)
    const xs = buffer.create(tokens * groups * 12)
    for t = 0, tokens - 1 do
        for g = 0, groups - 1 do
            const first = t * k + g * GROUP
            const o = (t * groups + g) * SPLAT_GROUP_BYTES
            local sum = 0
            for j = 0, GROUP - 1 do
                sum += buffer.readi8(q, first + j)
            end
            for l = 0, 2 do
                buffer.writef32(x, o + l * 4, -128 * sum)
                buffer.writef32(xs, (t * groups + g) * 12 + l * 4, buffer.readf32(scales, (t * groups + g) * 4))
            end
            for j = 0, 10 do
                const xa = buffer.readi8(q, first + j)
                const xb = buffer.readi8(q, first + j + 11)
                const xc = if j <= 9 then buffer.readi8(q, first + j + 22) else 0
                for l = 0, 2 do
                    buffer.writef32(x, o + 12 + j * 36 + l * 4, 256 * xa)
                    buffer.writef32(x, o + 12 + j * 36 + 12 + l * 4, xb)
                    buffer.writef32(x, o + 12 + j * 36 + 24 + l * 4, xc - 256 * xb)
                end
            end
        end
    end
    return x, xs
end

return {
    SPLAT_GROUP_BYTES = SPLAT_GROUP_BYTES,
    splatActivations = splatActivations,
    quantizeRows = quantizeRows,
    quantizeGroups = quantizeGroups,
    packActivations = packActivations,
}
