--!nonstrict
--!native
-- The splat form of group quantized activations for the vector Q8 kernel, `quant.splatActivations` computed straight from
-- the f32 rows. Strict typing is off because the type checker does not know `buffer.writevectorf32x3`. Only call it
-- where the probe finds the fast call.

const formats = require("@src/tensor/formats")

const GROUP = formats.GROUP

--- Bytes of one group of one token in the splat form.
const X_GROUP = 12 + 11 * 36

--- A word the activations round through to f32.
const rounder = buffer.create(8)

--- Quantizes `tokens` rows of `cols` f32 values (row stride `srcStride` floats) per group of 32 as
--- `quant.quantizeGroups` does and writes the splat form into `x` and the group scales as 3 equal lanes into `xs`. The
--- int8 values pass through `q`, `tokens` rows of `kpad`. Values past `cols` are 0.
const function splat(
    x: buffer,
    xs: buffer,
    q: buffer,
    src: buffer,
    srcStride: number,
    cols: number,
    kpad: number,
    tokens: number
)
    const groups = kpad // GROUP
    for t = 0, tokens - 1 do
        const rowAt = t * srcStride * 4
        for g = 0, groups - 1 do
            const first = g * GROUP
            const used = math.max(math.min(cols - first, GROUP), 0)
            const at = rowAt + first * 4
            local amax = 0
            for j = 0, used - 1 do
                amax = math.max(amax, math.abs(buffer.readf32(src, at + j * 4)))
            end
            const d = formats.roundF32(amax / 127)
            const id = if d ~= 0 then formats.roundF32(1 / d) else 0
            const qb = t * kpad + first
            local sum = 0
            -- The product goes through a scratch word to round to f32, and math.round rounds half away from zero like
            -- the reference. Adding 0 turns a rounded -0 into +0.
            for j = 0, used - 1 do
                buffer.writef32(rounder, 0, buffer.readf32(src, at + j * 4) * id)
                const r = math.round(buffer.readf32(rounder, 0)) + 0
                buffer.writei8(q, qb + j, r)
                sum += r
            end
            for j = used, GROUP - 1 do
                buffer.writei8(q, qb + j, 0)
            end
            const o = (t * groups + g) * X_GROUP
            const init = -128 * sum
            buffer.writevectorf32x3(x, o, vector.create(init, init, init))
            for j = 0, 9 do
                const xa = buffer.readi8(q, qb + j)
                const xb = buffer.readi8(q, qb + j + 11)
                const xc = buffer.readi8(q, qb + j + 22)
                const at3 = o + 12 + j * 36
                buffer.writevectorf32x3(x, at3, vector.create(256 * xa, 256 * xa, 256 * xa))
                buffer.writevectorf32x3(x, at3 + 12, vector.create(xb, xb, xb))
                const z = xc - 256 * xb
                buffer.writevectorf32x3(x, at3 + 24, vector.create(z, z, z))
            end
            do
                const xa = buffer.readi8(q, qb + 10)
                const xb = buffer.readi8(q, qb + 21)
                const at3 = o + 12 + 10 * 36
                buffer.writevectorf32x3(x, at3, vector.create(256 * xa, 256 * xa, 256 * xa))
                buffer.writevectorf32x3(x, at3 + 12, vector.create(xb, xb, xb))
                const z = 0 - 256 * xb
                buffer.writevectorf32x3(x, at3 + 24, vector.create(z, z, z))
            end
            const scale = formats.roundF16(d)
            buffer.writevectorf32x3(xs, (t * groups + g) * 12, vector.create(scale, scale, scale))
        end
    end
end

return {
    X_GROUP = X_GROUP,
    splat = splat,
}
