-- Preparing activations writes exactly what the plain reference conversions produce, for every kernel form, rows that
-- are not whole groups and a zero group, and leaves f32 activations alone.
const data = require("@tests/data")
const harness = require("@tests/harness")
const prepare = require("@src/ops/prepare")
const quant = require("@src/tensor/quant")
const registry = require("@src/kernels/registry")

const function sameBytes(a: buffer, b: buffer, length: number): boolean
    return buffer.readstring(a, 0, length) == buffer.readstring(b, 0, length)
end

return function(h: harness.Harness)
    -- 3 tokens of 70 values read from rows 72 floats apart, stored 96 wide.
    const tokens, cols, stride, kpad = 3, 70, 72, 96
    const rng = data.newRng(99)
    const src = data.activationsF32(rng, tokens, cols, stride, 5)
    for j = 32, 63 do
        buffer.writef32(src, (stride + j) * 4, 0)
    end
    -- The reference reads rows of kpad values with zeros past the columns.
    const wide = buffer.create(tokens * kpad * 4)
    for t = 0, tokens - 1 do
        buffer.copy(wide, t * kpad * 4, src, t * stride * 4, cols * 4)
    end

    h.case("the packed form equals the reference conversion", function()
        const entry = registry.find("q8_p8/R8S1U32")
        const scratch = prepare.new(4, kpad)
        const x, xs = prepare.run(entry, scratch, src, stride, cols, kpad, tokens)
        const q, scales = quant.quantizeGroups(wide, kpad, tokens)
        const wantX, wantXs = quant.packActivations(q, scales, kpad, tokens)
        h.truthy(sameBytes(x, wantX, tokens * kpad * 8), "packed values differ")
        h.truthy(sameBytes(xs :: buffer, wantXs, tokens * (kpad // 32) * 8), "group sums or scales differ")
        -- The zeroed group of token 1 has scale 0 and sum 0.
        h.equal(buffer.readf32(xs :: buffer, ((1 * 3) + 1) * 8), 0)
        h.equal(buffer.readf32(xs :: buffer, ((1 * 3) + 1) * 8 + 4), 0)
    end)

    if (buffer :: any).writevectorf32x3 == nil then
        h.skip("the splat form", "buffer.writevectorf32x3 does not exist in this mode")
    else
        h.case("the splat form equals the reference conversion", function()
            const entry = registry.find("q8_v12/R12S1U32")
            const scratch = prepare.new(4, kpad)
            const x, xs = prepare.run(entry, scratch, src, stride, cols, kpad, tokens)
            const q, scales = quant.quantizeGroups(wide, kpad, tokens)
            const wantX, wantXs = quant.splatActivations(q, scales, kpad, tokens)
            h.truthy(sameBytes(x, wantX, tokens * (kpad // 32) * quant.SPLAT_GROUP_BYTES), "splat vectors differ")
            h.truthy(sameBytes(xs :: buffer, wantXs, tokens * (kpad // 32) * 12), "group scales differ")
        end)
    end

    h.case("the int8 form equals the reference conversion", function()
        const entry = registry.find("q8_il4e/R8S1U8")
        const scratch = prepare.new(4, kpad)
        const x, xs = prepare.run(entry, scratch, src, stride, cols, kpad, tokens)
        const q, scales = quant.quantizeRows(wide, kpad, tokens)
        h.truthy(sameBytes(x, q, tokens * kpad), "int8 values differ")
        h.truthy(sameBytes(xs :: buffer, scales, tokens * 4), "token scales differ")
    end)

    h.case("the int8 form with group scales equals the reference conversion", function()
        const entry = registry.find("q4_byte/R8S1U4")
        const scratch = prepare.new(4, kpad)
        const x, xs = prepare.run(entry, scratch, src, stride, cols, kpad, tokens)
        const q, scales = quant.quantizeGroups(wide, kpad, tokens)
        h.truthy(sameBytes(x, q, tokens * kpad), "int8 values differ")
        h.truthy(sameBytes(xs :: buffer, scales, tokens * (kpad // 32) * 4), "group scales differ")
    end)

    h.case("an f32 kernel reads the rows as they are", function()
        const entry = registry.find("f32_scalar/R8S1U8")
        const x, xs = prepare.run(entry, prepare.new(1, 32), src, 96, 70, 96, 1)
        h.truthy(x == src and xs == nil)
        h.fails(function()
            prepare.run(entry, prepare.new(1, 32), src, 72, 70, 96, 1)
        end)
    end)

    h.case("rows wider than the scratch are refused", function()
        h.fails(function()
            prepare.run(registry.find("q8_p8/R8S1U32"), prepare.new(1, 64), src, stride, cols, kpad, tokens)
        end)
    end)
end
