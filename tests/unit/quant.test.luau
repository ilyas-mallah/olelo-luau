-- Activation quantization gives int8 values within -127 to 127 whose product with the scale is within half a step of
-- the input, and a zero row gets scale 1.
const harness = require("@tests/harness")
const quant = require("@src/tensor/quant")

return function(h: harness.Harness)
    h.case("values round to the nearest step of the per token scale", function()
        const k = 40
        const x = buffer.create(2 * k * 4)
        for j = 0, k - 1 do
            buffer.writef32(x, j * 4, math.sin(j) * 3)
            buffer.writef32(x, (k + j) * 4, if j == 5 then 100 else -j / 7)
        end
        const q, scales = quant.quantizeRows(x, k, 2)
        for t = 0, 1 do
            const scale = buffer.readf32(scales, t * 4)
            for j = 0, k - 1 do
                const v = buffer.readi8(q, t * k + j)
                h.truthy(v >= -127 and v <= 127)
                h.near(v * scale, buffer.readf32(x, (t * k + j) * 4), scale / 2 + 1e-6, `token {t} k {j}`)
            end
        end
        h.near(buffer.readi8(q, k + 5), 127, 0)
    end)

    h.case(
        "group quantization follows Q8_0: a half precision scale per 32 and values taken with its reciprocal",
        function()
            const x = buffer.create(64 * 4)
            -- Group 0: amax 63.5 gives d = 0.5, so the values are twice the inputs and a quarter rounds away from zero.
            buffer.writef32(x, 0, 63.5)
            buffer.writef32(x, 4, -31.75)
            buffer.writef32(x, 8, 0.25)
            buffer.writef32(x, 12, -0.25)
            -- Group 1 is all zero.
            const q, scales = quant.quantizeGroups(x, 64, 1)
            h.equal(buffer.readf32(scales, 0), 0.5)
            h.equal(buffer.readi8(q, 0), 127)
            h.equal(buffer.readi8(q, 1), -64)
            h.equal(buffer.readi8(q, 2), 1)
            h.equal(buffer.readi8(q, 3), -1)
            h.equal(buffer.readf32(scales, 4), 0)
            h.equal(buffer.readi8(q, 40), 0)
        end
    )

    h.case("the packed form carries 128 times the group sum and the scale", function()
        const q = buffer.create(32)
        for j = 0, 31 do
            buffer.writei8(q, j, j - 10)
        end
        const scales = buffer.create(4)
        buffer.writef32(scales, 0, 0.25)
        const x, xs = quant.packActivations(q, scales, 32, 1)
        h.equal(buffer.readf64(x, 5 * 8), -5)
        h.equal(buffer.readf32(xs, 0), 128 * (496 - 320))
        h.equal(buffer.readf32(xs, 4), 0.25)
    end)

    h.case("a zero row gets scale 1 and zero values", function()
        const x = buffer.create(16 * 4)
        const q, scales = quant.quantizeRows(x, 16, 1)
        h.equal(buffer.readf32(scales, 0), 1)
        h.equal(buffer.readi8(q, 3), 0)
    end)
end
