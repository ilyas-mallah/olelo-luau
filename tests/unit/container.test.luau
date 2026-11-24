const container = require("@src/container")
const harness = require("@tests/harness")
const host = require("@src/host")
const lute = require("@backends/lute")
const tensor = require("@src/tensor")

--- A weight source over a buffer.
const function memory(bytes: buffer): host.WeightSource
    return {
        size = function(): number
            return buffer.len(bytes)
        end,
        read = function(offset: number, length: number, dst: buffer, dstOffset: number)
            buffer.copy(dst, dstOffset, bytes, offset, length)
        end,
    }
end

const function expectError(h: harness.Harness, body: () -> (), needle: string)
    const ok, err = pcall(body)
    h.truthy(
        not ok and string.find(tostring(err), needle, 1, true) ~= nil,
        `expected an error with "{needle}", got {tostring(err)}`
    )
end

return function(h: harness.Harness)
    const c = container.open(lute.fileSource("tests/fixtures/tiny-llama/model.olelo"))

    h.case("the header carries the model description", function()
        h.equal(c.header.arch, "llama")
        h.equal(c.header.config.hidden, 60)
        h.equal(c.header.config.layers, 2)
        h.equal(c.dataStart % 64, 0)
    end)

    h.case("a matrix loads padded to the kernel tile", function()
        const t = container.loadTensor(c, "layers.0.q")
        h.equal(t.rows, 60)
        h.equal(t.rowsPadded, 64)
        h.equal(t.cols, 60)
        h.equal(t.kpad, 72)
        h.equal(#t.parts, 1)
        h.equal(buffer.len(t.parts[1].buf), 64 * 72 * 4)
        -- The padding is zero.
        h.equal(buffer.readf32(t.parts[1].buf, (0 * 72 + 60) * 4), 0)
        h.equal(buffer.readf32(t.parts[1].buf, (63 * 72) * 4), 0)
    end)

    h.case("a Q8 matrix loads with its scales and holds the f32 values to half a step", function()
        const q8 = container.open(lute.fileSource("tests/fixtures/tiny-llama/model-q8.olelo"))
        const quantized = container.loadTensor(q8, "layers.0.q")
        const exact = container.loadTensor(c, "layers.0.q")
        h.equal(quantized.format, "q8")
        h.equal(quantized.layout, "p8")
        h.equal(quantized.rowsPadded, 64)
        h.equal(quantized.kpad, 64)
        h.equal(buffer.len(quantized.parts[1].buf), 64 * 64)
        h.equal(buffer.len(quantized.parts[1].scales :: buffer), 64 * 2 * 4)
        for row = 0, 59 do
            for k = 0, 59 do
                const scale = tensor.getScale(quantized, row, k // 32)
                const value = tensor.getQuant(quantized, row, k) * scale
                h.truthy(
                    math.abs(value - tensor.getFloat(exact, row, k)) <= 0.6 * scale + 1e-9,
                    `row {row} k {k}: {value} against {tensor.getFloat(exact, row, k)}`
                )
            end
        end
        -- The padding rows and columns are the weight zero.
        h.equal(tensor.getQuant(quantized, 63, 3), 0)
        h.equal(tensor.getQuant(quantized, 0, 63), 0)
    end)

    h.case("a vector layout Q8 matrix holds the weights and scales of the packed one", function()
        const p8 = container.open(lute.fileSource("tests/fixtures/tiny-llama/model-q8.olelo"))
        const v12 = container.open(lute.fileSource("tests/fixtures/tiny-llama/model-q8v.olelo"))
        for _, name in { "layers.0.q", "layers.1.down", "embed" } do
            const a = container.loadTensor(p8, name)
            const b = container.loadTensor(v12, name)
            h.equal(b.layout, "v12", name)
            h.equal(b.rowTile, 12, name)
            h.equal(b.rows, a.rows, name)
            h.equal(b.kpad, a.kpad, name)
            for row = 0, a.rows - 1 do
                for k = 0, a.cols - 1 do
                    h.equal(tensor.getQuant(b, row, k), tensor.getQuant(a, row, k), `{name} row {row} k {k}`)
                end
                for g = 0, a.kpad // 32 - 1 do
                    h.equal(tensor.getScale(b, row, g), tensor.getScale(a, row, g), `{name} row {row} group {g}`)
                end
            end
            h.equal(
                tensor.getQuant(b, b.rowsPadded - 1, 3),
                if b.rowsPadded > b.rows then 0 else tensor.getQuant(b, b.rowsPadded - 1, 3)
            )
        end
    end)

    h.case("a vector loads unpadded", function()
        h.equal(buffer.len(container.loadVector(c, "final_norm")), 60 * 4)
    end)

    h.case("a missing tensor or blob names itself", function()
        expectError(h, function()
            container.loadTensor(c, "layers.9.q")
        end, "layers.9.q")
        expectError(h, function()
            container.blob(c, "nothing")
        end, "nothing")
    end)

    h.case("a file with the wrong magic is refused", function()
        expectError(h, function()
            container.open(memory(buffer.fromstring(string.rep("x", 64))))
        end, "wrong magic")
    end)

    h.case("a newer container version is refused with advice", function()
        const bytes = buffer.create(64)
        buffer.writestring(bytes, 0, "OLELO\0\0\0")
        buffer.writeu32(bytes, 8, 2)
        buffer.writeu32(bytes, 12, 2)
        buffer.writestring(bytes, 16, "{}")
        expectError(h, function()
            container.open(memory(bytes))
        end, "version 2")
    end)
end
