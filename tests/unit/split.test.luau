-- The division of a model among workers, and the loading of a share's slices of a matrix.
const config = require("@src/model/config")
const container = require("@src/container")
const harness = require("@tests/harness")
const lute = require("@backends/lute")
const split = require("@src/parallel/split")
const tensor = require("@src/tensor")
const tensorTypes = require("@src/tensor/types")

const function model(
    heads: number,
    kvHeads: number,
    headDim: number,
    intermediate: number,
    vocab: number
): config.Config
    return {
        hidden = heads * headDim,
        layers = 2,
        heads = heads,
        kvHeads = kvHeads,
        headDim = headDim,
        intermediate = intermediate,
        vocab = vocab,
        ropeTheta = 10000,
        rmsEps = 1e-5,
        maxPositions = 256,
        tieEmbeddings = true,
        activation = "silu",
    }
end

--- The same bytes of weights and scales at (`row`, `col`) of the slice `s` and of the whole `t`.
const function sameWeight(
    t: tensorTypes.Tensor,
    s: tensorTypes.Tensor,
    row: number,
    col: number,
    rowStart: number,
    colStart: number
): boolean
    if t.format == "f32" or t.format == "bf16" then
        return tensor.getFloat(t, rowStart + row, colStart + col) == tensor.getFloat(s, row, col)
    end
    if tensor.getQuant(t, rowStart + row, colStart + col) ~= tensor.getQuant(s, row, col) then
        return false
    end
    return col % 32 ~= 0
        or tensor.getScale(t, rowStart + row, (colStart + col) // 32) == tensor.getScale(s, row, col // 32)
end

return function(h: harness.Harness)
    h.case("a division covers every head, group of columns and row once", function()
        for _, workers in { 1, 2, 3, 4, 5 } do
            const cfg = model(10, 5, 64, 1000, 5003)
            const shares = split.split(cfg, workers, true)
            h.equal(#shares, workers)
            local heads, inter, vocab = 0, 0, 0
            for i, share in shares do
                h.equal(share.headStart, heads, "heads follow each other")
                h.equal(share.interStart, inter)
                h.equal(share.vocabStart, vocab)
                heads += share.headCount
                inter += share.interCount
                vocab += share.vocabCount
                h.equal(share.index, i)
                h.equal(share.interStart % 32, 0)
                h.equal(share.vocabStart % 8, 0)
                h.truthy(share.headCount >= 1 and share.interCount >= 1 and share.vocabCount >= 1)
            end
            h.equal(heads, 10)
            h.equal(inter, 1000)
            h.equal(vocab, 5003)
        end
    end)

    h.case("a KV head whose query group spans two workers is in both shares", function()
        -- 4 heads over 1 KV head, three workers: heads 0 and 1, then 2, then 3, all read KV head 0.
        const shares = split.split(model(4, 1, 32, 192, 400), 3, true)
        for _, share in shares do
            h.equal(share.kvStart, 0)
            h.equal(share.kvCount, 1)
        end
        -- 6 heads over 2 KV heads (groups of 3), 4 workers: heads 0 1, 2 3, 4, 5.
        const spans = split.split(model(6, 2, 32, 192, 400), 4, true)
        h.equal(spans[2].kvStart, 0)
        h.equal(spans[2].kvCount, 2)
        h.equal(spans[3].kvStart, 1)
        h.equal(spans[3].kvCount, 1)
    end)

    h.case("a model that cannot be divided says why", function()
        h.fails(function()
            split.split(model(4, 2, 64, 1000, 400), 5, false)
        end, "more workers than heads")
        h.fails(function()
            split.split(model(8, 2, 64, 64, 400), 3, false)
        end, "more workers than groups of columns")
        h.fails(function()
            split.split(model(4, 2, 10, 1000, 400), 2, true)
        end, "a quantized head dimension that is not a multiple of 32")
        h.equal(#split.split(model(4, 2, 10, 1000, 400), 2, false), 2)
    end)

    for _, format in { "", "-bf16", "-q8", "-q8v", "-q4" } do
        h.case(`slices of a {if format == "" then "-f32" else format} matrix hold the weights of the whole`, function()
            const c = container.open(lute.fileSource(`tests/fixtures/tiny-llama/model{format}.olelo`))
            const whole = container.loadTensor(c, "layers.0.gate")
            -- gate is 100 x 60, padded to 104 rows (a tile of 8) and 64 columns.
            const rows = container.loadSlice(c, "layers.0.gate", 8, 40, 0, 60)
            h.equal(rows.rows, 40)
            for row = 0, 39 do
                for col = 0, 59 do
                    h.truthy(sameWeight(whole, rows, row, col, 8, 0), `row {row}, column {col}`)
                end
            end
            const cols = container.loadSlice(c, "layers.0.gate", 0, 100, 32, 28)
            h.equal(cols.cols, 28)
            for row = 0, 99 do
                for col = 0, 27 do
                    h.truthy(sameWeight(whole, cols, row, col, 0, 32), `row {row}, column {col}`)
                end
            end
            -- The last rows and a corner, which end at the matrix's edge.
            const corner = container.loadSlice(c, "layers.0.gate", 96, 4, 32, 28)
            for row = 0, 3 do
                for col = 0, 27 do
                    h.truthy(sameWeight(whole, corner, row, col, 96, 32), `row {row}, column {col}`)
                end
            end
            -- The padding of a slice is zero weights.
            const padded = container.loadSlice(c, "layers.0.down", 0, 60, 0, 100)
            h.equal(padded.kpad % 8, 0)
            h.truthy(padded.rowsPadded >= 60)
        end)
    end

    h.case("a slice asks for whole groups and tiles where the format needs them", function()
        const c = container.open(lute.fileSource("tests/fixtures/tiny-llama/model-q8.olelo"))
        h.fails(function()
            container.loadSlice(c, "layers.0.gate", 0, 100, 10, 20)
        end, "columns of a quantized slice start on a group")
        h.fails(function()
            container.loadSlice(c, "layers.0.gate", 4, 40, 0, 60)
        end, "rows of a packed Q8 slice are whole tiles")
        h.fails(function()
            container.loadSlice(c, "layers.0.gate", 0, 101, 0, 60)
        end, "rows outside the matrix")
    end)

    h.case("a slice of a vector holds the values asked for", function()
        const c = container.open(lute.fileSource("tests/fixtures/tiny-qwen2/model.olelo"))
        const whole = container.loadVector(c, "layers.0.q_bias")
        const part = container.loadVectorSlice(c, "layers.0.q_bias", 20, 30)
        h.equal(buffer.len(part), 120)
        for i = 0, 29 do
            h.equal(buffer.readf32(part, i * 4), buffer.readf32(whole, (20 + i) * 4))
        end
    end)
end
