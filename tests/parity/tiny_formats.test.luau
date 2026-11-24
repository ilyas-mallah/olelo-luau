-- The BF16, Q8 and Q4 containers of the tiny random model against the f32 reference dumps, with the error each format's
-- rounding allows, and the prefill path against the decode path on the weights themselves.
const data = require("@tests/data")
const engineModule = require("@src/api/engine")
const graph = require("@src/model/graph")
const harness = require("@tests/harness")
const lute = require("@backends/lute")
const parity = require("@tests/parity")

-- Each format is compared with the f32 reference run on its own rounded weights, which leaves the rounding of the
-- activations and the order of the sums, and with the unrounded reference at the loose limits of its rounding.
const FORMATS: { { name: string, ref: string?, exact: parity.Tolerances, loose: parity.Tolerances } } = {
    {
        name = "bf16",
        exact = { hidden = 2e-3, logits = 2e-3, tie = 1e-2 },
        loose = { hidden = 0.3, logits = 0.1, tie = 0.3 },
    },
    {
        name = "q8",
        exact = { hidden = 0.1, logits = 0.1, tie = 0.5, rms = true },
        loose = { hidden = 0.25, logits = 0.12, tie = 0.5, rms = true },
    },
    {
        -- The vector layout holds the same Q8 weights, and needs the vector read.
        name = "q8v",
        ref = "q8",
        exact = { hidden = 0.1, logits = 0.1, tie = 0.5, rms = true },
        loose = { hidden = 0.25, logits = 0.12, tie = 0.5, rms = true },
    },
    {
        name = "q4",
        exact = { hidden = 0.2, logits = 0.15, tie = 1, rms = true },
        loose = { hidden = 1.2, logits = 1, tie = 3, rms = true },
    },
}

return function(h: harness.Harness)
    for _, format in FORMATS do
        if format.name == "q8v" and not data.vectorFast() then
            h.skip("q8v", "the vector layout needs the fast vector read")
            continue
        end
        const model = `tests/fixtures/tiny-llama/model-{format.name}.olelo`
        parity.register(h, model, `tests/fixtures/tiny-llama/ref-{format.ref or format.name}`, format.exact)
        parity.register(h, model, "tests/fixtures/tiny-llama/ref", format.loose)

        h.case(`{format.name} logits do not depend on how the prompt is split into passes`, function()
            const engine = engineModule.new({ host = lute.host(), source = lute.fileSource(model), context = 128 })
            const ids = engine:tokenize("the quick brown fox jumps over the lazy dog, 12345")
            const vocab = engine.core.config.vocab
            const whole = graph.forward(engine:session(), ids, 1, #ids, true, nil, nil) :: buffer
            const sess = engine:session()
            local last: buffer? = nil
            for i = 1, #ids do
                last = graph.forward(sess, ids, i, 1, i == #ids, nil, nil)
            end
            local worst, size = 0, 0
            for i = 0, vocab - 1 do
                const a = buffer.readf32(whole, i * 4)
                worst = math.max(worst, math.abs(a - buffer.readf32(last :: buffer, i * 4)))
                size = math.max(size, math.abs(a))
            end
            h.truthy(worst <= 1e-5 * size, `the two paths differ by {worst} on logits of size {size}`)
        end)
    end
end
