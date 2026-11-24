-- Parity of the tiny random model, whose shapes need padding, with its reference dumps. It runs in every flag mode.
const harness = require("@tests/harness")
const parity = require("@tests/parity")

return function(h: harness.Harness)
    parity.register(
        h,
        "tests/fixtures/tiny-llama/model.olelo",
        "tests/fixtures/tiny-llama/ref",
        { hidden = 1e-4, logits = 1e-4, tie = 1e-3 }
    )
end
