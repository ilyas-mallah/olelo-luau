-- Parity of a real converted model with its reference dumps. Runs when OLELO_MODEL (a container) and OLELO_REF (a
-- dump folder) are set, as tools/convert/README.md describes. OLELO_RECIPE=q8 checks a Q8 container against the same
-- f32 dumps with the tolerances Q8 rounding allows.
const harness = require("@tests/harness")
const parity = require("@tests/parity")
const process = require("@std/process")

return function(h: harness.Harness)
    const model = process.env.OLELO_MODEL
    const ref = process.env.OLELO_REF
    if model == nil or ref == nil then
        h.skip("real model parity", "OLELO_MODEL and OLELO_REF are not set")
        return
    end
    if process.env.OLELO_RECIPE == "q8" then
        parity.register(h, model, ref, { hidden = 0.6, logits = 0.08, tie = 1, rms = true })
    else
        parity.register(h, model, ref, { hidden = 3e-3, logits = 1e-3, tie = 1e-3 })
    end
end
