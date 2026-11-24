-- The tiny random Qwen2 (q/k/v biases), Qwen3 (per head q/k norm), Gemma 3 (post norms, local and global layers), LFM2
-- (short convolution layers) and Llama 4 (a unit q/k norm, rows reordered for the rotation) models
-- against their reference dumps, in f32 and in Q8.
const harness = require("@tests/harness")
const parity = require("@tests/parity")

return function(h: harness.Harness)
    for _, name in { "tiny-qwen2", "tiny-qwen3", "tiny-gemma3", "tiny-lfm2", "tiny-llama4" } do
        const dir = `tests/fixtures/{name}`
        parity.register(h, `{dir}/model.olelo`, `{dir}/ref`, { hidden = 1e-4, logits = 1e-4, tie = 1e-3 })
        parity.register(h, `{dir}/model-q8.olelo`, `{dir}/ref`, { hidden = 0.25, logits = 0.12, tie = 0.5, rms = true })
    end
end
