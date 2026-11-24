-- Runs every suite and exits 1 when a case fails. `lute run tests/run.luau [filter]` runs the suites whose name
-- contains the filter. Run from the repository root.
const harness = require("@tests/harness")
const process = require("@std/process")

const SUITES: { { name: string, suite: harness.Suite } } = {
    { name = "unit/formats", suite = require("@tests/unit/formats.test") },
    { name = "unit/store", suite = require("@tests/unit/store.test") },
    { name = "unit/tensor", suite = require("@tests/unit/tensor.test") },
    { name = "unit/quant", suite = require("@tests/unit/quant.test") },
    { name = "unit/prepare", suite = require("@tests/unit/prepare.test") },
    { name = "unit/json", suite = require("@tests/unit/json.test") },
    { name = "unit/sampling", suite = require("@tests/unit/sampling.test") },
    { name = "unit/tokenizer", suite = require("@tests/unit/tokenizer.test") },
    { name = "unit/container", suite = require("@tests/unit/container.test") },
    { name = "unit/split", suite = require("@tests/unit/split.test") },
    { name = "unit/attention", suite = require("@tests/unit/attention.test") },
    { name = "unit/ops", suite = require("@tests/unit/ops.test") },
    { name = "unit/kvcache", suite = require("@tests/unit/kvcache.test") },
    { name = "unit/shortconv", suite = require("@tests/unit/shortconv.test") },
    { name = "unit/index", suite = require("@tests/unit/index.test") },
    { name = "kernels/f32_vector", suite = require("@tests/kernels/f32_vector.test") },
    { name = "kernels/f32_buffer", suite = require("@tests/kernels/f32_buffer.test") },
    { name = "kernels/f32_scalar", suite = require("@tests/kernels/f32_scalar.test") },
    { name = "kernels/bf16", suite = require("@tests/kernels/bf16.test") },
    { name = "kernels/q8", suite = require("@tests/kernels/q8.test") },
    { name = "kernels/q4", suite = require("@tests/kernels/q4.test") },
    { name = "kernels/registry", suite = require("@tests/kernels/registry.test") },
    { name = "kernels/probe", suite = require("@tests/kernels/probe.test") },
    { name = "gen/generator", suite = require("@tests/gen/generator.test") },
    { name = "gen/spill", suite = require("@tests/gen/spill.test") },
    { name = "api/chat", suite = require("@tests/api/chat.test") },
    { name = "api/prefix", suite = require("@tests/api/prefix.test") },
    { name = "parallel/parallel", suite = require("@tests/parallel/parallel.test") },
    { name = "serve/scheduler", suite = require("@tests/serve/scheduler.test") },
    { name = "serve/layout", suite = require("@tests/serve/layout.test") },
    { name = "serve/pool", suite = require("@tests/serve/pool.test") },
    { name = "roblox/roblox", suite = require("@tests/roblox/roblox.test") },
    { name = "roblox/chat_service", suite = require("@tests/roblox/chat_service.test") },
    { name = "parity/tiny", suite = require("@tests/parity/tiny.test") },
    { name = "parity/tiny_formats", suite = require("@tests/parity/tiny_formats.test") },
    { name = "parity/tiny_families", suite = require("@tests/parity/tiny_families.test") },
    { name = "parity/real", suite = require("@tests/parity/real.test") },
    { name = "gen/native", suite = require("@tests/gen/native.test") },
}

const filter = process.args[2]
const totals: harness.Totals = { passed = 0, failed = 0, skipped = 0 }
for _, entry in SUITES do
    if filter == nil or string.find(entry.name, filter, 1, true) then
        harness.runSuite(entry.name, entry.suite, totals)
    end
end
print(`\n{totals.passed} passed, {totals.failed} failed, {totals.skipped} skipped`)
if totals.failed > 0 then
    process.exit(1)
end
