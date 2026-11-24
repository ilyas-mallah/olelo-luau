-- The registry lists every generated kernel with a reference, selects the vector kernels only when the host can run
-- them, and falls back in the order of its preference lists.
const harness = require("@tests/harness")
const registry = require("@src/kernels/registry")

return function(h: harness.Harness)
    h.case("every kernel has a tile, a step that is a multiple of its unroll, and a reference", function()
        h.truthy(#registry.all() >= 8)
        for _, entry in registry.all() do
            h.truthy(entry.R >= 1 and entry.S >= 1 and entry.U >= 1, entry.id)
            h.equal(entry.kstep % entry.U, 0, entry.id)
            h.truthy(entry.reference ~= nil, entry.id)
            h.equal(entry.id, `{entry.module}/{entry.name}`)
        end
    end)

    h.case("with the vector read the decode kernels are the measured primaries", function()
        const caps = { vectorRead = true }
        h.equal(registry.select("f32", caps).id, "f32_vector/R8S1U8")
        h.equal(registry.select("bf16", caps).id, "bf16_vector/R4S1U8")
        h.equal(registry.select("q8", caps).id, "q8_p8/R8S1U32")
        h.equal(registry.select("q4", caps).id, "q4_p4/R8S1U32")
        h.equal(registry.select("f32", caps, true).id, "f32_vector/R4S2U2")
        h.equal(registry.select("q8", caps, true).id, "q8_p8/R4S2U32")
        h.equal(registry.select("q4", caps, true).id, "q4_p4/R4S2U32")
    end)

    h.case("Q8 matrices in the vector layout get the vector kernels, and an error without the vector read", function()
        h.equal(registry.select("q8", { vectorRead = true }, false, "v12").id, "q8_v12/R12S1U32")
        h.equal(registry.select("q8", { vectorRead = true }, true, "v12").id, "q8_v12/R6S2U32")
        h.fails(function()
            registry.select("q8", { vectorRead = false }, false, "v12")
        end, "v12")
    end)

    h.case("without it only kernels that do not need it are selected", function()
        const caps = { vectorRead = false }
        h.equal(registry.select("f32", caps).id, "f32_buffer/R8S1U8")
        h.equal(registry.select("f32", caps, true).id, "f32_buffer/R8S1U8")
        h.equal(registry.select("bf16", caps).id, "bf16_scalar/R8S1U8")
        for _, format in { "f32", "bf16", "q8", "q4" } do
            h.equal(registry.select(format :: any, caps).needsVectorRead, false)
        end
    end)

    h.case("a kernel id that does not exist is an error", function()
        h.fails(function()
            registry.find("f32_vector/R1S1U1")
        end)
    end)
end
