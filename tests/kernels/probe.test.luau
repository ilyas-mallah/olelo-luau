-- The probe selects the vector kernels when buffer.readvectorf32x3 exists and is at least 1.5 times faster than the
-- 3 x readf32 kernel, and the fallback otherwise. OLELO_FLAG_MODE (none, library or both) says which flags the run
-- set, and the result must match it.
const harness = require("@tests/harness")
const probe = require("@src/kernels/probe")
const process = require("@std/process")
const registry = require("@src/kernels/registry")

return function(h: harness.Harness)
    const result = probe.run(nil, 0.02)
    const mode = process.env.OLELO_FLAG_MODE
    const exists = (buffer :: any).readvectorf32x3 ~= nil

    h.case("the probe agrees with whether the vector read exists", function()
        if not exists then
            h.equal(result.vectorRead, false)
            h.equal(result.vectorRate, nil)
        else
            h.truthy(result.vectorRate ~= nil and result.bufferRate ~= nil)
            h.equal(result.vectorRead, (result.vectorRate :: number) >= probe.MIN_RATIO * (result.bufferRate :: number))
        end
    end)

    h.case("the probe matches the flags of the run", function()
        if mode == "none" then
            h.equal(exists, false, "no flag should leave the vector read missing")
            h.equal(result.vectorRead, false)
        elseif mode == "library" then
            h.equal(exists, true, "the library flag should register the vector read")
            h.equal(result.vectorRead, false, result.reason)
        elseif mode == "both" then
            h.equal(exists, true)
            h.equal(result.vectorRead, true, result.reason)
        else
            h.skip("flag mode", "OLELO_FLAG_MODE is not set")
        end
    end)

    h.case("the registry follows the probe", function()
        const entry = registry.select("f32", result)
        h.equal(entry.needsVectorRead, result.vectorRead)
    end)
end
