-- The generator renders the same text every time, the checked-in kernels are exactly what it renders, each carries the
-- generator version and the native directives, and no kernel carries a `+ 0` term.
const fs = require("@std/fs")
const generate = require("@tools/gen/generate")
const harness = require("@tests/harness")

return function(h: harness.Harness)
    const rendered = generate.renderAll()

    h.case("rendering twice gives the same text", function()
        const again = generate.renderAll()
        h.equal(#again, #rendered)
        for i, file in rendered do
            h.equal(again[i].text, file.text, file.path)
        end
    end)

    h.case("the checked-in kernels are what the generator renders", function()
        for _, file in rendered do
            h.truthy(fs.exists(file.path), `{file.path} is missing, run lute run tools/gen/main.luau`)
            h.truthy(
                fs.readFileToString(file.path) == file.text,
                `{file.path} differs, run lute run tools/gen/main.luau`
            )
        end
    end)

    h.case("every module starts native and names the generator version", function()
        for _, file in rendered do
            h.truthy(string.sub(file.text, 1, 9) == "--!native", file.path)
            h.truthy(string.find(file.text, "--!optimize 2", 1, true) ~= nil, file.path)
            h.truthy(
                string.find(file.text, `generator version {generate.GENERATOR_VERSION}.`, 1, true) ~= nil,
                file.path
            )
            h.truthy(string.find(file.text, "const generated: kernelsTypes.Module", 1, true) ~= nil, file.path)
        end
    end)

    h.case("no kernel has a + 0 term and every kernel is a const function", function()
        for _, file in rendered do
            h.truthy(string.find(file.text, "%+ 0[^%d%.]") == nil, `{file.path} has a + 0 term`)
            h.truthy(string.find(file.text, "local function", 1, true) == nil, `{file.path} has a local function`)
        end
    end)
end
