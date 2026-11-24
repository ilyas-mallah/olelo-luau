--!native
-- Finds the architecture plug-in a container names.

const gemma = require("@self/gemma")
const lfm2 = require("@self/lfm2")
const llama = require("@self/llama")
const llama4 = require("@self/llama4")
const qwen = require("@self/qwen")
const types = require("@self/types")

export type Arch = types.Arch

--- The plug-in for the `arch` field of a container header.
const function get(name: string): types.Arch
    if name == "llama" then
        return llama
    end
    if name == "gemma3" then
        return gemma
    end
    if name == "lfm2" then
        return lfm2
    end
    if name == "llama4" then
        return llama4
    end
    if name == "qwen2" then
        return qwen.qwen2
    elseif name == "qwen3" then
        return qwen.qwen3
    end
    error(`architecture {name} is not supported`, 0)
end

return {
    get = get,
}
