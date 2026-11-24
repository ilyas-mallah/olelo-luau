--!native
-- Chat prompts from the declarative template the converter extracts, so the runtime needs no Jinja. Three kinds: ChatML
-- (SmolLM2, Qwen, LFM2), Llama 3's header markers (MobileLLM) and Gemma's turn markers.

const types = require("./types")

type Message = types.Message
type Spec = types.Spec

const function check(spec: Spec)
    if spec.kind ~= "chatml" and spec.kind ~= "gemma" and spec.kind ~= "llama3" then
        error(`chat template kind {spec.kind} is not supported`, 0)
    end
end

--- Removes leading and trailing whitespace.
const function trim(text: string): string
    return (string.match(text, "^%s*(.-)%s*$")) :: string
end

--- One message as the model saw it in training.
const function message(spec: Spec, role: string, content: string): string
    check(spec)
    if spec.kind == "gemma" then
        return `<start_of_turn>{if role == "assistant" then "model" else role}\n{trim(content)}<end_of_turn>\n`
    end
    if spec.kind == "llama3" then
        return `<|start_header_id|>{role}<|end_header_id|>\n\n{trim(content)}<|eot_id|>`
    end
    return `<|im_start|>{role}\n{content}<|im_end|>\n`
end

--- The text that starts a model turn.
const function generationPrompt(spec: Spec): string
    check(spec)
    if spec.kind == "gemma" then
        return "<start_of_turn>model\n"
    end
    if spec.kind == "llama3" then
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"
    end
    return "<|im_start|>assistant\n" .. (spec.generationPrefix or "")
end

--- The text that closes a model turn after its content, the stop token included.
const function turnEnd(spec: Spec): string
    check(spec)
    if spec.kind == "gemma" then
        return "<end_of_turn>\n"
    end
    if spec.kind == "llama3" then
        return "<|eot_id|>"
    end
    return "<|im_end|>\n"
end

--- The system turn that follows the opening: `system`, or the template's default, "" for none.
const function systemTurn(spec: Spec, system: string?): string
    if system ~= nil then
        return message(spec, "system", system)
    elseif spec.defaultSystem ~= nil then
        return message(spec, "system", spec.defaultSystem :: string)
    end
    return ""
end

--- The text every conversation with the system message `system` starts with, nil when the template has no separate
--- system turn (Gemma) or no system message at all.
const function systemPrefix(spec: Spec, system: string?): string?
    check(spec)
    if spec.kind == "gemma" then
        return nil
    end
    const turn = systemTurn(spec, system)
    return if turn == "" then nil else (spec.opening or "") .. turn
end

--- The text of a conversation up to and including its first user message. `system` is the system message, nil for
--- the model's own default. Gemma has no system role and puts the system message in front of the first user message.
const function opening(spec: Spec, system: string?, user: string): string
    check(spec)
    if spec.kind == "gemma" then
        const prefix = if system ~= nil then system .. "\n\n" else ""
        return (spec.opening or "") .. message(spec, "user", prefix .. trim(user))
    end
    return (spec.opening or "") .. systemTurn(spec, system) .. message(spec, "user", user)
end

--- The whole prompt of a conversation, as the Hugging Face chat template renders it.
const function render(spec: Spec, messages: { Message }, addGenerationPrompt: boolean): string
    check(spec)
    const parts: { string } = {}
    local first = 1
    local system: string? = nil
    if #messages > 0 and messages[1].role == "system" then
        system = messages[1].content
        first = 2
    end
    table.insert(parts, spec.opening or "")
    if spec.kind == "gemma" then
        for i = first, #messages do
            const content = if i == first and system ~= nil
                then system .. "\n\n" .. trim(messages[i].content)
                else messages[i].content
            table.insert(parts, message(spec, messages[i].role, content))
        end
    else
        const fallback = spec.defaultSystem
        if system == nil and fallback ~= nil then
            table.insert(parts, message(spec, "system", fallback))
        end
        for _, m in messages do
            table.insert(parts, message(spec, m.role, m.content))
        end
    end
    if addGenerationPrompt then
        table.insert(parts, generationPrompt(spec))
    end
    return table.concat(parts)
end

return {
    message = message,
    generationPrompt = generationPrompt,
    turnEnd = turnEnd,
    opening = opening,
    systemPrefix = systemPrefix,
    render = render,
}
