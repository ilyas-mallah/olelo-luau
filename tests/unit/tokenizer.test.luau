-- The tokenizer against the Hugging Face tokenizers: the SmolLM2 and Qwen3 vocabularies on a corpus, and the small trained one.
const container = require("@src/container")
const fs = require("@std/fs")
const harness = require("@tests/harness")
const json = require("@src/util/json")
const lute = require("@backends/lute")
const templates = require("@src/tokenizer/templates")
const tokenizer = require("@src/tokenizer")

const function listOf(values: { number }): string
    return table.concat(values, ",")
end

const function register(h: harness.Harness, label: string, modelPath: string, fixturePath: string)
    const c = container.open(lute.fileSource(modelPath))
    const tok = tokenizer.new(c)
    const fixture = json.decode(fs.readFileToString(fixturePath))

    h.case(`{label}: corpus ids equal the reference`, function()
        for _, entry in fixture.texts do
            const got = tokenizer.encode(tok, entry.text, nil, true)
            h.equal(listOf(got), listOf(entry.ids), `text {string.format("%q", entry.text)}`)
        end
    end)

    h.case(`{label}: decode returns the text`, function()
        for _, entry in fixture.decoded do
            h.equal(tokenizer.decode(tok, entry.ids), entry.text)
        end
    end)

    h.case(`{label}: chat prompts equal apply_chat_template`, function()
        for _, chat in fixture.chats do
            const text = templates.render(tok.chat, chat.messages, true)
            h.equal(text, chat.text)
            h.equal(listOf(tokenizer.encode(tok, text)), listOf(chat.ids))
        end
    end)

    h.case(`{label}: a stream never splits a character`, function()
        for _, entry in fixture.decoded do
            const stream = tokenizer.stream(tok)
            const parts: { string } = {}
            for _, id in entry.ids do
                const piece = tokenizer.push(stream, id)
                h.truthy(utf8.len(piece) ~= nil, "a piece is valid UTF-8")
                table.insert(parts, piece)
            end
            table.insert(parts, tokenizer.flush(stream))
            h.equal(table.concat(parts), entry.text)
        end
    end)
end

return function(h: harness.Harness)
    register(h, "smollm2", "tests/fixtures/tokenizer/smollm2.olelo", "tests/fixtures/tokenizer/smollm2.json")
    register(h, "qwen3", "tests/fixtures/tokenizer/qwen3.olelo", "tests/fixtures/tokenizer/qwen3.json")
    if fs.exists("models/gemma-3-1b-it-q8.olelo") then
        register(h, "gemma3", "models/gemma-3-1b-it-q8.olelo", "tests/fixtures/tokenizer/gemma3.json")
    else
        h.skip("gemma3 tokenizer", "models/gemma-3-1b-it-q8.olelo is not converted")
    end
    for _, real in
        {
            { "lfm2", "models/lfm2.5-350m-q8.olelo" },
            { "mobilellm", "models/mobilellm-r1.5-360m-q8.olelo" },
        }
    do
        if fs.exists(real[2]) then
            register(h, real[1], real[2], `tests/fixtures/tokenizer/{real[1]}.json`)
        else
            h.skip(`{real[1]} tokenizer`, `{real[2]} is not converted`)
        end
    end
    register(h, "tiny", "tests/fixtures/tiny-llama/model.olelo", "tests/fixtures/tiny-llama/tokens.json")
    register(h, "tiny-gemma3", "tests/fixtures/tiny-gemma3/model.olelo", "tests/fixtures/tiny-gemma3/tokens.json")
    register(h, "tiny-lfm2", "tests/fixtures/tiny-lfm2/model.olelo", "tests/fixtures/tiny-lfm2/tokens.json")
    register(h, "tiny-llama4", "tests/fixtures/tiny-llama4/model.olelo", "tests/fixtures/tiny-llama4/tokens.json")

    h.case("special tokens parse as one id, and as text when asked", function()
        const tok = tokenizer.new(container.open(lute.fileSource("tests/fixtures/tokenizer/smollm2.olelo")))
        h.equal(listOf(tokenizer.encode(tok, "<|im_end|>")), "2")
        h.truthy(#tokenizer.encode(tok, "<|im_end|>", false) > 1)
    end)

    h.case("invalid UTF-8 becomes the replacement character", function()
        const tok = tokenizer.new(container.open(lute.fileSource("tests/fixtures/tokenizer/smollm2.olelo")))
        h.equal(listOf(tokenizer.encode(tok, "a\255b")), listOf(tokenizer.encode(tok, "a\u{FFFD}b")))
    end)

    h.case("a very long piece encodes", function()
        const tok = tokenizer.new(container.open(lute.fileSource("tests/fixtures/tokenizer/smollm2.olelo")))
        const text = string.rep("abc", 5000)
        h.equal(tokenizer.decode(tok, tokenizer.encode(tok, text)), text)
    end)
end
