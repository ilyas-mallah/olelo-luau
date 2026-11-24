-- Compares the engine with reference dumps from tools/convert/reference.py: token ids of the prompts, the hidden state
-- after every layer for the first positions, the logits of the last prompt position and the greedy continuation.
const engineModule = require("@src/api/engine")
const fs = require("@std/fs")
const graph = require("@src/model/graph")
const harness = require("@tests/harness")
const json = require("@src/util/json")
const lute = require("@backends/lute")
const tokenizer = require("@src/tokenizer")

--- Tolerances, relative to the RMS of the reference values.
export type Tolerances = {
    hidden: number,
    logits: number,
    --- Greedy steps whose two best logits differ by less than this may flip.
    tie: number,
    --- Measure the error as the RMS of the difference, not its largest entry. Quantized models have a few channels
    --- whose rounding error is a large part of a row's RMS.
    rms: boolean?,
}

const function readFloats(path: string): buffer
    return buffer.fromstring(fs.readFileToString(path))
end

const function rms(b: buffer, offset: number, count: number): number
    local sum = 0
    for i = 0, count - 1 do
        const v = buffer.readf32(b, offset + i * 4)
        sum += v * v
    end
    return math.sqrt(sum / count)
end

--- The largest absolute difference of `count` floats, or with `meanSquare` their RMS difference.
const function maxDiff(
    a: buffer,
    aOffset: number,
    b: buffer,
    bOffset: number,
    count: number,
    meanSquare: boolean?
): number
    local worst = 0
    local sum = 0
    for i = 0, count - 1 do
        const d = buffer.readf32(a, aOffset + i * 4) - buffer.readf32(b, bOffset + i * 4)
        worst = math.max(worst, math.abs(d))
        sum += d * d
    end
    return if meanSquare then math.sqrt(sum / count) else worst
end

--- Registers the parity cases of the model at `modelPath` against the dumps in `refDir`.
const function register(h: harness.Harness, modelPath: string, refDir: string, tol: Tolerances)
    const meta = json.decode(fs.readFileToString(`{refDir}/meta.json`))
    const engine = engineModule.new({ host = lute.host(), source = lute.fileSource(modelPath), context = 512 })
    const hidden: number = meta.hidden
    for index, prompt in meta.prompts do
        const label = prompt.name

        h.case(`{label}: token ids equal the reference`, function()
            const got = tokenizer.encode(engine.core.tokenizer, prompt.text, nil, prompt.addSpecial ~= false)
            h.equal(#got, #prompt.ids, "token count")
            for i, id in prompt.ids do
                h.equal(got[i], id, `token {i}`)
            end
        end)

        h.case(`{label}: hidden states and logits`, function()
            const dump = readFloats(`{refDir}/p{index - 1}.hidden.f32`)
            const logitsDump = readFloats(`{refDir}/p{index - 1}.logits.f32`)
            const n = math.min(meta.positions, #prompt.ids)
            const sess = engine:session()
            const worst: { number } = {}
            local at = 1
            local logits: buffer? = nil
            while at <= #prompt.ids do
                const count = math.min(engine.core.chunk, #prompt.ids - at + 1)
                const first = at == 1
                logits = graph.forward(
                    sess,
                    prompt.ids,
                    at,
                    count,
                    at + count > #prompt.ids,
                    nil,
                    function(layer: number, state: buffer, stride: number, tokens: number)
                        if not first then
                            return
                        end
                        const reference = (layer - 1) * n * hidden
                        -- The error of each token's row, relative to that row's RMS.
                        local rowWorst = 0
                        for t = 0, n - 1 do
                            const diff =
                                maxDiff(state, t * stride * 4, dump, (reference + t * hidden) * 4, hidden, tol.rms)
                            const scale = rms(dump, (reference + t * hidden) * 4, hidden)
                            rowWorst = math.max(rowWorst, diff / scale)
                        end
                        worst[layer] = rowWorst
                    end
                )
                at += count
            end
            local layerWorst = 0
            for _, v in worst do
                layerWorst = math.max(layerWorst, v)
            end
            const vocab: number = meta.vocab
            const logitsError = maxDiff(logits :: buffer, 0, logitsDump, 0, vocab, tol.rms) / rms(logitsDump, 0, vocab)
            print(`    {label}: hidden {layerWorst}, logits {logitsError}`)
            h.truthy(layerWorst < tol.hidden, `hidden state error {layerWorst} of the RMS, limit {tol.hidden}`)
            h.truthy(logitsError < tol.logits, `logits error {logitsError} of the RMS, limit {tol.logits}`)
        end)

        h.case(`{label}: greedy continuation equals the reference`, function()
            const want: { number } = prompt.generated
            const got = engine:generate(prompt.ids, #want)
            for i, id in want do
                if got[i] ~= id then
                    h.truthy(
                        prompt.margins[i] < tol.tie,
                        `step {i}: got {got[i]}, expected {id}, margin {prompt.margins[i]}`
                    )
                    print(`    {label}: diverged at step {i} inside a near tie (margin {prompt.margins[i]})`)
                    return
                end
            end
        end)
    end
end

return {
    register = register,
}
