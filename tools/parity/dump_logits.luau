-- Writes the logits of the last position of a prompt as raw f32, for comparison with another engine's dump.
--
--   lute run tools/parity/dump_logits.luau <model.olelo> <out.f32> <prompt text>
--
-- The text is tokenized as it is, with no chat template.
const engineModule = require("@src/api/engine")
const fs = require("@std/fs")
const graph = require("@src/model/graph")
const lute = require("@backends/lute")
const process = require("@std/process")

const modelPath = process.args[2]
const outPath = process.args[3]
const text = process.args[4]
if modelPath == nil or outPath == nil or text == nil then
    print("usage: lute run tools/parity/dump_logits.luau <model.olelo> <out.f32> <prompt text>")
    process.exit(2)
end

const engine = engineModule.new({ host = lute.host(), source = lute.fileSource(modelPath), context = 512 })
const ids = engine:tokenize(text)
const sess = engine:session()
local at = 1
local logits: buffer? = nil
while at <= #ids do
    const count = math.min(engine.core.chunk, #ids - at + 1)
    logits = graph.forward(sess, ids, at, count, at + count > #ids, nil, nil)
    at += count
end
const vocab = engine.core.config.vocab
fs.writeStringToFile(outPath, buffer.readstring(logits :: buffer, 0, vocab * 4))
print(`wrote {vocab} logits for {#ids} tokens to {outPath}`)
