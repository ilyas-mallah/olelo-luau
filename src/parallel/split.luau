--!native
-- How the model is divided among workers: query heads and the KV heads they read, the columns of the MLP and the rows
-- of the output matrix. Pure arithmetic, no host.

const config = require("@src/model/config")
const formats = require("@src/tensor/formats")
const types = require("./types")

type Share = types.Share

--- Divides `units` units into `workers` runs that differ by at most one, the first runs the longer. Returns the start
--- and the count of run `index` from 1.
const function run(units: number, workers: number, index: number): (number, number)
    const each, extra = units // workers, units % workers
    const count = each + (if index <= extra then 1 else 0)
    const start = (index - 1) * each + math.min(index - 1, extra)
    return start, count
end

--- The shares of `workers` workers. The MLP is divided in groups of 32 columns, so that a quantized `down` matrix is cut
--- on its groups, the vocabulary in groups of 8 rows, and a quantized model also needs a head dimension that is a
--- multiple of 32, so that `o` is cut on its groups. Errors with the reason when the model cannot be divided so.
const function split(cfg: config.Config, workers: number, quantized: boolean): { Share }
    if workers < 1 then
        error("a model is divided among at least one worker", 0)
    end
    if workers > cfg.heads then
        error(`the model has {cfg.heads} attention heads and cannot be divided among {workers} workers`, 0)
    end
    const interUnits = (cfg.intermediate + formats.GROUP - 1) // formats.GROUP
    if workers > interUnits then
        error(`the MLP has {interUnits} groups of columns and cannot be divided among {workers} workers`, 0)
    end
    if quantized and cfg.headDim % formats.GROUP ~= 0 then
        error(`a quantized model is divided only when its head dimension {cfg.headDim} is a multiple of 32`, 0)
    end
    const group = cfg.heads // cfg.kvHeads
    const vocabUnits = (cfg.vocab + 7) // 8
    const shares: { Share } = {}
    for i = 1, workers do
        const headStart, headCount = run(cfg.heads, workers, i)
        const interUnit, interUnitCount = run(interUnits, workers, i)
        const vocabUnit, vocabUnitCount = run(vocabUnits, workers, i)
        const kvStart = headStart // group
        const kvEnd = (headStart + headCount - 1) // group
        const interStart = interUnit * formats.GROUP
        const vocabStart = vocabUnit * 8
        table.insert(shares, {
            index = i,
            workers = workers,
            headStart = headStart,
            headCount = headCount,
            kvStart = kvStart,
            kvCount = kvEnd - kvStart + 1,
            interStart = interStart,
            interCount = math.min(interUnitCount * formats.GROUP, cfg.intermediate - interStart),
            vocabStart = vocabStart,
            vocabCount = math.min(vocabUnitCount * 8, cfg.vocab - vocabStart),
        })
    end
    return shares
end

--- The share of one worker that runs the whole model.
const function whole(cfg: config.Config): Share
    return {
        index = 1,
        workers = 1,
        headStart = 0,
        headCount = cfg.heads,
        kvStart = 0,
        kvCount = cfg.kvHeads,
        interStart = 0,
        interCount = cfg.intermediate,
        vocabStart = 0,
        vocabCount = cfg.vocab,
    }
end

return {
    split = split,
    whole = whole,
}
