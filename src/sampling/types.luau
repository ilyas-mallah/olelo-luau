--!native
-- Types shared by the modules of src/sampling.

export type Rng = {
    state: number,
}

--- Sampling settings, every field optional.
export type Options = {
    --- 0 picks the best token, the default.
    temperature: number?,
    --- Keep this many of the best tokens, 0 for all.
    topK: number?,
    --- Keep the smallest set of best tokens whose probabilities sum to this, 1 for all.
    topP: number?,
    --- Drop tokens whose probability is below this fraction of the best one.
    minP: number?,
    --- Divides positive and multiplies negative logits of recent tokens, once per distinct token, 1 for none.
    repetitionPenalty: number?,
    --- How many recent tokens count as recent, 64 when nil.
    repetitionWindow: number?,
    --- Subtracted from the logit of a recent token for each time it occurred in the window.
    frequencyPenalty: number?,
    --- Subtracted from the logit of every token that occurred in the window.
    presencePenalty: number?,
}

return {}
