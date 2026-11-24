--!native
-- Types shared by the modules of src/arch.

const config = require("@src/model/config")
const opsTypes = require("@src/ops/types")

--- What differs between decoder families.
export type Arch = {
    name: string,
    --- q, k and v projections carry a bias.
    qkvBias: boolean,
    --- Each head's q and k are normalized before the rotation.
    qkNorm: boolean,
    --- The q and k norm has no weight, only the scale to unit mean square.
    qkNormUnit: boolean,
    --- Norm weights apply as `1 + w`.
    normPlusOne: boolean,
    --- Scale applied to the embeddings.
    embedScale: (hidden: number) -> number,
    activation: opsTypes.Activation,
    --- Positions layer `layer` (from 1) may look back, nil for all of them. A layer with a window is a local layer
    --- and rotates with the model's local base frequency when it has one.
    window: (layer: number, cfg: config.Config) -> number?,
    --- Each block's output is normalized before it joins the residual, with `post_attn_norm` and `post_mlp_norm`.
    postNorms: boolean,
}

return {}
