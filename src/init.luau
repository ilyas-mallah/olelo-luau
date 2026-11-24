--!native
-- The public entry of the core. It exposes the tensor and kernel layers.

return {
    formats = require("@src/tensor/formats"),
    store = require("@src/tensor/store"),
    tensor = require("@src/tensor"),
    quant = require("@src/tensor/quant"),
    registry = require("@src/kernels/registry"),
    probe = require("@src/kernels/probe"),
}
