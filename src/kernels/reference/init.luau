--!native
-- Plain Luau references for the kernels. The public face of `src/kernels/reference`: its types and the modules a caller uses.

return {
    float = require("@self/float"),
    quant = require("@self/quant"),
}
