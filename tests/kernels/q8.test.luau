-- The Q8 kernels equal the plain Luau reference on every shape, part split and row range, exactly for the integer and
-- packed ones and within the f32 sum of the groups for the vector one, and with unit scales they equal the integer sum of
-- the weights and activations. The packed and vector kernels also run the extreme weights and activations.
const check = require("@tests/kernels/check")

return check.suiteFor({ "q8_il4e", "q8_p8", "q8_v12" })
