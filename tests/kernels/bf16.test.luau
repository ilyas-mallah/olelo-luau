-- The BF16 kernels, vector and scalar, match the plain Luau reference within K x 2^-24 x the sum of |w x| on every
-- shape, part split and row range.
const check = require("@tests/kernels/check")

return check.suiteFor({ "bf16_vector", "bf16_scalar" })
