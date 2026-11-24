-- The f32 vector kernels, which read with buffer.readvectorf32x3, match the plain Luau reference within K x 2^-24 x the
-- sum of |w x| on every shape, part split and row range.
const check = require("@tests/kernels/check")

return check.suiteFor({ "f32_vector" })
