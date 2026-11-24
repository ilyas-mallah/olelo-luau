-- The fallback f32 kernel, which builds each vector from three readf32 calls, matches the plain Luau reference within
-- K x 2^-24 x the sum of |w x| on every shape, part split and row range.
const check = require("@tests/kernels/check")

return check.suiteFor({ "f32_buffer" })
