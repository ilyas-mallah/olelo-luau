-- The Q4 kernel equals the plain Luau reference exactly on every shape, part split and row range, and with unit scales
-- it equals the integer sum of the weights minus 8 and the activations.
const check = require("@tests/kernels/check")

return check.suiteFor({ "q4_byte", "q4_p4" })
