-- Prints `present` when this lute registers buffer.readvectorf32x3, `missing` otherwise. ci.sh uses it to check that a
-- lute honors LUTE_SET_FFLAGS before it trusts a flag mode.
print(if (buffer :: any).readvectorf32x3 ~= nil then "present" else "missing")
