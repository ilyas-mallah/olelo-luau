# A lute that sets the vector flags and has the vm fix

The vector kernels need `buffer.readvectorf32x3` and its fast call. Luau 0.747 has them behind two fast flags,
`DebugLuauBufferVectorLibrary` and `DebugLuauBufferVectorFastcalls`, and the stock lute turns on only flags whose name
starts with `Luau`. `set-fflags.patch` adds `LUTE_SET_FFLAGS=Name=true,Other=false` to lute's flag setup, nothing else.
`build.sh` applies it to a lute checkout and builds it. The checkout also needs the fix for the crash of repeated
`vm.create` calls (a data race on registry refs, branch `fix/vm-cross-thread-ref-release`), which tensor parallel in Lute
needs. The tests that start worker VMs run only with `OLELO_VM=1`, which `ci.sh` sets for this lute.

```sh
tools/lute-fix/build.sh ~/lute
OLELO_FLAG_LUTE=~/lute/build/release/lute/cli/lute tools/ci.sh
```

Flags are process wide, so a worker VM started with `vm.create` sees them too. The probe finds the vector read fast
inside every worker VM (`docs/benchmarks.md`, L6).
