#!/usr/bin/env bash
# Runs every check, and the tests in every flag mode. Run from the repository root.
#
#   LUTE            the pinned lute (1.0.1-nightly.20261120), default `lute`
#   OLELO_FLAG_LUTE a lute that honors LUTE_SET_FFLAGS, see tools/lute-fix/README.md. Needed for the flag modes.
#   STYLUA          StyLua 2.6.0, default `stylua`
#   LUAU_COMPILE    luau-compile built from Luau 0.747, for the spill guard and the native test
#
# Modes: none sets no flag, library sets DebugLuauBufferVectorLibrary, both adds DebugLuauBufferVectorFastcalls. In
# none and library the probe must pick the fallback kernels, in both it must pick the vector kernels.
set -euo pipefail

LUTE=${LUTE:-lute}
STYLUA=${STYLUA:-stylua}
FLAG_LUTE=${OLELO_FLAG_LUTE:-}
LIBRARY=DebugLuauBufferVectorLibrary=true
FASTCALLS=DebugLuauBufferVectorFastcalls=true

step() { printf '\n== %s\n' "$*"; }

step "format"
dirs=()
for d in src tools tests backends examples; do [ -d "$d" ] && dirs+=("$d"); done
"$STYLUA" --check "${dirs[@]}"

step "type check"
"$LUTE" check src tools tests backends examples

step "generated kernels match the generator"
"$LUTE" run tools/gen/main.luau --check

step "spill guard"
"$LUTE" run tools/gen/guard.luau

step "Roblox bundle builds"
"$LUTE" run tools/roblox/build.luau build/roblox

if [ -z "$FLAG_LUTE" ]; then
    echo "OLELO_FLAG_LUTE is not set, so only mode none can run. Build one with tools/lute-fix/build.sh." >&2
    step "tests, mode none"
    OLELO_FLAG_MODE=none "$LUTE" run tests/run.luau
    exit 1
fi

step "the flag lute sets the flags it is asked to"
test "$(LUTE_SET_FFLAGS=$LIBRARY,$FASTCALLS "$FLAG_LUTE" run tools/probe_flags.luau)" = "present"

step "tests, mode none"
OLELO_FLAG_MODE=none "$LUTE" run tests/run.luau

# The flag lute must have the fix for the vm.create crash, which the worker VM tests need (OLELO_VM=1).
step "tests, mode none with worker VMs"
OLELO_VM=1 OLELO_FLAG_MODE=none "$FLAG_LUTE" run tests/run.luau parallel/

step "tests, mode library (the vector read exists but is a slow call)"
OLELO_VM=1 OLELO_FLAG_MODE=library LUTE_SET_FFLAGS=$LIBRARY "$FLAG_LUTE" run tests/run.luau

step "tests, mode both (the vector read with its fast call)"
OLELO_VM=1 OLELO_FLAG_MODE=both LUTE_SET_FFLAGS=$LIBRARY,$FASTCALLS "$FLAG_LUTE" run tests/run.luau

step "all checks passed"
