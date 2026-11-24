#!/usr/bin/env bash
# Builds a lute that honors LUTE_SET_FFLAGS, which the flag modes of ci.sh need until the vector read ships.
#
#   tools/lute-fix/build.sh <lute checkout> [build dir]
#
# The checkout is luau-lang/lute at v1.0.1-nightly.20261120 with the fix for the repeated `vm.create` crash (Luau 0.748, see
# tools/lute-fix/README.md) and its `extern/` dependencies fetched. The patch only adds the environment variable.
# The binary is `<build dir>/lute/cli/lute`, default build dir `<checkout>/build/release`. Needs cmake, ninja and clang.
set -euo pipefail

checkout=$(realpath "$1")
build=${2:-$checkout/build/release}
patch=$(realpath "$(dirname "$0")/set-fflags.patch")

cd "$checkout"
if ! grep -q LUTE_SET_FFLAGS lute/cli/src/luauflags.cpp; then
    git apply "$patch"
fi
lute run tools/luthier.luau generate
mkdir -p "$build"
cd "$build"
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DLUTE_VERSION_SUFFIX=nightly.20261120 "$checkout"
ninja Lute.CLI
echo "built $build/lute/cli/lute"
