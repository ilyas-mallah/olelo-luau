# Contributing

## Setup

`foreman install` installs the pinned lute and StyLua (`foreman.toml`). The flag modes and the spill guard need two more
tools, both optional for a small change:

- A lute that sets the vector flags, built by `tools/lute-fix/build.sh <lute checkout>`. Set `OLELO_FLAG_LUTE` to its binary.
- `luau-compile` from Luau 0.747. Set `LUAU_COMPILE` to it.

## Checks

```sh
tools/ci.sh
```

runs, in order: StyLua check, type check, generator drift, the spill guard, then `tests/run.luau` in three modes. The
modes set no flag (`none`), the vector read behind the library flag (`library`), and the library flag with the fast call
(`both`). In `none` and `library` the probe must pick the fallback kernels, and in `both` the vector kernels. A change is
finished when `tools/ci.sh` passes. Run one piece with:

```sh
stylua src tools tests
lute check src tools tests
lute run tools/gen/main.luau --check
lute run tests/run.luau
```

## Rules and why

- **Generated code is never edited.** Change `tools/gen/`, regenerate, and commit both. CI fails on drift, so a hand edit
  is lost.
- **A kernel stays native.** Over 23 spill slots native lowering skips the function and it runs interpreted, a large slowdown. `tools/gen/guard.luau` turns that into a failure.
- **A kernel matches its reference.** Q8 and Q4 produce the reference's exact integer sums, f32 and BF16 stay within the bound in `tests/kernels/check.luau`.
- **Rates are compared in one session.** Rates drift between sessions, so a number from another day proves nothing.
  `tools/bench/kernels.luau` runs the kernels interleaved. See [docs/benchmarks.md](docs/benchmarks.md).
- **No `math.random` in a hot loop.** Each call costs a function call. Use a small LCG or xorshift of your own. One call per
  token is fine.
- **Any Luau feature may be used.** Do not add notes about whether Roblox supports one.
- **Prose is plain.** No dash joins two clauses, comments run one or two lines and say what the code cannot, and docs have
  one purpose each.

## Commits

A subject line and at most a few lines naming the change and the evidence. Commit one change at a time, so a reviewer can read each alone.

## Issues and pull requests

https://github.com/ilyas-mallah/olelo-luau
