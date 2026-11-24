# Benchmarks

## Method

`tools/bench/kernel.luau` times one kernel in one process on a 1024 x 4096 matrix with one token. It cycles through copies
of the weights so the matrix streams from memory, times about 0.2 s per repeat, and checks the output against the
reference. `tools/bench/kernels.luau` runs several kernels, or a benchmark file that prints the same `RES` line,
as separate processes interleaved by round, each pinned to CPUs 14 and 15 with `taskset`, and prints the median of the
process medians with the range. Only rates from one session are compared.

```sh
LUTE_SET_FFLAGS=DebugLuauBufferVectorLibrary=true,DebugLuauBufferVectorFastcalls=true \
  $OLELO_FLAG_LUTE run tools/bench/kernels.luau processes=9 f32_vector/R8S1U8 f32_buffer/R8S1U8
```

## Summary

### End to end, one VM, against the first build

Greedy, a 119 token prompt (116 for Qwen), 48 new tokens (32 for Qwen), Lute with both vector flags, one process on
CPUs 10 and 11 per run. Median of 3 runs in tokens per second with the range in brackets. The first build of the engine
(before the vector Q8 kernels, the packed Q4 layout and the other changes of this release) and the current tree
alternate inside one stream, and the last two columns are the median of the per round ratios. Measured 2026-11-23 on the
current tree (vector Q8 included, in the `v12` container rows). Absolute rates varied by up to 2x between rows with CPU
load (the first build's Q8 decodes at 14.4 in an unloaded run), so only the paired columns compare.

| Model | Format | First build prefill | Now prefill | First build decode | Now decode | Paired prefill | Paired decode |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M | f32 | 6.85 (6.10 to 8.09) | 8.07 (7.45 to 9.02) | 5.43 (4.27 to 5.48) | 6.28 (5.18 to 7.32) | x1.11 | x1.35 |
| SmolLM2-135M | Q8, packed p8 | 11.03 (6.47 to 14.28) | 16.57 (11.97 to 18.39) | 6.86 (3.77 to 7.86) | 12.46 (6.59 to 12.76) | x1.50 | x1.75 |
| SmolLM2-135M | Q8, vector v12 | 11.03 (6.47 to 14.28) | 23.49 (16.96 to 25.08) | 6.86 (3.77 to 7.86) | 11.76 (10.81 to 17.36) | x2.13 | x2.53 |
| SmolLM2-135M | Q4 | 4.40 (4.25 to 6.05) | 15.70 (13.73 to 21.42) | 4.45 (2.90 to 4.66) | 10.76 (9.00 to 13.93) | x3.54 | x2.99 |
| Qwen2.5-0.5B | Q8, packed p8 | 3.81 (2.63 to 4.33) | 4.07 (2.92 to 4.12) | 2.01 (1.68 to 2.43) | 1.88 (1.62 to 2.34) | x1.07 | x0.96 |
| Qwen2.5-0.5B | Q8, vector v12 | 3.81 (2.63 to 4.33) | 4.39 (3.60 to 4.44) | 2.01 (1.68 to 2.43) | 2.49 (2.35 to 2.90) | x1.17 | x1.24 |
| Qwen2.5-0.5B | Q4 | 1.03 (0.98 to 1.11) | 3.66 (3.45 to 5.93) | 0.85 (0.81 to 0.94) | 2.71 (2.00 to 3.09) | x3.55 | x2.88 |

The first build's Q4 rows read the row major Q4 container that build writes, every other row reads the same container
on both sides, and the `v12` rows read the repacked container (`repack_q8v.py`) against the plain Q8 one for the first
build. The
Qwen2.5-0.5B Q8 packed row (x0.96 decode against x1.75 for SmolLM2) is inside the noise of that stream and has no
explanation yet, so no claim rests on it.

### Decode on workers

Decode tokens per second, greedy, a 119 token prompt, Lute with the vector flags. The sections below hold the method
and the runs behind each number, and the noise of the runs.

| Model | Format | One VM | Workers | Section |
|---|---|---|---|---|
| SmolLM2-135M-Instruct | f32 | 11.0 to 11.7 | | Generation |
| SmolLM2-135M-Instruct | Q8 | 14.4 | 25.0 at 3 | Tensor parallel in Lute |
| Qwen2.5-0.5B-Instruct | Q8 | 3.1 | 11.2 at 4, 12.0 at 8 | Tensor parallel in Lute |

Serving, SmolLM2-135M Q8 and Qwen2.5-0.5B Q8, one replica VM per chat, the sum of the chats' decode tokens per second:

| Model | 1 replica | 4 replicas | 10 replicas | Memory at 10 | Section |
|---|---|---|---|---|---|
| SmolLM2-135M-Instruct Q8 | 8.5 | 22.3 to 38.2 | 43.5 to 60.5 | 2.5 GiB | Serving |
| Qwen2.5-0.5B-Instruct Q8 | 3.4 to 4.3 | 6.5 to 7.1 | 17.6 to 20.7 | 6.8 GiB | Serving |

The rates in Roblox Studio are in [roblox.md](roblox.md).

### Q4 and Q8 on workers

Decode and prefill tokens per second, greedy, 2 chats per run (the run's value is their median), a 119 token prompt (116
for Qwen), 48 new tokens (32 for Qwen), `tools/bench/generate.luau`, Lute with both vector flags, Q8 in the vector `v12`
container and Q4 packed `p4`, 2026-11-23, 3 interleaved rounds. Workers ran on all 16 CPUs, or on one thread per core
(CPUs 0, 2 ... 14); the single VM ran on CPUs 14 and 15. The runs started with no process above 15 percent CPU. Each run
checks for other CPU-heavy processes before and after it. From round 3 of SmolLM2 Q4 at 8 workers another process
ran, so the 24 runs from there on are marked contaminated and left out of every number below, which leaves 2 or 3
runs per cell. Bursts shorter than a run are not seen by that check. Four runs failed with an engine error, all SmolLM2
Q8 v12 (the last point), so those cells also have fewer rounds.

| Model, format | Workers | CPUs | Decode tok/s (range) | Decode vs 1 VM | Prefill tok/s (range) | Prefill vs 1 VM | Clean runs of 3 |
|---|---|---|---|---|---|---|---|
| SmolLM2 Q8 v12 | 0 | 14,15 | 16.55 (16.20 to 16.89) | x1.00 | 23.54 (20.53 to 26.54) | x1.00 | 2 |
| SmolLM2 Q8 v12 | 2 | 0 to 15 | 17.15 (15.17 to 19.13) | x1.04 | 34.56 (26.75 to 42.37) | x1.47 | 2 |
| SmolLM2 Q8 v12 | 2 | even 0 to 14 | 13.19 (7.45 to 18.92) | x0.80 | 24.80 (21.60 to 28.00) | x1.05 | 2 |
| SmolLM2 Q8 v12 | 3 | 0 to 15 | 16.26 (11.95 to 20.57) | x0.98 | 33.74 (27.13 to 42.00) | x1.43 | 3 |
| SmolLM2 Q8 v12 | 3 | even 0 to 14 | 16.97 (14.02 to 25.06) | x1.03 | 38.81 (31.88 to 52.70) | x1.65 | 3 |
| SmolLM2 Q8 v12 | 4 | 0 to 15 | 16.14 (13.94 to 18.34) | x0.98 | 36.93 (32.74 to 41.12) | x1.57 | 2 |
| SmolLM2 Q8 v12 | 4 | even 0 to 14 | 19.24 (12.84 to 20.25) | x1.16 | 43.94 (27.32 to 49.16) | x1.87 | 3 |
| SmolLM2 Q8 v12 | 6 | 0 to 15 | 14.31 (11.43 to 17.20) | x0.87 | 41.32 (27.16 to 55.48) | x1.76 | 2 |
| SmolLM2 Q8 v12 | 6 | even 0 to 14 | 12.19 (12.19 to 12.19) | x0.74 | 24.25 (24.25 to 24.25) | x1.03 | 1 |
| SmolLM2 Q8 v12 | 8 | 0 to 15 | 11.86 (5.83 to 17.88) | x0.72 | 40.28 (31.44 to 49.12) | x1.71 | 2 |
| SmolLM2 Q8 v12 | 8 | even 0 to 14 | 14.87 (13.22 to 16.52) | x0.90 | 36.55 (30.25 to 42.85) | x1.55 | 2 |
| SmolLM2 Q4 | 0 | 14,15 | 14.63 (12.93 to 16.90) | x1.00 | 22.27 (20.73 to 22.53) | x1.00 | 3 |
| SmolLM2 Q4 | 2 | 0 to 15 | 18.33 (15.19 to 18.49) | x1.25 | 31.17 (24.04 to 36.45) | x1.40 | 3 |
| SmolLM2 Q4 | 2 | even 0 to 14 | 19.12 (16.96 to 21.98) | x1.31 | 31.74 (29.79 to 36.72) | x1.43 | 3 |
| SmolLM2 Q4 | 3 | 0 to 15 | 20.21 (15.67 to 25.05) | x1.38 | 39.61 (29.22 to 44.83) | x1.78 | 3 |
| SmolLM2 Q4 | 3 | even 0 to 14 | 18.97 (17.49 to 23.88) | x1.30 | 37.40 (32.29 to 40.44) | x1.68 | 3 |
| SmolLM2 Q4 | 4 | 0 to 15 | 17.45 (16.81 to 19.08) | x1.19 | 36.11 (29.47 to 38.82) | x1.62 | 3 |
| SmolLM2 Q4 | 4 | even 0 to 14 | 16.00 (12.77 to 16.02) | x1.09 | 32.77 (31.73 to 33.85) | x1.47 | 3 |
| SmolLM2 Q4 | 6 | 0 to 15 | 14.90 (11.49 to 16.11) | x1.02 | 37.24 (29.08 to 43.71) | x1.67 | 3 |
| SmolLM2 Q4 | 6 | even 0 to 14 | 18.13 (14.02 to 20.81) | x1.24 | 36.66 (29.00 to 41.37) | x1.65 | 3 |
| SmolLM2 Q4 | 8 | 0 to 15 | 17.59 (16.21 to 18.97) | x1.20 | 42.39 (37.23 to 47.55) | x1.90 | 2 |
| SmolLM2 Q4 | 8 | even 0 to 14 | 12.27 (7.99 to 16.54) | x0.84 | 35.39 (31.49 to 39.30) | x1.59 | 2 |
| Qwen Q8 v12 | 0 | 14,15 | 4.47 (3.98 to 4.96) | x1.00 | 8.79 (8.47 to 9.10) | x1.00 | 2 |
| Qwen Q8 v12 | 2 | 0 to 15 | 6.59 (6.09 to 7.10) | x1.48 | 11.92 (11.33 to 12.51) | x1.36 | 2 |
| Qwen Q8 v12 | 2 | even 0 to 14 | 7.28 (6.93 to 7.62) | x1.63 | 12.82 (12.24 to 13.41) | x1.46 | 2 |
| Qwen Q8 v12 | 3 | 0 to 15 | 7.88 (7.38 to 8.37) | x1.76 | 16.11 (15.94 to 16.29) | x1.83 | 2 |
| Qwen Q8 v12 | 3 | even 0 to 14 | 5.46 (3.89 to 7.03) | x1.22 | 10.64 (8.00 to 13.28) | x1.21 | 2 |
| Qwen Q8 v12 | 4 | 0 to 15 | 3.76 (3.75 to 3.78) | x0.84 | 8.88 (7.53 to 10.22) | x1.01 | 2 |
| Qwen Q8 v12 | 4 | even 0 to 14 | 3.95 (3.68 to 4.22) | x0.88 | 8.07 (7.33 to 8.82) | x0.92 | 2 |
| Qwen Q8 v12 | 6 | 0 to 15 | 5.99 (4.00 to 7.98) | x1.34 | 11.98 (8.85 to 15.12) | x1.36 | 2 |
| Qwen Q8 v12 | 6 | even 0 to 14 | 7.30 (3.97 to 10.63) | x1.63 | 16.46 (8.49 to 24.43) | x1.87 | 2 |
| Qwen Q8 v12 | 8 | 0 to 15 | 9.80 (9.26 to 10.34) | x2.19 | 23.07 (21.60 to 24.53) | x2.63 | 2 |
| Qwen Q8 v12 | 8 | even 0 to 14 | 7.70 (7.54 to 7.86) | x1.72 | 19.76 (17.30 to 22.21) | x2.25 | 2 |
| Qwen Q4 | 0 | 14,15 | 3.12 (2.72 to 3.53) | x1.00 | 5.00 (4.38 to 5.62) | x1.00 | 2 |
| Qwen Q4 | 2 | 0 to 15 | 5.13 (4.97 to 7.06) | x1.64 | 9.63 (7.98 to 11.86) | x1.93 | 3 |
| Qwen Q4 | 2 | even 0 to 14 | 4.69 (3.37 to 6.56) | x1.50 | 7.45 (5.99 to 10.74) | x1.49 | 3 |
| Qwen Q4 | 3 | 0 to 15 | 5.93 (4.49 to 6.85) | x1.90 | 10.09 (9.73 to 11.97) | x2.02 | 3 |
| Qwen Q4 | 3 | even 0 to 14 | 6.96 (5.71 to 7.09) | x2.23 | 11.93 (7.81 to 14.44) | x2.39 | 3 |
| Qwen Q4 | 4 | 0 to 15 | 7.56 (7.15 to 9.31) | x2.42 | 13.88 (11.72 to 17.15) | x2.78 | 3 |
| Qwen Q4 | 4 | even 0 to 14 | 6.49 (6.22 to 6.71) | x2.08 | 13.81 (9.45 to 14.34) | x2.76 | 3 |
| Qwen Q4 | 6 | 0 to 15 | 7.47 (5.66 to 9.28) | x2.39 | 15.45 (12.20 to 18.70) | x3.09 | 2 |
| Qwen Q4 | 6 | even 0 to 14 | 6.85 (3.27 to 10.43) | x2.19 | 13.29 (9.52 to 17.07) | x2.66 | 2 |
| Qwen Q4 | 8 | 0 to 15 | 7.12 (6.83 to 7.40) | x2.28 | 14.56 (13.84 to 15.29) | x2.91 | 2 |
| Qwen Q4 | 8 | even 0 to 14 | 7.97 (7.25 to 8.70) | x2.55 | 14.29 (14.19 to 14.40) | x2.86 | 2 |

Best configuration of each model and format, median of the clean runs with the range:

- Qwen Q4 decode: 7.97 tok/s (7.25 to 8.70) at 8 workers, one thread per core
- Qwen Q4 prefill: 15.45 tok/s (12.20 to 18.70) at 6 workers, CPUs 0 to 15
- Qwen Q8 v12 decode: 9.80 tok/s (9.26 to 10.34) at 8 workers, CPUs 0 to 15
- Qwen Q8 v12 prefill: 23.07 tok/s (21.60 to 24.53) at 8 workers, CPUs 0 to 15
- SmolLM2 Q4 decode: 20.21 tok/s (15.67 to 25.05) at 3 workers, CPUs 0 to 15
- SmolLM2 Q4 prefill: 42.39 tok/s (37.23 to 47.55) at 8 workers, CPUs 0 to 15
- SmolLM2 Q8 v12 decode: 19.24 tok/s (12.84 to 20.25) at 4 workers, one thread per core
- SmolLM2 Q8 v12 prefill: 43.94 tok/s (27.32 to 49.16) at 4 workers, one thread per core

Best single runs, not rates to expect: 25.06 decode tokens per second (SmolLM2 Q8 v12, 3 workers, one thread per core,
the top of a cell whose median is 16.97) and 55.48 prefill tokens per second (SmolLM2 Q8 v12, 6 workers, all CPUs, cell
median 41.32). The command of the first, which every row of the table follows with its own model, tokens, workers and
CPU list:

```sh
LUTE_SET_FFLAGS=DebugLuauBufferVectorLibrary=true,DebugLuauBufferVectorFastcalls=true \
  taskset -c 0,2,4,6,8,10,12,14 nice -n 5 $OLELO_FLAG_LUTE run tools/bench/generate.luau \
  models/smollm2-135m-q8v.olelo 48 2 1024 3
```

Per run, PAR of the clean runs on all CPUs (summed over the 2 chats): the main worker's wait, its own time (wall minus
wait), the longest worker time of each phase, and the wait that no worker's compute accounts for, in seconds and per
exchange.

| Model, format | Workers | CPUs | Wall s | Main wait s | Main own s | Exchanges | Worker attn s | Worker mlp s | Worker logits s | Wait beyond workers s | Per exchange us |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SmolLM2 Q8 v12 | 2 | all | 9.2 | 8.7 | 0.5 | 3166 | 2.41 | 4.85 | 0.28 | 1.2 | 368 |
| SmolLM2 Q8 v12 | 3 | all | 11.0 | 10.1 | 0.9 | 6216 | 2.22 | 4.70 | 0.49 | 2.7 | 436 |
| SmolLM2 Q8 v12 | 4 | all | 9.7 | 8.7 | 1.0 | 4386 | 2.16 | 3.68 | 0.26 | 2.7 | 605 |
| SmolLM2 Q8 v12 | 6 | all | 9.0 | 7.7 | 1.2 | 3166 | 2.00 | 2.52 | 0.14 | 3.1 | 970 |
| SmolLM2 Q8 v12 | 8 | all | 9.9 | 8.4 | 1.5 | 3166 | 2.07 | 2.18 | 0.17 | 4.0 | 1265 |
| SmolLM2 Q4 | 2 | all | 9.3 | 8.8 | 0.4 | 2922 | 2.44 | 5.32 | 0.30 | 0.8 | 266 |
| SmolLM2 Q4 | 3 | all | 9.6 | 8.9 | 0.7 | 5240 | 2.04 | 4.72 | 0.43 | 1.7 | 333 |
| SmolLM2 Q4 | 4 | all | 8.2 | 7.4 | 0.7 | 2922 | 1.97 | 3.71 | 0.19 | 1.5 | 529 |
| SmolLM2 Q4 | 6 | all | 11.2 | 9.9 | 1.2 | 5240 | 2.79 | 3.76 | 0.33 | 3.0 | 578 |
| SmolLM2 Q4 | 8 | all | 9.5 | 8.1 | 1.4 | 5240 | 2.10 | 2.70 | 0.23 | 3.1 | 586 |
| Qwen Q8 v12 | 2 | all | 28.0 | 27.3 | 0.8 | 3424 | 3.63 | 19.62 | 2.02 | 2.0 | 581 |
| Qwen Q8 v12 | 3 | all | 20.6 | 19.7 | 0.9 | 3326 | 3.12 | 13.17 | 1.31 | 2.1 | 619 |
| Qwen Q8 v12 | 4 | all | 35.5 | 33.6 | 1.8 | 3326 | 5.12 | 17.27 | 2.00 | 9.3 | 2783 |
| Qwen Q8 v12 | 6 | all | 30.7 | 28.5 | 2.2 | 3424 | 3.89 | 12.40 | 1.40 | 10.8 | 3147 |
| Qwen Q8 v12 | 8 | all | 15.1 | 13.4 | 1.7 | 3424 | 2.51 | 7.08 | 0.70 | 3.1 | 902 |
| Qwen Q4 | 2 | all | 35.8 | 35.0 | 0.8 | 3424 | 4.31 | 25.44 | 2.89 | 2.4 | 694 |
| Qwen Q4 | 3 | all | 32.9 | 31.8 | 1.1 | 3424 | 4.72 | 21.21 | 2.53 | 3.4 | 987 |
| Qwen Q4 | 4 | all | 23.3 | 22.2 | 1.1 | 3424 | 3.55 | 14.39 | 1.63 | 2.6 | 773 |
| Qwen Q4 | 6 | all | 23.3 | 21.8 | 1.5 | 3424 | 4.13 | 11.92 | 1.38 | 4.3 | 1269 |
| Qwen Q4 | 8 | all | 21.6 | 19.7 | 1.9 | 3424 | 3.40 | 10.27 | 1.14 | 4.9 | 1432 |

- Single VM, 2 CPUs: SmolLM2 Q8 v12 decodes 16.55 and prefills 23.54, Q4 14.63 and 22.27; Qwen Q8 v12 4.47 and 8.79,
  Q4 3.12 and 5.00.
- The best decode is 1.16 to 1.38 times the single VM for SmolLM2 and 2.19 to 2.55 times for Qwen, prefill 1.87 to 1.90
  times for SmolLM2 and 2.63 to 3.09 times for Qwen. The ranges of neighbouring cells overlap everywhere, so only the
  trend counts: SmolLM2 gains little past 3 or 4 workers and loses at 6 and 8, Qwen keeps gaining to 6 or 8, apart from the Qwen Q8 v12 cell at 4 workers, whose 2 clean runs are both slow.
- Q4 against Q8 v12 on SmolLM2: the single VM is faster with Q8 (16.55 against 14.63 decode), Q8 gains 1.16 at best
  and Q4 1.38. The Q4 gain peaks at 3 workers and falls to x0.84 to x1.24 at 6 and 8. This agrees with the earlier
  reading that the faster single step of Q8 makes the exchange a bigger share, and not with a Q4 that plateaus earlier
  because its single step is faster.
- The main worker's own time is 0.4 to 2.2 s of 7.7 to 40 s, at most 16 percent of a run, and it grows only with the
  worker count (sends and sums more partials). It does not explain a plateau.
- The wait that no worker's compute accounts for is 0.8 to 1.8 s at 2 workers on SmolLM2 and 2.7 to 5.9 s at 6 and 8,
  262 to 574 us per exchange at 2 workers, 422 to 611 at 4, 586 to 1319 at 8 (Qwen 536 to 3529, from cells of 2 runs
  with a wide spread). An earlier run under CPU load showed 0.4 to 0.7 ms per exchange, so none of it was
  contention: an unloaded run shows the same. It is 4 to 5 times the 117 us measured for a bare 2304 byte exchange at 4 workers
  (Tensor parallel in Lute). What it holds is the exchange itself, the larger prefill messages (a chunk of 32 tokens is
  73 KB for SmolLM2), and any worker time outside the three timed phases. The counters cannot split these, so a worker
  side timer around message handling is the next measurement.
- The four failures were the startup probe: it timed the vector read as slow under load and the engine then refused
  the `v12` model ("this model stores its Q8 matrices in the v12 layout, which needs the fast
  buffer.readvectorf32x3"). Rerunning the 4 worker configuration 8 times under load reproduced it once. The
  probe now repeats a failed timing up to 3 times and keeps the best ratio, which costs nothing when the vector read is
  fast.
- The exchange counts follow the number of forward passes, which differ by run (2922 is 48 passes of 61 exchanges, 5240
  is 86) because each greedy output ends at its own length.

### Attention by context length

Where one decode token's attention goes with the unrolled vector attention (f32 cache) and the scalar
`attention_q8` (Q8 cache), SmolLM2-135M (9 heads, 3 KV heads, 64 wide, 30 layers) and Qwen2.5-0.5B (14 heads, 2 KV
heads, 64 wide, 24 layers). `tools/bench/attention_profile.luau` runs the attention op over every layer's own synthetic
cache, like a decode pass, so the caches stream from memory. The parts come from stage variants of the same function:
the f32 kernel ended after the dot products, after the softmax and whole (the generator's `stage`), and for Q8 a replica
of `attention_q8` cut the same way plus the value scale pass as its own part (it prints `replica equals attention_q8:
true` against the real kernel). A part is the difference of two stages. Each stage runs 0.1 s windows that alternate
inside a repeat, and the fastest of 5 to 7 repeats counts, because load only adds time.

```sh
LUTE_SET_FFLAGS=DebugLuauBufferVectorLibrary=true,DebugLuauBufferVectorFastcalls=true \
  taskset -c 4,5 $OLELO_FLAG_LUTE run tools/bench/attention_profile.luau f32 smol,qwen 128,512,1024,2048 7
```

Run 2026-11-23 on CPUs 4 and 5. Timings moved by up to 2x between runs (the matrix kernels
below drift the same way), so the table gives the best of the runs of a cell and the range of their totals. Cells have 2
runs at 128 and 1024 positions, 6 at 512 and 2048 (the score spread runs, see below, count as runs).

**Per cell, best run.** Attention for one token over every layer in ms, its parts in ms with the MMAC/s of the dot and
value loops, ns per head and position, and the step. The step is the best attention plus the decode time without
attention (68 ms for SmolLM2, 248 ms for Qwen2.5-0.5B, the lowest `n = 0` decode of the context runs), so it is a
prediction from two best cases. A part below the noise of the difference (a negative value) is written `~0`.

| Model | n | Cache | Attention ms (range) | Scores ms, MMAC/s | Softmax ms | Value scale ms | Values ms, MMAC/s | ns per head, position | Share of step | Step tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| SmolLM2 | 128 | f32 | 3.4 (3.4 to 4.0) | 1.5, 1517 | ~0 | | 1.9, 1139 | 97 | 5% | 14.0 |
| SmolLM2 | 512 | f32 | 9.5 (9.5 to 22.2) | 3.1, 2876 | 0.6 | | 5.8, 1521 | 68 | 12% | 12.9 |
| SmolLM2 | 1024 | f32 | 30.6 (30.6 to 30.7) | 11.2, 1574 | 2.3 | | 17.0, 1040 | 111 | 31% | 10.2 |
| SmolLM2 | 2048 | f32 | 51.6 (51.6 to 86.7) | 13.4, 2632 | 3.3 | | 34.8, 1015 | 93 | 43% | 8.4 |
| SmolLM2 | 128 | Q8 | 12.1 (12.1 to 13.1) | 6.3, 349 | 0.2 | 1.3 | 4.2, 521 | 350 | 15% | 12.5 |
| SmolLM2 | 512 | Q8 | 46.7 (46.7 to 92.9) | 23.1, 383 | 6.2 | 1.5 | 15.9, 556 | 338 | 41% | 8.7 |
| SmolLM2 | 1024 | Q8 | 107.9 (107.9 to 136.3) | 57.7, 306 | 7.9 | 3.1 | 39.1, 453 | 390 | 61% | 5.7 |
| SmolLM2 | 2048 | Q8 | 184.7 (184.7 to 386.5) | 117.1, 302 | ~0 | 16.9 | 54.3, 652 | 334 | 73% | 4.0 |
| Qwen2.5-0.5B | 128 | f32 | 3.9 (3.9 to 4.9) | 1.1, 2503 | 0.5 | | 2.3, 1205 | 91 | 2% | 4.0 |
| Qwen2.5-0.5B | 512 | f32 | 10.0 (10.0 to 24.4) | 2.9, 3833 | 1.4 | | 5.7, 1932 | 58 | 4% | 3.9 |
| Qwen2.5-0.5B | 1024 | f32 | 22.9 (22.9 to 25.2) | 8.7, 2534 | 0.3 | | 13.9, 1583 | 67 | 8% | 3.7 |
| Qwen2.5-0.5B | 2048 | f32 | 31.0 (31.0 to 100.9) | 10.0, 4392 | 3.5 | | 17.5, 2523 | 45 | 11% | 3.6 |
| Qwen2.5-0.5B | 128 | Q8 | 14.7 (14.7 to 14.9) | 8.0, 345 | ~0 | 1.6 | 5.1, 537 | 341 | 6% | 3.8 |
| Qwen2.5-0.5B | 512 | Q8 | 54.8 (54.8 to 120.6) | 33.1, 333 | ~0 | 5.0 | 17.4, 634 | 319 | 18% | 3.3 |
| Qwen2.5-0.5B | 1024 | Q8 | 104.7 (104.7 to 121.6) | 59.4, 371 | 10.7 | ~0 | 37.4, 589 | 304 | 30% | 2.8 |
| Qwen2.5-0.5B | 2048 | Q8 | 204.3 (204.3 to 506.1) | 110.9, 397 | 13.0 | 10.3 | 70.1, 629 | 297 | 45% | 2.2 |

The key scale of the Q8 cache is folded into its score loop, so it has no part of its own. The softmax of the Q8 cache is a
scalar `math.exp` per position and its difference is below the noise in several cells.

**The matrix kernels on the same CPUs**, 5 interleaved rounds, a 1536 x 576 matrix, 1 token, MMAC/s.
The rounds drifted by 2x (round 5 was the fastest), so both the median of the rounds and the best round are given.

| Kernel | Median of 5 | Best round |
|---|---|---|
| `q8_v12/R12S1U32` | 1420 | 2466 |
| `q8_p8/R8S1U32` | 990 | 2078 |
| `f32_vector/R8S1U8` | 809 | 1542 |
| `f32_scalar/R8S1U8` | 380 | 691 |

Reading the numbers:

- **The f32 attention is no longer slow.** It costs 0.015 to 0.025 ms per position and token at the best run (0.042 to 0.049
  at the worst) against the 0.13 to 0.21 of the context run in [Serving](#serving), which ran before the unrolled vector
  attention existed, so it holds for the old kernel only. At 2048 positions it is 52 ms of a 120 ms SmolLM2 token and 31
  ms of a 280 ms Qwen token. The hypothesis that it is a scalar loop with an `exp` per position and no vector path is
  refuted for the f32 cache: the dot loops run at 2.6 and 4.4 GMAC/s (best), 1.7 and 2.9 times the best `f32_vector`
  matrix kernel, and the softmax is 6 to 11% of the attention at 2048 (3.3 of 51.6 ms on SmolLM2).
- **The value loop is the slow f32 part**: 1.0 GMAC/s on SmolLM2 and 2.5 on Qwen at 2048, 0.66 and 1.6 times the best
  `f32_vector` matrix kernel and 0.4 and 0.57 times its own dot loop, 67% of the attention on SmolLM2. What makes it slower
  than the dot loop is not isolated. Its rate at 128 positions (1139 MMAC/s, rows in L1 and L2) is the rate at 2048
  (1015), so cache size does not explain it.
- **It is not memory traffic.** SmolLM2 at 2048 positions reads 292 MB per token (every head reads the 528 bytes of key and
  value per position, 553 thousand head positions) in 51.6 ms, 5.7 GB/s, against 16.6 GB/s of copy bandwidth for one
  process, and the dot loop is no faster when the rows are resident (1517 MMAC/s at 128, 2632 at 2048).
- **The Q8 cache attention is the slow path now.** Its dot loop runs at 0.30 to 0.40 GMAC/s, 5 to 8 times below the best
  `q8_p8` and `q8_v12` matrix kernels (2.1 and 2.5 GMAC/s) and half of the scalar f32 matrix kernel, and it takes
  297 to 390 ns per head and position against 45 to 111 for the f32 cache. At 2048 positions the Q8 attention is 185 ms of
  a 253 ms SmolLM2 token (73%) and 204 of 452 ms on Qwen (45%), 3.6 and 6.6 times the f32 attention. The context run
  in [Serving](#serving) had the two caches level (2.7 against 2.6 tok/s at 2048) because both ran scalar loops, and
  since then only the f32 one got faster. Each multiply add reads a float from the query and an int8 from the cache through two
  bounds checked buffer reads (the packed matrix kernel reads four bytes at once), which is a hypothesis, not a
  measurement. The value scale pass is 5 to 9% of the Q8 attention.
- **Not a denormal effect.** Query vectors scaled by 300, so scores spread over a range that makes most softmax weights
  tiny, gave the same times as scale 1 within the noise (SmolLM2 at 2048 in the two rounds, Q8: 387 and 379 ms at scale 1, 357
  and 361 at 300; f32: 52 and 87 ms at scale 1, 52 and 87 at 300).
- **The context benchmark agrees within the spread.** `tools/bench/context.luau` on CPUs 14 and 15 ran with competing CPU load.
  Its SmolLM2 f32 decode time at 2048 positions is 117.6 ms against 120 predicted. Its Q8 decode times at 2048 (437 ms
  SmolLM2, 754 ms Qwen) are 1.7 times the best profile estimate and inside the 2x spread of the profile itself. The Qwen
  f32 run failed once with `[C] function error` (not reproduced, the rerun completed). Its decode columns are not used
  above.

**The biggest remaining win is a fast kernel for the Q8 cache.** The estimates are arithmetic on the measured part rates, not
runs of a new kernel. The kernel is built, see [Q8 cache attention in the vector layout](#q8-cache-attention-in-the-vector-layout)
for the measured result.

| Change | Estimated attention at 2048 | Estimated step at 2048 | Worth building |
|---|---|---|---|
| Q8 cache dot and value loops at 1 to 2 GMAC/s (the packed Q8 matrix rate, with the query quantized per head or an unrolled scalar loop that reads four bytes at once) | SmolLM2 40 to 76 ms (from 185), Qwen 57 to 100 ms (from 204) | SmolLM2 6.9 to 9.3 tok/s (from 4.0, f32 cache 8.4), Qwen 2.9 to 3.3 (from 2.2) | Yes. It makes the Q8 cache as fast as the f32 one at a quarter of the bytes. A quantized query adds an error that `kvquality.luau` has to measure first. |
| f32 value loop at the rate of the dot loop | SmolLM2 30 ms (from 52), Qwen 24 ms (from 31) | SmolLM2 10.2 tok/s (from 8.4), Qwen 3.7 (from 3.6) | Only after its slowness is explained, 22% on SmolLM2 and 3% on Qwen. |
| Cheaper `exp` | saves under 3 ms | under 3% | No. |
| Two positions per pass | none left | none | No. The dot loop already takes four positions per pass and runs above the matrix kernel. |

The weights stay the larger cost at short contexts and on Qwen2.5-0.5B (89% of its step at 2048 with the f32 cache), so
no attention change moves the tokens per second of Qwen more than 1.5 times.

### Q8 cache attention in the vector layout

The Q8 cache has a second row layout, `v3`, and a generated attention kernel that reads it (`ops/kvquant`,
`tools/gen/attention_q8.luau`, `ops/generated/attention_q8_unrolled_<lanes>.luau`). A row stores three int8 values to each
f32 lane as `c + b / 256 + a / 65536` (offset binary bytes, exact in f32), so one `buffer.readvectorf32x3` takes nine
values, and the kernel pulls the bytes apart with `vector.floor`. It has the structure of the f32 kernel: the query vectors
are shared by three positions, the softmax is the vector `exp`, and the value scale of each position rides in the softmax
weight. The offset 128 comes off after the loops (128 times the sum of the query for a score, 128 times the sum of the
weights for a value). The engine picks `v3` when the vector read exists and a kernel has the head size, else the `i8`
layout and the scalar `attention_q8`. The stored integers and scales are the same in both layouts, so the kernels agree
with attention over the stored values to 1e-5 (`tests/unit/kvcache.test.luau`: the 4 lane test head with windows of 1 to
12 positions and shares of heads, and for each of the 5 generated kernels a head that fills its last lane and one that
leaves two values in it).

Setup: SmolLM2-135M and Qwen2.5-0.5B (v12 Q8 containers, 64 wide heads, so the 22 lane kernel), both flags set, CPUs 14 and
15, 2026-11-23. The timings were taken before each head size became a module of its own, which is the only change to the kernels
since, and the tests pass on the current tree.
`tools/bench/kvlayout.luau` opens the model once per variant (f32 cache, Q8 `i8`, Q8 `v3`) in one process, fills one
session per variant with the same tokens, and at each length times 16 decode passes per variant, 5 repeats with the
variants rotating inside a repeat (median shown), so a slow stretch hits the three alike. The prefill column is
the rate of the chunks of 32 tokens that took the session up to that length. At `n = 0` the cache is empty, so the three
variants should decode level, and they spread by 10% on SmolLM2 (7.8 to 8.7 tok/s) and by 40% on Qwen (3.9 to 5.4, the
`i8` one lowest). That is the noise to read a decode cell with. The attention column is steadier.

```sh
LUTE_SET_FFLAGS=DebugLuauBufferVectorLibrary=true,DebugLuauBufferVectorFastcalls=true \
  taskset -c 14,15 $OLELO_FLAG_LUTE run tools/bench/kvlayout.luau models/smollm2-135m-q8v.olelo 128,512,1024,2048 16 5
```

**Decode tok/s, prefill tok/s and the attention of one decode token.** Before is Q8 `i8`, after is Q8 `v3`.

| Model | n | Decode f32 | Decode Q8 before | Decode Q8 after | After over before | Prefill f32 | Prefill Q8 before | Prefill Q8 after | After over before | Attention f32 ms | Attention Q8 before ms | Attention Q8 after ms | After over before |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SmolLM2 | 128 | 8.29 | 7.28 | 8.86 | 1.22 | 13.47 | 13.65 | 14.00 | 1.03 | 4.95 | 21.45 | 5.30 | 4.0 |
| SmolLM2 | 512 | 8.82 | 5.26 | 8.57 | 1.63 | 15.65 | 11.20 | 15.38 | 1.37 | 14.73 | 69.81 | 19.81 | 3.5 |
| SmolLM2 | 1024 | 9.42 | 6.09 | 9.33 | 1.53 | 9.69 | 5.20 | 8.98 | 1.73 | 29.30 | 118.59 | 32.65 | 3.6 |
| SmolLM2 | 2048 | 8.18 | 3.04 | 7.86 | 2.59 | 11.54 | 4.44 | 9.95 | 2.24 | 43.87 | 185.04 | 55.66 | 3.3 |
| Qwen2.5-0.5B | 128 | 4.94 | 4.23 | 5.04 | 1.19 | 6.05 | 5.30 | 6.15 | 1.16 | 2.76 | 15.50 | 3.92 | 4.0 |
| Qwen2.5-0.5B | 512 | 3.65 | 2.48 | 3.50 | 1.41 | 6.64 | 5.46 | 6.81 | 1.25 | 13.71 | 66.63 | 19.14 | 3.5 |
| Qwen2.5-0.5B | 1024 | 3.84 | 3.25 | 4.06 | 1.25 | 3.80 | 2.46 | 3.81 | 1.55 | 24.17 | 122.51 | 33.58 | 3.6 |
| Qwen2.5-0.5B | 2048 | 4.92 | 2.64 | 4.34 | 1.64 | 3.76 | 2.33 | 3.72 | 1.60 | 41.26 | 223.51 | 63.96 | 3.5 |

The absolute rates are below the step estimates of the table above (SmolLM2 with the f32 cache at 128 positions reads 8.3
tok/s here against 14.0 there), because rates drift by 2x between runs, and the f32 variant of the same process is
the reference. The SmolLM2 run had no competing load and the Qwen run a little, none of it on CPUs 14 and 15.

**The attention parts, from `tools/bench/attention_profile.luau`.** The same stages as the table above (dot products, softmax,
whole), now with `kv=q8v` for the vector layout, all three caches in one stream, 7 repeats, two rounds a few minutes
apart, the best of the two. MMAC/s of the dot and value loops are of the `v3` kernel.

| Model | n | f32 ms | Q8 `i8` ms | Q8 `v3` ms | `i8` over `v3` | `v3` over f32 | `v3` scores ms, MMAC/s | `v3` softmax ms | `v3` values ms, MMAC/s | ns per head, position (f32, `i8`, `v3`) |
|---|---|---|---|---|---|---|---|---|---|---|
| SmolLM2 | 128 | 3.5 | 19.5 | 6.1 | 3.2 | 1.76 | 1.7, 1303 | 1.4 | 3.0, 730 | 100, 563, 176 |
| SmolLM2 | 512 | 13.2 | 70.9 | 17.3 | 4.1 | 1.31 | 6.9, 1275 | 1.8 | 8.5, 1039 | 96, 513, 125 |
| SmolLM2 | 1024 | 27.9 | 159.1 | 33.9 | 4.7 | 1.22 | 8.4, 2107 | 4.8 | 20.7, 854 | 101, 575, 122 |
| SmolLM2 | 2048 | 61.9 | 245.0 | 72.7 | 3.4 | 1.17 | 22.9, 1545 | 7.6 | 42.2, 838 | 112, 443, 131 |
| Qwen2.5-0.5B | 128 | 4.6 | 26.5 | 6.0 | 4.4 | 1.31 | 2.0, 1361 | 0.6 | 3.4, 806 | 107, 617, 140 |
| Qwen2.5-0.5B | 512 | 22.4 | 90.4 | 24.7 | 3.7 | 1.10 | 6.7, 1649 | 4.8 | 13.2, 837 | 130, 526, 144 |
| Qwen2.5-0.5B | 1024 | 36.7 | 163.7 | 56.0 | 2.9 | 1.53 | 18.1, 1220 | 4.6 | 33.3, 660 | 107, 476, 163 |
| Qwen2.5-0.5B | 2048 | 98.9 | 440.6 | 103.4 | 4.3 | 1.05 | 37.6, 1172 | 6.7 | 59.2, 744 | 144, 640, 150 |

The two measurements agree that `v3` attention costs 1.05 to 1.55 times the f32 attention (the profile's spread, 1.05 to
1.76, is the noise of single cells) and the `i8` attention 4 to 5.8 times. Absolute times varied by up to 2x between
rounds, so the ratios are the result.

**Quality.** `tools/bench/kvquality.luau q8 48`, SmolLM2 and Qwen2.5-0.5B, the `v3` tree against the `i8` tree, the same 5
prompts (CPUs 6 to 9). The stored integers are the same, so what differs is the order of the sums and the `exp`.

| Model | Layout | Logits error, mean | Logits error, max | Top 1 agreement |
|---|---|---|---|---|
| SmolLM2 | `i8` | 4.42% | 31.1% | 98.3% |
| SmolLM2 | `v3` | 4.40% | 26.2% | 99.2% |
| Qwen2.5-0.5B | `i8` | 5.94% | 53.0% | 95.4% |
| Qwen2.5-0.5B | `v3` | 6.06% | 55.1% | 96.3% |

The differences are inside the noise of the measure (one flip is 0.4 to 0.8% of top 1 agreement), and the centred Q8 cache
stays centred, because the store path subtracts the centre before the bytes are packed, in both layouts.

**Spill guard.** The kernels take three positions a pass. Four positions, like the f32 kernel, need 23 stack slots in the
largest function on the Windows target (`x64_ms`), the limit above which native lowering skips a function and it runs
about 19 times slower, which would leave the guard (`tools/gen/guard.luau`, x64, x64_ms and a64) no margin. Three need 13
to 16 on `x64_ms`, 4 to 5 on `x64` and none on `a64`, for every head size. The largest module is 138 KB (head 256), under
the 199,997 characters of a Roblox ModuleScript.

Reading the numbers:

- **The Q8 cache attention is 2.9 to 4.7 times faster and within 1.05 to 1.55 times of the f32 attention.** Decode at 2048
  positions goes from 3.0 to 7.9 tok/s on SmolLM2 (f32 8.2) and from 2.6 to 4.3 on Qwen2.5-0.5B (f32 4.9), prefill from 4.4 to 10.0
  and from 2.3 to 3.7. At 128 positions the cache is a small part of a token and the gain is 1.2 times. The decode of `v3`
  is within about 12% of the f32 decode in every cell, the noise level of the run, so it is level with it.
- **The dot loop of `v3` runs at 1.2 to 2.1 GMAC/s and the value loop at 0.7 to 1.0** (the f32 kernel: 1.5 to 4.4 and 1.0
  to 2.5 in the table above), against 0.3 to 0.4 for the scalar kernel. Why it does not reach the f32 rate is a hypothesis,
  not measured: the `vector.floor`, subtract and multiply operations that take the bytes apart add work to every lane
  that the f32 kernel does not have.
- **The cost is memory.** A `v3` row is 96 bytes for a head of 64 against 64 for `i8` and 264 for f32, so the cache is 2.6
  times smaller than f32 instead of 3.9 times (SmolLM2: 18,000 B a position against 12,240 and 47,520, 35 MiB against 24 and
  93 at 2048 positions). `kvLayout = "i8"` on the engine keeps the smaller cache and its slower attention. The estimate
  of the load (`plan.estimate`) counts the layout in use.
- **Not measured.** Studio (no Roblox run), the kernels of head sizes 96, 128 and 256 on a real model (they pass the unit
  tests and the spill guard, only the 64 wide kernel was timed), the aarch64 target (guard only), the workers path of the
  Q8 cache with real VMs (the tests pass, no timing).

## Kernel rates

Lute 1.0.1-nightly.20261120 with both vector flags, Luau 0.747, 2026-11-23. MMAC/s is millions of multiply adds per
second, the median of the process medians.

| Kernel | MMAC/s | Range |
|---|---|---|
| `f32_vector/R8S1U8` | 2103 | 1797 to 2602 |
| `f32_buffer/R8S1U8` | 1184 | 815 to 1475 |
| `f32_scalar/R8S1U8` | 1010 | 806 to 1128 |
| `bf16_vector/R4S1U12` | 1111 | 865 to 1658 |
| `bf16_scalar/R8S1U8` | 610 | 493 to 912 |
| `q8_il4e/R8S1U8` | 773 | 698 to 1118 |
| `q4_byte/R8S1U4` | 692 | 594 to 965 |

The rates move by a factor of two between runs on the benchmark machine (the range column), so a difference under 25 percent is not a
finding. The `q8_il4e` row is the first Q8 kernel, kept as a second reference. The packed kernel that replaced it is below.

## Packed Q8 kernel

`tools/bench/kernels.luau`, 15 processes each on CPUs 14 and 15, Lute 1.0.1-nightly.20261120 with both vector flags, 2026-11-23. The
rates varied between runs, a first run of 7 processes gave 2043, 579, 1227 and 1354 MMAC/s in the same order.

| Kernel | MMAC/s | Range |
|---|---|---|
| `q8_p8/R8S1U32` | 2264 | 1853 to 3070 |
| `q8_il4e/R8S1U8` | 898 | 711 to 1205 |
| `f32_vector/R8S1U8` | 2361 | 1641 to 3220 |

The packed kernel is 2.5 times the first Q8 kernel and level with the f32 vector kernel at a quarter of the bytes.

## Prefill tile and the other formats

`tools/bench/kernels.luau`, 1024 x 4096, 9 processes on CPUs 14 and 15 with the vector flags, 2026-11-23. Two tokens per
call for the first block, one for the second. The sessions differ, so compare inside a block.

| Kernel | Tokens | MMAC/s | Range |
|---|---|---|---|
| `q8_p8/R8S1U32` | 2 | 2003 | 1420 to 2913 |
| `q8_p8/R4S2U32` | 2 | 2517 | 1964 to 3377 |
| `f32_vector/R8S1U8` | 2 | 2559 | 2130 to 3218 |
| `f32_vector/R4S2U2` | 2 | 1859 | 1670 to 2573 |
| `q8_p8/R8S1U32` | 1 | 2174 | 1962 to 2628 |
| `bf16_vector/R4S1U8` | 1 | 1053 | 952 to 1181 |
| `bf16_vector/R4S1U12` | 1 | 1113 | 1028 to 1535 |
| `q4_byte/R8S1U4` | 1 | 790 | 607 to 983 |
| `f32_vector/R8S1U8` | 1 | 2258 | 1932 to 2632 |

The 4 row, 2 token Q8 tile is 1.26 times the 8 row tile on two tokens, because the field extraction from each weight word
serves both tokens. BF16 and Q4 run at half the f32 rate or less.

## Q4 and BF16 quality

SmolLM2-135M-Instruct, "The capital of France is", RMS error of the last position's logits, 2026-11-23:

| Pair | RMS error |
|---|---|
| Olelo BF16, Hugging Face f32 | 2.5e-6 |
| Olelo Q4, Hugging Face f32 | 7.6e-1 |
| llama.cpp Q4_0, Hugging Face f32 | 7.7e-1 |
| Olelo Q4, llama.cpp Q4_0 | 3.4e-2 |

SmolLM2's weights are BF16, so the BF16 container is lossless. The Q4_0 layer weights and scales equal `llama-quantize`
in all 210 matrices, and the token embedding is Q8_0 in both. Q4_0 on a 135 M model loses a lot against f32 in either
engine, so Q4 is checked against llama.cpp and against a reference run on the dequantized weights
(`tests/fixtures/tiny-llama/ref-q4`), not against f32.

## Q8 against f32, SmolLM2-135M-Instruct

`tools/bench/generate.luau`, 119 token prompt, 64 new tokens, medians of 3 runs, two processes on CPUs 14 and 15 with the
vector kernels (a third pair is left out, its probe fell back to scalar kernels under load).

| Container | Size | Prefill tokens/s | Decode tokens/s | First token |
|---|---|---|---|---|
| f32 | 539 MB | 15.4, 17.1 | 10.5, 10.5 | 7.8 s, 7.0 s |
| Q8 | 152 MB | 15.8, 18.6 | 11.8, 12.8 | 7.5 s, 6.4 s |

Decode is 12 to 22 percent faster. The output matrix is 28 M of the 135 M weights, and the ops between the products (norm,
rope, attention, activation preparation) are the same, so the end to end gain is smaller than the kernel gain.

## Q8 quality

SmolLM2-135M-Instruct, the prompt "The capital of France is", RMS error of the last position's logits, 2026-11-23:

| Pair | RMS error |
|---|---|
| Olelo f32, Hugging Face f32 | 2.5e-6 |
| Olelo Q8, Hugging Face f32 | 3.8e-2 |
| llama.cpp Q8_0, Hugging Face f32 | 5.6e-2 |
| Olelo Q8, llama.cpp Q8_0 | 6.0e-2 |

The Q8_0 weights and scales of the two are the same bytes in all 211 matrices. The two Q8 runs differ by about as much as
each differs from f32, so the activation rounding noise of this small model is not reproduced between engines. The
greedy continuation of the 6 reference prompts matches the f32 one for the 64 tokens on two prompts and up to the first near tie
on the others (steps 4, 27, 39 and 63).

## Generation

`tools/bench/generate.luau` answers one 119 token prompt with 64 new tokens, greedy, three times per process, on SmolLM2-135M-Instruct
in f32 with the vector kernels. Three processes on CPUs 14 and 15 under the same Lute as above, 2026-11-23, medians
of three runs each:

| Process | Prefill tokens/s | Decode tokens/s | First token |
|---|---|---|---|
| 1 | 17.2 | 11.0 | 6.9 s |
| 2 | 17.8 | 11.0 | 6.7 s |
| 3 | 18.1 | 11.7 | 6.6 s |

The product kernels alone would allow about 15 decode tokens per second at 2.1 GMAC/s for 134 M multiply adds, so the
ops between them (norm, rope, attention, the gated activation) take roughly a quarter of a decode step. That split is
an estimate from the rates, not a profile.

## Parity

`tests/parity/real.test.luau` against the Hugging Face model in float32, all six prompts, with the vector kernels and
with the scalar fallback:

| Quantity | Measured | Limit |
|---|---|---|
| hidden state after each layer, first 16 positions, max error relative to the row RMS | 1.4e-4 | 3e-3 |
| logits of the last prompt position, max error relative to the RMS | 9e-6 | 1e-3 |
| greedy tokens over 64 steps | equal on every prompt | equal except where the two best logits differ by under 1e-3 |

## Tensor parallel in Lute

Lute 1.0.1-nightly.20261120 with the fix for the `vm.create` crash (Luau 0.748, `tools/lute-fix/README.md`) and the flags
set, CPUs 0 to 13, 2026-11-23. `tools/bench/parallel.luau` for L2 to L4 and L6, `tools/bench/generate.luau` with a
worker count for L5. The rates differ by up to 2 times between runs of the
same command. The ratios
inside one table are from one session, the runs of one row were not interleaved with the others.

**L2, one exchange** (a request of the given size to every worker and the join of the replies, median of 5 processes of
9 blocks of 300, microseconds):

| Workers | 0 B | 2304 B | 16 KiB |
|---|---|---|---|
| 1 | 35 | 43 | 58 |
| 2 | 58 | 59 | 97 |
| 4 | 113 | 117 | 128 |
| 8 | 188 | 204 | 235 |

The cost grows by about 22 us per worker, since the caller sends one request after the other. SmolLM2-135M has 61
exchanges per token, which at 4 workers is 7 ms. The cost model used 31 to 40 us for one worker, which this matches.

**L3 and L4, compute and bandwidth.** Every worker runs one kernel over a matrix of its own at the same time, 3072 x 2048
for Q8 (7 MiB) and 2048 x 2048 for f32 (16 MiB), several copies each so the weights stream from memory. Aggregate rates,
median of 5 processes of 7 repeats:

| Workers | Q8 MMAC/s | Q8 scaling | Q8 GB/s | f32 MMAC/s | f32 GB/s |
|---|---|---|---|---|---|
| 1 | 2952 | 1.00 | 3.3 | 3003 | 12.1 |
| 2 | 4990 | 1.69 | 5.6 | 4477 | 18.1 |
| 4 | 7701 | 2.61 | 8.7 | 5804 | 23.4 |
| 8 | 10972 | 3.72 | 12.3 | 6541 | 26.4 |

Compute scales to 0.85 of linear at 2 workers, 0.65 at 4 and 0.46 at 8, which is below the 0.7 the cost model took as the
limit at 4 and 8. The f32 kernel is bound by memory: its aggregate bandwidth stops at about 26 GB/s, 2.2 times the single
thread rate. Q8 reads 3.6 times fewer bytes per MAC, so it keeps scaling and is the format for parallel runs.

**L6, flags in workers.** The probe finds the vector read fast inside the worker VMs (4 of 4 workers). With 8 workers
probing at once 6 of 8 did, since concurrent probes time each other. `spawnWorkers` therefore passes the main VM's probe
result to the workers.

**L5, decoding with workers.** `generate.luau`, a 119 token prompt, 48 new tokens, greedy, 3 runs per process,
`workers=0` is the single VM, median decode tokens per second:

| Model | 0 | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M Q8 | 14.4 | 13.2 | 19.4 | 25.0 | 24.7 | 22.5 | 24.6 |
| SmolLM2-135M f32 | 13.5 | 10.1 | 16.7 | 11.8 | 18.9 | 6.8 | 19.9 |
| Qwen2.5-0.5B Q8 | 3.1 | 3.4 | 6.8 | 9.9 | 11.2 | 11.4 | 12.0 |

Prefill rises the same way, SmolLM2 Q8 from 22.6 to 50.4 tokens per second at 8 workers and Qwen2.5-0.5B Q8 from 6.1 to
24.2. The f32 row is noisy, its runs at 2, 4 and 8 workers range from 5.1 to 22.4, because the probe of some runs found
the scalar fallback and the CPU load was higher.

Q8 decoding of Qwen2.5-0.5B is 3.7 times faster at 8 workers and 3.6 times at 4, which is close to the cost model
(compute over 4 at 0.65 efficiency plus 49 exchanges at 117 us). SmolLM2-135M gains 1.7 times at 3 to 4 workers and
then stops, since its exchanges (61 per token) and the work of the main worker are a larger part of a step. At 8 workers
the workers' own time per op no longer falls (`PAR` line of `generate.luau`), which matches the 0.46 scaling of L3.
The first-token time of Qwen2.5-0.5B falls from 19 s to 4.8 s.

## Serving

Lute 1.0.1-nightly.20261120 with the vector flags, 2026-11-23, on the hardware of the sections above. The context runs are `tools/bench/context.luau` pinned to CPUs 14 and
15 (medians of 3 repeats of 16 decode passes), one run per cache format. The f32 run had no competing CPU load and the
Q8 run had some, so the Q8 rows had the worse of the two conditions. The replica, concurrent chat and quality runs are on CPUs 0 to 13.

### Context length, SmolLM2-135M Q8 and Qwen2.5-0.5B Q8

These rows ran before the unrolled vector attention and hold for the old attention kernels only. The rerun follows this
section, and [Attention by context length](#attention-by-context-length) has the parts.

**Decode** after `n` positions, tokens per second, and the attention op alone for one token over every layer, in ms.
Scores, softmax and values are the three loops of the f32 attention timed separately, their sum is within 7% of the whole.

| Model | n | f32 tok/s | Q8 KV tok/s | attention f32 | attention Q8 | scores / softmax / values (f32) | KV f32 MiB | KV Q8 MiB |
|---|---|---|---|---|---|---|---|---|
| SmolLM2 | 0 | 17.5 | 13.7 | 0.4 | 0.2 | | | |
| SmolLM2 | 128 | 11.2 | 9.3 | 16.6 | 17.9 | 11.0 / 1.3 / 9.3 | 5.8 | 1.5 |
| SmolLM2 | 512 | 4.9 | 5.3 | 93.1 | 61.1 | 51.3 / 5.2 / 37.9 | 23.2 | 6.0 |
| SmolLM2 | 1024 | 3.4 | 4.8 | 170.5 | 130.6 | 109.2 / 9.8 / 85.6 | 46.5 | 12.0 |
| SmolLM2 | 2048 | 2.6 | 2.7 | 275.1 | 281.7 | 151.4 / 14.8 / 128.0 | 92.9 | 23.9 |
| Qwen2.5-0.5B | 0 | 3.2 | 3.2 | 0.7 | 0.2 | | | |
| Qwen2.5-0.5B | 128 | 2.8 | 2.5 | 25.3 | 17.4 | 13.3 / 1.4 / 11.0 | 3.1 | 0.8 |
| Qwen2.5-0.5B | 512 | 2.9 | 2.5 | 96.9 | 46.8 | 52.7 / 5.4 / 41.2 | 12.4 | 3.2 |
| Qwen2.5-0.5B | 1024 | 1.6 | 2.5 | 218.2 | 120.1 | 99.5 / 9.6 / 80.4 | 24.8 | 6.4 |
| Qwen2.5-0.5B | 2048 | 1.3 | 1.3 | 436.6 | 309.9 | 231.8 / 25.1 / 181.5 | 49.5 | 12.8 |

The decode rates carry about 15% of noise (the n = 512 row of Qwen is faster than its n = 128 row at f32), the attention
column is steadier. Reading the table:

- **These attention numbers are from before the unrolled vector attention.** The f32 attention is 5 to 13 times faster
  now and the Q8 cache one is not, see [Attention by context length](#attention-by-context-length).
- **The extra time is the attention loops.** The attention op costs 0.13 to 0.18 ms per position on SmolLM2 and 0.19 to 0.21
  ms on Qwen with the f32 cache, at every length from 128 to 2048, so it is linear in `n` with no step where the cache
  leaves the 16 MB L3. At 2048 it is 275 ms of the 385 ms of a SmolLM2 token and 437 ms of the 773 ms of a Qwen token. The
  scores loop takes 52% of it, the value loop 43 to 47% and the softmax 5%.
- **It is not memory traffic.** At 2048 SmolLM2 reads 92.9 MiB of cache in 275 ms, 0.35 GB/s, against 16.6 GB/s of copy
  bandwidth for one process (`tools/bench/membw.luau`, [below](#memory-copy-bandwidth)). The loops are bound by their own
  instructions. The Q8 cache reads 3.9 times fewer bytes and its attention is 0 to 52% faster, not 3.9 times, so the bytes
  are a part of the cost that was not isolated. Q8 is a memory saving first. Its attention takes 23 to 34% less time on
  SmolLM2 at 512 and 1024 positions and 29 to 52% less on Qwen2.5-0.5B from 512 to 2048, and 8% more at 128 on SmolLM2,
  one run each. The decode rate follows by 8 to 55% at 512 to 1024 and is equal at 2048, inside the noise at 128.

**Prefill** in passes of 32 tokens, tokens per second of the window that ends at `n`, and the seconds to prefill `n` tokens.
The attention share is the attention op over one chunk (`attn_chunk`) against the time of that chunk, f32 cache.

| Model | n | f32 window tok/s | Q8 KV window tok/s | f32 total s | Q8 KV total s | attention share of a chunk |
|---|---|---|---|---|---|---|
| SmolLM2 | 128 | 15.8 | 14.7 | 8.1 | 8.7 | 24% |
| SmolLM2 | 512 | 9.3 | 9.3 | 41.8 | 44.9 | 83% |
| SmolLM2 | 1024 | 4.7 | 5.7 | 135.6 | 128.1 | 82% |
| SmolLM2 | 2048 | 2.7 | 3.3 | 440.1 | 416.4 | 76% |
| Qwen2.5-0.5B | 128 | 5.0 | 5.4 | 25.8 | 23.8 | 11% |
| Qwen2.5-0.5B | 512 | 3.5 | 3.4 | 129.2 | 117.3 | 35% |
| Qwen2.5-0.5B | 1024 | 2.4 | 3.0 | 320.0 | 284.3 | 45% |
| Qwen2.5-0.5B | 2048 | 1.6 | 2.0 | 861.2 | 714.6 | 65% |

A 119 token prompt takes 7.5 s on SmolLM2 and 24 s on Qwen to the first token at the n = 128 window rates. Prefill is bound by the
weight kernels at the start and by attention later: the attention op of one 32 token chunk at 2048 costs 9125 ms, 285 ms a
token, which is what a decode token's attention costs (275 ms), so a chunk gets nothing from sharing its keys and values
across its rows, and prefill falls to the decode rate (2.7 against 2.6 tokens per second). The weight work per token is
the same at every length. The attention share of a chunk passes 75% from 512 positions on for SmolLM2, and for
Qwen2.5-0.5B, whose weights cost more per token, it is 45% at 1024 and 65% at 2048.

### Context length after the vector attention

The same benchmark with the unrolled vector attention (f32 cache) and the scalar `attention_q8` (Q8 cache), CPUs 14 and
15, one run per cache format. A first Q8 pass (SmolLM2 4.3 tok/s at n = 0 against 13.0 on f32) ran with competing CPU load and is
not used. The decode rates carry about 25% of noise (Qwen f32 reads 3.2 tok/s at 128 positions and 2.4
at 0), so the attention column, which is steadier, carries the comparison. The score, softmax and value parts of the
`context.luau` output replicate the old loops and are not shown.

| Model | n | f32 decode tok/s | Q8 KV decode tok/s | attention f32 ms | attention Q8 ms | prefill f32 tok/s | prefill Q8 tok/s |
|---|---|---|---|---|---|---|---|
| SmolLM2 | 128 | 8.2 | 8.0 | 3.2 | 12.2 | 15.1 | 14.1 |
| SmolLM2 | 512 | 9.4 | 7.2 | 11.3 | 52.4 | 14.3 | 8.8 |
| SmolLM2 | 1024 | 8.7 | 4.5 | 21.7 | 162.1 | 11.6 | 5.6 |
| SmolLM2 | 2048 | 6.6 | 2.5 | 50.4 | 268.2 | 7.4 | 3.0 |
| Qwen2.5-0.5B | 128 | 3.2 | 3.0 | 3.3 | 17.6 | 3.2 | 5.1 |
| Qwen2.5-0.5B | 512 | 2.5 | 3.1 | 25.5 | 66.0 | 3.6 | 3.6 |
| Qwen2.5-0.5B | 1024 | 1.9 | 1.9 | 50.9 | 198.3 | 3.3 | 3.1 |
| Qwen2.5-0.5B | 2048 | 1.5 | 1.6 | 101.1 | 258.9 | 2.2 | 2.2 |

- **The f32 cache is faster than the Q8 cache now.** On SmolLM2 the f32 attention is 3.8 to 7.5 times faster and decode at
  2048 positions is 6.6 against 2.5 tok/s. The earlier 23 to 52% gain of the Q8 cache is gone because only the f32
  attention got a vector path. Qwen2.5-0.5B decode is level at 1024 and 2048 positions, since its weights cost 250 ms
  of a 650 ms token.
- **Prefill** of SmolLM2 at 2048 positions is 7.4 tok/s with the f32 cache against 3.0 with Q8, and one chunk of 32 tokens
  spends 1.6 s in attention against 7.9 s. The weight kernels now bound a short prompt, and attention bounds a long one
  only on the Q8 cache.
- **The Q8 cache is a memory saving, 3.9 times fewer bytes per position, at a speed cost until its attention kernel is
  rewritten.** The profile of its loops and the estimated gain of a fast kernel are in
  [Attention by context length](#attention-by-context-length). The kernel was rewritten for the vector layout of the cache
  (1.5 times the bytes of the int8 rows), and the numbers of this table for the Q8 cache no longer hold, see
  [Q8 cache attention in the vector layout](#q8-cache-attention-in-the-vector-layout).

### Q8 KV cache quality

`tools/bench/kvquality.luau`: each prompt runs twice, the candidate follows the reference's greedy tokens for the logits
error (RMS of the difference over the RMS of the reference, mean over positions) and the top 1 agreement, and runs free
for the greedy continuation. 48 steps per prompt, 5 prompts for SmolLM2 (240 positions) and 4 for Qwen (192).
`fq` variants round the f32 cache in place to simulate a format. One flip is 0.4 to 0.5% of top 1 agreement, and the
free continuation diverges at the first flip, so the logits error and the top 1 agreement are the measures.

**f32 weights**, so only the cache is rounded:

| Variant | SmolLM2 error | SmolLM2 top 1 | Qwen2.5-0.5B error | Qwen top 1 |
|---|---|---|---|---|
| 16 bit round trip (floor of the harness) | 0.008% | 100% | 0.05% | 100% |
| Q8, a scale per head and token | 2.18% | 99.6% | 12.82% | 95.3% |
| Q8 with centred keys (the shipped cache) | 1.77% | 99.2% | 5.10% | 96.4% |
| Q8 keys only, per head and token | not run | not run | 12.57% | 93.8% |
| Q8 values only, per head and token | not run | not run | 0.65% | 100% |
| Q8 centred, fake (both) | 1.86% | 99.2% | 4.75% | 97.9% |
| Q8, a scale per 32 values, fake | 1.19% | 99.2% | 9.57% | 94.3% |
| Q4, a scale per head and token, fake | 51.9% | 79.2% | 96.8% | 42.2% |
| Q4, a scale per 32 values, fake | 29.2% | 92.5% | not run | not run |
| Q4, minimum and maximum per 32 values, fake | 16.2% | 92.9% | not run | not run |

**Q8 weights**, the models the speed tables use. Q8 activations alone put a floor of 3 to 4% on the logits error (the 16 bit
row), so the cache is judged against that:

| Variant | SmolLM2 error | SmolLM2 top 1 | Qwen2.5-0.5B error | Qwen top 1 |
|---|---|---|---|---|
| 16 bit round trip (the floor) | 3.92% | 96.3% | 3.03% | 99.0% |
| Q8, before centring | 4.59% | 98.3% | 13.21% | 95.3% |
| Q8 with centred keys (the shipped cache) | 4.58% | 97.5% | 5.82% | 95.3% |

What follows:

- **The error of Qwen2.5-0.5B is in its keys.** Keys alone give 12.6% and values alone 0.65%. Its key biases put a large
  offset in a few channels of every key, so the largest value of a row sets a step that rounds the other channels away.
  A scale per 32 values does not cure it (9.6%). Subtracting the mean of the first 32 keys, which the softmax does not see,
  does (4.7% fake, 5.1% shipped). SmolLM2 has no such offset and gains little (2.2% to 1.8%).
- **The shipped Q8 cache is near the Q8 weight floor on SmolLM2** (4.58% against 3.92%, top 1 agreement 97.5% against
  96.3%, both inside the 0.4% a flip moves it) and 1.9 times above it on Qwen2.5-0.5B (5.82% against 3.03%).
- **Q4 is not built.** The best of three layouts loses 16% of the logits and 7% of the top 1 agreement on SmolLM2, and the
  plain one 97% on Qwen. Q8 costs 12,240 bytes a position on SmolLM2 against 6,480 for Q4, a saving that does not pay for
  that.
- **A scale per 32 values is not built.** It halves the error on SmolLM2 (1.19% against 2.18%) for 72 bytes a row instead of
  68, and does nothing for the key offset. Combining it with centring was not measured.

### Replica scaling

`tools/bench/replicas.luau`: N replica VMs behind the scheduler, one chat each, the same 119 token prompt, 48 new tokens past
the stop token, 3 rounds, median. Two complete passes, 13 minutes apart, the sum of the chats' decode rates:

| Replicas | SmolLM2 pass 1 | SmolLM2 pass 2 | per replica | Qwen pass 1 | Qwen pass 2 | per replica | SmolLM2 peak MiB | Qwen peak MiB |
|---|---|---|---|---|---|---|---|---|
| 1 | 8.5 | 8.5 | 8.5 | 4.3 | 3.4 | 3.4 to 4.3 | 265 | 698 |
| 2 | 19.1 | 18.3 | 9.1 to 9.6 | 7.4 | 5.6 | 2.8 to 3.7 | 518 | 1375 |
| 3 | 28.0 | 23.5 | 7.8 to 9.4 | 5.8 | 6.2 | 1.9 to 2.1 | 765 | 2058 |
| 4 | 38.2 | 22.3 | 5.2 to 9.6 | 7.1 | 6.5 | 1.6 to 1.8 | 1014 | 2733 |
| 5 | 38.3 | 31.3 | 6.1 to 7.7 | 11.4 | 8.7 | 1.7 to 2.4 | 1254 | 3417 |
| 6 | 37.0 | 36.2 | 5.9 to 6.3 | 14.1 | 10.5 | 1.7 to 2.4 | 1508 | 4095 |
| 8 | 49.6 | 41.1 | 4.9 to 6.3 | 15.9 | 14.8 | 1.8 to 2.0 | 2012 | 5466 |
| 10 | 60.5 | 43.5 | 3.6 to 6.3 | 20.7 | 17.6 | 1.7 to 2.3 | 2513 | 6815 |

- **Aggregate speed rises with replicas but not in proportion**: 10 replicas give 7.1 (pass 1) and 5.2 (pass 2) times one
  on SmolLM2 and 4.8 and 5.2 times on Qwen2.5-0.5B. The two passes differ by up to 1.7 times at the same count (4
  replicas, 38.2 against 22.3), so the numbers say the
  order of magnitude and the shape, not the second digit.
- **Memory is linear**: a replica adds about 250 MiB (SmolLM2) and 680 MiB (Qwen2.5-0.5B) to the process, the slope between 1
  and 10 replicas in both passes. The engine's estimate for a 512 position context is 169 MiB (145 of weights, 23 of f32
  cache) and 547 MiB (534 and 12), so a VM takes about 81 and 133 MiB beyond it. `layout.luau` takes 150 MiB.
- **What bounds it is not settled.** At 10 replicas the weights stream at 6.6 to 9.2 GB/s (SmolLM2, 152.5 MB per token) and
  9.9 to 11.6 GB/s (Qwen, 559.5 MB), against 17 to 22 GB/s of copy traffic (read plus write) for 1 to 7 processes. That is
  30 to 55% and 45 to 70% of the copy figure, which a read only stream could exceed, so memory bandwidth is not shown to be
  the limit for either model. The 14 threads of CPUs 0 to 13 are 7 cores, and two replicas on one core's two threads
  probably each run slower than one alone. That is a hypothesis, not measured (`perf stat -e instructions:u,cycles:u` per
  count would settle it).

#### Memory copy bandwidth

`tools/bench/membw.luau`, N processes started together on CPUs of different cores first, each copying 128 MB onto another
buffer for 4 s, memory traffic counted as read plus write. Sum over the processes, GB/s: 1 process 16.6, 2 processes 16.9,
4 processes 21.9, 7 processes 20.7, 10 processes 32.5, 14 processes 40.6 (a single run). The runs
with 10 and 14 processes have two or three processes at 8 to 9 GB/s and the rest at 1.5 to 3, so they did not copy in step
and the sums are not a ceiling. The figure used above is 17 to 22 GB/s from 1 to 7 processes.

### The scheduler against the engine alone

One chat through one replica behind the scheduler against `generate.luau` on the same two CPUs, interleaved, 5 rounds,
48 tokens, decode tokens per second: SmolLM2 direct 8.8 to 14.2 (median 11.5), through the scheduler 9.9 to 15.1 (10.9);
Qwen2.5-0.5B direct 2.2 to 3.4 (2.9), scheduler 2.5 to 4.0 (2.8). Prefill: SmolLM2 11.0 to 21.5 (16.1) against 15.5 to 23.7
(18.0), Qwen 5.0 to 6.0 (5.7) against 3.7 to 5.9 (5.0). The two are inside each other's spread (rates varied between rounds):
the scheduler, the VM transport and the copy of the text per step cost nothing measurable per token.

### More chats than lanes

`tools/bench/users.luau`: `users` chats submit the 119 token prompt together to `lanes` replicas of `slots` conversations,
32 new tokens each, 3 rounds, median. The aggregate counts all new tokens over the seconds to the last reply, prefill
included, so it is far below the decode sums above (a prefill is 7 to 24 s). Jain's index is over the end times of the
chats.

| Model | Chats | Lanes | Slots | Aggregate tok/s | First token median / max s | First end / last end s | Jain |
|---|---|---|---|---|---|---|---|
| SmolLM2 | 4 | 4 | 1 | 7.0 | 12.0 / 12.2 | 17.7 / 18.4 | 1.00 |
| SmolLM2 | 8 | 8 | 1 | 15.7 | 10.9 / 11.2 | 16.1 / 16.4 | 1.00 |
| SmolLM2 | 8 | 4 | 1 | 10.3 | 8.7 / 20.5 | 12.0 / 24.9 | 0.90 |
| SmolLM2 | 8 | 4 | 2 | 9.3 | 19.1 / 19.9 | 26.9 / 27.7 | 1.00 |
| SmolLM2 | 16 | 4 | 4 | 8.7 | 40.3 / 42.2 | 56.5 / 58.9 | 1.00 |
| Qwen2.5-0.5B | 4 | 4 | 1 | 3.5 | 23.2 / 23.4 | 36.5 / 36.9 | 1.00 |
| Qwen2.5-0.5B | 8 | 8 | 1 | 4.9 | 34.0 / 34.9 | 51.6 / 52.8 | 1.00 |
| Qwen2.5-0.5B | 8 | 4 | 2 | 3.6 | 45.7 / 49.3 | 68.3 / 71.7 | 1.00 |

Chats sharing a lane (slots above 1) are served in turns by lane seconds, so they finish together (Jain 1.00) and all wait
the same long time for the first token. With one slot the second chat of a lane waits for the first to end (8 chats on 4
lanes: first tokens at 8.7 s for the median and 20.5 s for the slowest, Jain 0.90), the queue works as first come first
served. More slots does not raise the aggregate (9.3 against 10.3). The lane is busy either way and prefill, which
dominates these runs, does not get faster by interleaving (a reading of the numbers, not isolated). The slots are for
memory-bound pools and for keeping idle conversations resident, not for speed.

### Batching decode steps

`tools/bench/batch.luau`: B tokens through `graph.forward` in one pass (the prefill tile kernels) at a context of n, no
logits, plus B head passes, against one single token step. Per token milliseconds of the batched pass over
the single step, below 1 is a win; 5 repeats, sizes interleaved, CPUs 14 and 15:

| Model | n | B = 1 | B = 2 | B = 4 | B = 8 | B = 32 |
|---|---|---|---|---|---|---|
| SmolLM2 | 128 | 0.84 | 0.78 | 0.87 | 0.82 | 0.81 |
| SmolLM2 | 512 | 1.02 | 1.08 | 1.01 | 1.01 | 1.05 |
| Qwen2.5-0.5B | 128 | 1.61 | 1.38 | 1.16 | 1.09 | 1.27 |
| Qwen2.5-0.5B | 512 | 1.08 | 0.85 | 0.97 | 1.01 | 0.93 |

B = 1 should read 1.0, so the spread of the column, 0.84 to 1.61, is the noise of the measurement, and no column falls
with B beyond it. The pass time per token without the head is flat in B on SmolLM2 (n = 128: 113.9, 103.1, 117.6, 110.6,
108.6 ms for B = 1, 2, 4, 8, 32): the tile kernels do not make a token cheaper when more tokens share the weights, since
the kernels are bound by their own instructions and not by reading the weights (above). A batched decode step would
therefore not win, and it was not built.

### Sampler

Milliseconds per token over 151,936 random logits in [-8, 8] (the flat worst case), temperature 0.8, median of 9 calls,
CPUs 14 and 15, before and after the rewrite described below. One run each:

| Settings | before | after |
|---|---|---|
| greedy | 1.4 | 1.4 |
| temperature only | 12.5 | 4.5 |
| top k 40 | 170.4 | 2.6 |
| top p 0.9 | 209.6 | 11.6 |
| top k 40, top p 0.9, repetition penalty 1.1 | 168.4 | 2.1 |
| min p 0.05 | 186.0 | 3.4 |

The old sampler sorted the candidates. The new one selects the best 40 or 256 with a bounded insertion, and a top p or min p
that needs more of a flat vocabulary sums the mass into 1024 buckets of the logit range and sorts only the bucket where top
p is reached. An intermediate version that grew its head with sorted insertion took 4467 ms for top p 0.9 here.
