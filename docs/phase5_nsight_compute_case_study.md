# Phase 5: Nsight Compute performance forensics

This stage treats optimization as an evidence loop:

```text
ordinary runtime -> hardware symptom -> source location -> causal hypothesis
                 -> one controlled change -> ordinary runtime verdict
                 -> same counters explain the verdict
```

Test platform: NVIDIA GeForce RTX 5060 Ti, compute capability 12.0, 36 SMs.
The main comparison uses `M=N=K=4096` and configuration 21:

```text
threads=128, BM=128, BN=128, BK=16,
WM=128, WN=32, WNITER=2, TM=4, TN=8
```

## Case 1 — K11 double buffering lost to K10

Ordinary execution established the result first:

| Kernel | Time | Throughput |
|---|---:|---:|
| K10 warp tiling | 9.064 ms | 15,163.3 GFLOP/s |
| K11 split double buffering | 10.702 ms | 12,842.4 GFLOP/s |

K10 and K11 both used 168 registers per thread and had nearly identical
achieved occupancy (24.55% versus 24.38%). Therefore a register or occupancy
regression did not explain the slowdown.

The schedule changed, however. K11 split four resident warps into two groups:
one group computed while the other loaded, with block-wide barriers joining
the paths. Barrier stall rose from `0.115` to `1.326`, eligible warps per
scheduler fell from `1.669` to `1.143`, issue active fell from `72.36%` to
`59.27%`, and FMA activity fell from `66.56%` to `58.46%`.

This proves that K11 lost issue opportunities at synchronization points. It
does not, by itself, identify which branch arrived late.

PC sampling at the first overlap resolved the direction:

```text
compute-path barrier waiter:  15,742 samples
load-path barrier waiter:     90,349 samples
```

The load group reached the barrier early and accumulated waiting samples; the
compute group arrived late. For this tile, the attempted overlap removed half
the warps from the current compute segment, while the hidden load work was too
small to repay that loss.

## Controlled experiments

### More resident warps

Configuration 38 increased the block to 256 threads. K11 occupancy rose to
32.97% and barrier ratio fell to `1.19`, so extra warps did hide part of the
wait. Ordinary K11 time nevertheless changed from 10.702 ms to 10.869 ms.
The mechanism improved, but the kernel regressed by 1.6%; the optimization was
rejected.

### 3:1 compute/load split

Giving three warps to compute and one warp to loading reversed the late path:

```text
compute-path barrier waiter: 185,846 samples
load-path barrier waiter:         84 samples
```

The lone loader became the late arrival. Ordinary time rose to 14.969 ms.
Changing the split moved the bottleneck rather than removing it.

## Case 2 — K12 asynchronous copy

K12 did execute `LDGSTS` asynchronous global-to-shared instructions, but
“asynchronous” only removed the immediate data dependency; issuing requests
still consumed instruction and MIO capacity.

The baseline had only `0.20` eligible warps per scheduler, `13.42%` issue
active, MIO throttle `11.51`, and long scoreboard `7.87`. A paced-copy
experiment reduced MIO throttle to `8.68` and long scoreboard to `6.14`, but
left the number of `LDGSTS` instructions unchanged at 16,777,216 while total
instructions increased from 3,838,124,066 to 4,455,495,483 (+16.1%).
Ordinary time remained effectively unchanged: 79.733 ms versus 79.828 ms.

The controlled change improved the targeted counters but added enough loop and
address bookkeeping to cancel the gain. It was rejected and remains available
only as an opt-in reproduction path.

## Case 3 — K10 register double buffering

K10's low eligible-warp count suggested that more independent register work
might hide shared-memory dependencies. The experiment slightly reduced short
scoreboard (`0.514` to `0.48`) but raised registers per thread from 168 to 182.
That crossed a residency boundary:

```text
resident blocks/SM:              3 -> 2
active warps/scheduler:      2.943 -> 1.99
eligible warps/scheduler:    1.669 -> 1.01
issue active:               72.36% -> 57.33%
FMA active:                 66.56% -> 54.74%
ordinary time:              9.064 ms -> 10.577 ms
```

The local dependency improvement was real, but the register cost removed an
entire resident block. This experiment was also rejected.

## Transferable diagnosis model

The decisive object is not occupancy alone. A resident warp may be active yet
unable to issue. Follow the execution chain:

```text
resident warps
  -> eligible warps
  -> scheduler issue activity
  -> productive pipeline activity
  -> ordinary runtime
```

When eligible warps collapse, identify the blocking state, map it back to the
specific source path, then change only the suspected mechanism. A counter
moving in the intended direction supports the mechanism; only ordinary runtime
decides whether the optimization succeeded.

All extracted values are available in
[`../reports/phase5/metrics.csv`](../reports/phase5/metrics.csv).

