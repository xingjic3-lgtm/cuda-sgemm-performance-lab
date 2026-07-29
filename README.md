# CUDA SGEMM Performance Lab

A CUDA performance-engineering project built around single-precision matrix
multiplication:

```text
C = alpha * A * B + beta * C
```

The project follows the kernel from source code to GPU execution: memory
coalescing, shared-memory tiling, vectorized access, warp tiling, compile-time
parameter search, shape-aware dispatch, and Nsight Compute diagnosis.

The final result is not presented as one universally fastest kernel. It is a
reproducible workflow that separates:

- correctness from performance;
- parameter search from runtime dispatch;
- correlated profiler counters from source-level causes;
- locally improved counters from actual end-to-end speedup.

## What was built

### Correctness for arbitrary shapes

The benchmark accepts arbitrary positive `M`, `N`, and `K`, not only aligned
square matrices. Kernel 10 selects a bounds-checked fallback when a shape does
not fit a specialized tile.

Every custom kernel is checked against cuBLAS before timing. Deterministic
random-shape tests use a fixed seed so a failure is reproducible.

### Compile-time parameter search

Kernel 10 exposes the main tiling parameters:

```text
NUM_THREADS, BM, BN, BK, WM, WN, WNITER, TM, TN
```

The search pipeline rejects illegal candidates before compilation by checking
warp/block coverage, 32-thread warp mapping, vectorized load divisibility, tile
divisibility, and shared-memory capacity.

A reproducible 48-candidate sample was benchmarked on six square shapes. The
winning configurations became separate CUDA template specializations.

### Shape-aware runtime dispatch

Runtime code selects only configurations that were actually measured:

| Matrix shape | Kernel 10 specialization |
|---:|---:|
| 128 x 128 x 128 | config 29 |
| 256 x 256 x 256 | config 11 |
| 512 x 512 x 512 | config 29 |
| 1024 x 1024 x 1024 | config 38 |
| 2048 x 2048 x 2048 | config 5 |
| 4096 x 4096 x 4096 | config 21 |

An unmeasured or rectangular shape uses a correctness-safe general
configuration. The dispatcher does not assume that the nearest measured shape
has the same optimal kernel.

### Nsight Compute performance forensics

Parameter search had already found the best measured `4096^3` configuration
before the Nsight Compute stage. Profiling did not produce a new peak result.
Its role was to explain why later optimization attempts failed.

The analysis followed one evidence chain:

```text
ordinary runtime
  -> resident and eligible warps
  -> scheduler issue activity
  -> dominant stall state
  -> exact source path
  -> one controlled code change
  -> ordinary runtime verdict
```

This produced three concrete conclusions:

1. K11's split double-buffer schedule increased barrier waiting. PC sampling
   showed that the loading group arrived first and waited for the compute
   group. Changing the group ratio moved the late path to the loader instead
   of removing the imbalance.
2. K12 executed asynchronous global-to-shared copies, but request issue and MIO
   pressure left too few eligible warps. Pacing the requests reduced the
   targeted stalls while adding 16.1% more instructions, so runtime did not
   improve.
3. Register double buffering slightly reduced K10's shared-memory dependency
   stall, but raised registers per thread from 168 to 182. Resident blocks per
   SM fell from three to two and the kernel became slower.

The value of this stage is diagnostic: it turned unsuccessful optimization
ideas into hardware-supported conclusions and preserved the parameter-search
winner.

## Measured results

Test system:

```text
GPU:            NVIDIA GeForce RTX 5060 Ti
Compute target: sm_120
Workload:       FP32 SGEMM, M=N=K=4096
Timing:         CUDA events, ordinary execution outside Nsight Compute
```

| Kernel or experiment | Time | Throughput | Decision |
|---|---:|---:|---|
| K10 warp tiling, config 21 | **9.064 ms** | **15.163 TFLOP/s** | selected |
| K11 2:2 split double buffer | 10.702 ms | 12.842 TFLOP/s | rejected |
| K11 3:1 compute/load split | 14.969 ms | 9.182 TFLOP/s | rejected |
| K10 register double buffer | 10.577 ms | 12.994 TFLOP/s | rejected |
| K12 asynchronous copy | 79.733 ms | 1.724 TFLOP/s | rejected |

These numbers describe one GPU and one workload. They are not claimed to
transfer unchanged to another architecture.

The complete Phase 5 reasoning is documented in
[docs/phase5_nsight_compute_case_study.md](docs/phase5_nsight_compute_case_study.md).
The extracted metrics and raw-report manifest are under
[reports/phase5/](reports/phase5/).

## Kernel progression

| ID | Kernel |
|---:|---|
| 0 | cuBLAS reference |
| 1 | naive SGEMM |
| 2 | coalesced global-memory access |
| 3 | shared-memory block tiling |
| 4 | 1D block tiling |
| 5 | 2D block tiling |
| 6 | vectorized memory access |
| 7-8 | shared-memory bank-conflict layouts |
| 9 | statically tuned kernel |
| 10 | warp tiling, autotuning and runtime dispatch |
| 11 | split-schedule double-buffer experiment |
| 12 | asynchronous global-to-shared copy experiment |
| 13 | shared-memory B-only diagnostic kernel |

## Build

The project was developed and tested on Windows with:

- NVIDIA CUDA Toolkit 12.8;
- Visual Studio 2022 Build Tools with the MSVC C++ workload;
- CMake and Ninja;
- Python for configuration generation and result processing;
- Nsight Compute for profiling.

Set `CUDA_COMPUTE_CAPABILITY` in
[CMakeLists.txt](CMakeLists.txt) for the target GPU. The checked-in value is
`120` for the test system.

From an x64 Visual Studio Developer PowerShell:

```powershell
conda env create -f environment.yml
conda activate SGEMM_CUDA

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sgemm
```

If CUDA, CMake, Ninja, or MSVC are installed outside `PATH`, pass their paths
through the corresponding CMake options. The batch files under `scripts/`
record the Windows toolchain used during development and may require local path
changes on another machine.

## Run and verify

Run one explicit shape:

```powershell
.\build\sgemm.exe 10 4096 4096 4096
```

Run the default square-shape suite:

```powershell
.\build\sgemm.exe 10
```

Run deterministic random-shape correctness tests:

```powershell
.\build\sgemm.exe 10 --random-tests 100
```

Command format:

```text
sgemm <kernel 0-13>
sgemm <kernel 0-13> <M> <N> <K>
sgemm <kernel 1-13> --random-tests <count>
```

## Reproduce the parameter search

Generate the legal configuration set and fixed-seed sample:

```powershell
python scripts/generate_k10_configs.py
```

The search artifacts are:

```text
scripts/k10_legal_configs.csv
scripts/k10_sampled_configs.csv
scripts/autotune_k10.py
scripts/build_k10_config.bat
```

`autotune_k10.py` compiles every sampled template configuration, benchmarks
the six target shapes, compares against cuBLAS, and persists partial results.
The Windows build script contains local toolchain paths and must be adapted
before running on another machine.

## Profile with Nsight Compute

Example full report for Kernel 10:

```powershell
ncu --kernel-name regex:sgemmWarptiling `
  --launch-skip 1 --launch-count 1 `
  --set full `
  -f -o phase5_k10_config21_4096 `
  .\build\sgemm.exe 10 4096 4096 4096
```

Ordinary execution decides whether a change is faster. Nsight replay duration
is used only for diagnosis and is never compared directly with ordinary
runtime.

Rejected diagnostic mechanisms remain reproducible through:

```text
SGEMM_K11_DIAGNOSTIC_SEQUENTIAL
SGEMM_K11_SPLIT_THREE_TO_ONE
SGEMM_K12_PACED_ASYNC
```

They are disabled by default.

## Repository layout

```text
src/kernels/       CUDA kernels and controlled experiments
src/dispatch.*     measured-shape dispatch policy
scripts/           configuration generation and autotuning automation
docs/              Nsight Compute case study
reports/phase5/    extracted metrics and raw-report manifest
sgemm.cu           benchmark, correctness validation and command-line entry
```

## Acknowledgements

The initial educational SGEMM kernel progression and benchmark structure were
derived from:

- [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)
- [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE)

This repository extends that base with arbitrary-shape correctness, legal
configuration generation, reproducible parameter search, runtime dispatch, and
the Nsight Compute case study documented above.

## License

See [LICENSE](LICENSE).
