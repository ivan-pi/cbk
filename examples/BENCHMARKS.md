# Benchmarks

> Assisted-by: Claude:claude-opus-4-8 Claude:claude-fable-5

Throughput benchmarks for the compact batched kernels, alongside the programs
they drive. Most run the *same* math several ways -- this project's
open compact kernels, MKL's own compact kernels, and the conventional
one-matrix-at-a-time LAPACK path -- over pools of many small matrices,
reporting per-size throughput and a geometric-mean speedup; `bench_sysvnp_compact`
has no MKL yardstick (MKL ships no compact `sytrf`) and compares the fused compact
solver with per-matrix LAPACK alone. Each also cross-checks its result against per-matrix
LAPACK, so it doubles as an integration test (CTest-registered on a small pool).

| Program | Measures | Compares |
|---------|----------|----------|
| [`bench_geqrf_compact`](#bench_geqrf_compact) | QR *factorization* | `cbk_dgeqrf_compact` vs `mkl_dgeqrf_compact` vs `LAPACKE_dgeqrf` |
| [`bench_potrf_compact`](#bench_potrf_compact) | Cholesky *factorization* (SPD) | `cbk_dpotrf_compact` vs `mkl_dpotrf_compact` vs `LAPACKE_dpotrf` |
| [`bench_qr_compact`](#bench_qr_compact) | end-to-end QR *solve* `AX = B` | fully-open compact pipeline (three steps, and the one-call `gels`) vs MKL's pipeline vs per-matrix LAPACK (the three-step chain, and `LAPACKE_dgels`) |
| [`bench_posv_compact`](#bench_posv_compact) | end-to-end SPD *solve* `AX = B` | `cbk_dposv_compact` (fused Cholesky) vs its own `potrf + potrs` two-step vs MKL's compact `potrf + trsm x2` pipeline vs per-matrix `LAPACKE_dposv` |
| [`bench_sysvnp_compact`](#bench_sysvnp_compact) | end-to-end symmetric *solve* `AX = B` (indefinite) | `cbk_dsysvnp_compact` (fused unpivoted LDL^T) vs per-matrix `LAPACKE_dsysv` |

The worked, self-validating solver `solve_qr_compact` and the portable QR
workflow `qr_workflow_compact` (neither a benchmark) live in the same
folder; see [`docs/examples.md`](../docs/examples.md).

## Running them

The benchmarks are built by the MKL build, `-DCBK_WITH_MKL=ON` (see
[`docs/building.md`](../docs/building.md)). From a configured tree:

```sh
cmake --build build -j
./build/bench_geqrf_compact      # QR factorization
./build/bench_potrf_compact      # Cholesky (SPD) factorization
./build/bench_qr_compact         # end-to-end QR solve
./build/bench_posv_compact       # end-to-end SPD (Cholesky) solve
./build/bench_sysvnp_compact     # end-to-end symmetric (LDL^T) solve
```

**Build with `-march=native` for a fair comparison.** This library sets no
`-march` of its own, so a default build emits only the baseline ISA while MKL's
compact kernels dispatch to the host's widest vectors (AVX-512) at runtime -- an
unfair matchup. Pass host-tuned flags so the open kernels emit the full width:

```sh
cmake -S . -B build -DCBK_WITH_MKL=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"
```

Correctness (the gate each benchmark carries) is independent of these flags; only
throughput changes. Threading: the factorization benchmarks' cbk paths hand the
whole pool to one call and let the library thread its loop over groups
([`docs/threading.md`](../docs/threading.md)). The MKL compact paths link
sequential MKL (no internal threading; `mkl_set_num_threads(1)` pins it
regardless), so they and the per-matrix LAPACK
path are driven from an OpenMP loop over groups / matrices with the same thread
count -- every path gets the same parallelism. `bench_qr_compact` keeps its
whole *pipeline* per group inside the caller's loop for both backends: the five
steps then work on one group's cache-resident buffers, which measured 15-55%
faster than three whole-pool calls streaming the pool through separate passes.

## `bench_geqrf_compact`

Throughput of the QR factorization over pools of small square matrices. To
measure the kernels rather than data movement, the pool is packed once and only
the factorization is timed (the destroyed input restored, untimed, between
passes). Reports GFLOP/s (the standard `2mn^2 - (2/3)n^3` count), matrices/s per
path, the pairwise speedups, and a geometric-mean speedup over per-matrix LAPACK;
the error column is the compact `(H, tau)` elementwise vs a fresh `LAPACKE_dgeqrf`.

```
bench_geqrf_compact [--size-sweep=nmin:nmax[:stride]] [--simdlen=2|4|8] [nmat] [reps]
```

## `bench_potrf_compact`

The Cholesky counterpart, on the tuned column-major lower path (`A = L L^T`).
Same shape as `bench_geqrf_compact`, with three differences: `potrf` needs no
workspace; GFLOP/s uses the `n^3/3 + n^2/2 + n/6` Cholesky count (the `n` square
roots uncounted, as in LAPACK's own timing); and, since the SPD factor is *unique*
(positive diagonal), the error column is the compact factor compared elementwise
over the lower triangle vs `LAPACKE_dpotrf` -- sharper than a residual. The SPD
pool uses the cheap diagonally dominant `A_ii = 2n` form (SPD and well conditioned
without an `O(n^3)` `M^T M`).

```
bench_potrf_compact [--size-sweep=nmin:nmax[:stride]] [--simdlen=2|4|8] [nmat] [reps]
```

Three codegen prerequisites decide what this benchmark measures
(`docs/building.md`): `-march` for the vector width, `-fno-math-errno` for the
vector square root (set by the library's own CMake), and full-width vectors
under clang and icpx (set by the kernel headers). A build missing any of them
runs a different kernel.

## `bench_qr_compact`

Throughput of the end-to-end solve of many systems `A_v X_v = B_v` via QR
(`X = R^-1 Q^T B`, `--nrhs` right-hand sides per system, default one), five ways:

* **MKL batched** -- `mkl_dgeqrf_compact` -> `cbk_dormqr_compact` -> `mkl_dtrsm_compact`
* **cbk batched** -- `cbk_dgeqrf_compact` -> `cbk_dormqr_compact` -> `cbk_dtrsm_compact`
* **cbk gels** -- `cbk_dgels_compact`, the three steps as one call per group
* **unbatched** -- `LAPACKE_dgeqrf` -> `LAPACKE_dormqr` -> `cblas_dtrsm`
* **dgels** -- `LAPACKE_dgels`, LAPACK's own one-call driver, per matrix

The two three-step batched paths share `cbk_dormqr_compact` (MKL ships no
compact `ormqr`), so the cbk path runs the whole solve with *no* MKL compute
kernel and the cbk-vs-MKL ratio is the end-to-end open-vs-MKL comparison. The
gels path runs the same open group kernels inside one call between the same
pack and unpack, so gels-vs-cbk-batch is the cost of the one-call driver over
the hand-driven chain.
`LAPACKE_dgels` is the like-for-like baseline for the one-call routine (it runs
the same three steps inside, blocked, plus its norm scaling and rank test), so
gels-vs-dgels is the headline one-call-vs-one-call batched win; the unbatched
chain stays as the hand-rolled LAPACK reference. Every path is checked against
the known solution `X(:,j) = j + 1`, so the reported error is a forward error,
not a comparison to LAPACK.

```
bench_qr_compact [--nrhs=k] [--simdlen=2|4|8] [nmat] [reps]
```

`--nrhs` (default 1) moves the weight of the solve between its steps: with one
right-hand side the `O(n^3)` factorization dominates and the chain measures
`geqrf`; with many, the `O(n^2 nrhs)` apply-`Q^T` and triangular solve take
over, so a large `--nrhs` is how to weigh `cbk_dormqr_compact` (shared by both
batched chains) and `cbk_dtrsm_compact` against per-matrix `dormqr` and
`dtrsm`. Indicative run (4-core AVX-512 container, gcc `-O3 -march=native`,
1000 matrices, `n = 10..120`, one thread):

| `--nrhs` | gels vs `dgels`, geomean | at `n = 100` | cbk-batch vs MKL-batch | gels vs cbk-batch |
|---|---|---|---|---|
| 1 | `2.7x` | `1.5x` | `1.25x` | `1.01x` |
| 4 | `2.5x` | `1.5x` | `1.24x` | `1.00x` |
| 16 | `2.0x` | `1.4x` | `1.22x` | `1.00x` |
| 64 | `1.3x` | `1.0x` | `1.12x` | `1.01x` |

The batched win over per-matrix LAPACK narrows as `nrhs` grows: at `nrhs = 64`
the compact chain is still `1.5-2.5x` ahead at `n <= 30`, level at `n = 60`
and `100`, and `0.85-0.95x` at `n = 80` and `120`. Against MKL's compact
pipeline the open kernels stay ahead at every `nrhs`.

## `bench_posv_compact`

Throughput of the end-to-end solve of many symmetric positive-definite systems
`A_v X_v = B_v` via Cholesky, four ways:

* **cbk fused** -- `cbk_dposv_compact`: the Cholesky factorization and its
  two-sweep solve, fused per group of `V` matrices, one call on the whole pool
  (the library threads the group loop).
* **cbk 2-step** -- `cbk_dpotrf_compact` then `cbk_dpotrs_compact`: the
  *same* group kernels (bit-identical result), but two whole-pool calls that
  stream the pool twice. fused-vs-2-step is what the fusion buys -- pure memory
  traffic, no arithmetic difference -- so grow the pool past the cache to see
  it (`~1.0x` on a cache-resident pool, `1.1-1.3x` measured at orders `32-96`
  on pools of `134-300 MB`).
* **mkl-compact** -- `mkl_dpotrf_compact` + `mkl_dtrsm_compact` twice: MKL's
  native pipeline (it ships no compact `potrs`/`posv`), kept per group from an
  OpenMP loop so its factors are solved with cache-resident too, on the same
  thread count.
* **unbatched** -- `LAPACKE_dposv`, one matrix at a time from an OpenMP loop of
  the same thread count.

The SPD pool is `bench_potrf_compact`'s (symmetric, off-diagonals in `[-1, 1]`,
diagonal `2n`: strictly diagonally dominant with a positive diagonal, hence SPD
and well conditioned, `O(n^2)` to build). `A` and `B = A X` are packed once,
only the solve is timed (every path destroys its input, restored untimed
between passes), and every path is checked against the known solution
`X(:,j) = j + 1`, so the reported error is a forward error. GFLOP/s uses the
Cholesky `n^3/3 + n^2/2 + n/6` count plus `2 n^2 nrhs` for the two sweeps.

```
bench_posv_compact [--nrhs=k] [--size-sweep=nmin:nmax[:stride]] [--simdlen=2|4|8] [nmat] [reps]
```

Indicative run (4-core AVX-512 container, gcc `-O3 -march=native`, 512
matrices, one RHS): the fused compact solve matched MKL's compact pipeline
(geometric mean `0.9-1.0x`, `0.7-1.3x` per size) and outran per-matrix
`LAPACKE_dposv` by `2-9x` at orders `8-48` and `1.4-2.2x` at `60-105`, with the
crossover near `128-168` and LAPACK ahead from there (`0.25x` at `256`) -- a
geometric mean of `1.7x` over the size list. fused-vs-2-step was `~1.0x`
throughout: at 512 matrices the pools are cache-resident up to `n ~ 90`, and
past the crossover the solve is a small fraction of the factorization at one
RHS. On out-of-cache pools the fusion
showed directly: `1.26x` at `n = 32` (16384 matrices), `1.12x` at `n = 64`
(8192), `1.14x` at `n = 96` (4096), each with `nrhs = 8`. Every path recovered
the known solution to `~5e-15`.

## `bench_sysvnp_compact`

Throughput of the end-to-end solve of many symmetric *indefinite* systems
`A_v X_v = B_v`, two ways:

* **cbk fused** -- `cbk_dsysvnp_compact`: the unpivoted LDL^T factorization
  and its three-sweep solve, fused per group of `V` matrices, one call on the
  whole pool (the library threads the group loop).
* **unbatched** -- `LAPACKE_dsysv`: Bunch-Kaufman LDL^T factor + solve, one
  matrix at a time from an OpenMP loop of the same thread count.

Unlike the other benchmarks there is no MKL compact yardstick -- MKL has no
compact `sytrf`/`sysv`, which is why these routines exist -- and the two paths
do not run the same arithmetic: LAPACK pivots, the compact solver does not. The
pool is built so the unpivoted factorization is safe: symmetric, off-diagonals
in `[-1, 1]`, diagonal of magnitude `2n` with *alternating sign* -- strictly
diagonally dominant (every leading principal minor nonsingular, bounded element
growth) yet genuinely indefinite, so `?posv` is not an option and `?sysv` is the
standard tool. `A` and `B = A X` are packed once, only the solve is timed (both
paths destroy their input, restored untimed between passes), and both paths are
checked against the known solution `X(:,j) = j + 1`, so the reported errors are
forward errors. GFLOP/s uses the Cholesky-style `n^3/3 + n^2/2 + n/6` count plus
`2 n^2 nrhs + n nrhs` for the sweeps and the diagonal scaling.

```
bench_sysvnp_compact [--nrhs=k] [--size-sweep=nmin:nmax[:stride]] [--simdlen=2|4|8] [nmat] [reps]
```

`--nrhs` (default 1) sets the number of right-hand sides; the flags are the shared
command line of `bench_util.hpp`, so the factorization benchmarks accept it too
and ignore it.

Two codegen prerequisites decide what this benchmark measures
(`docs/building.md`): `-march` for the vector width, and full-width vectors
under clang and icpx (set by the kernel headers), which the blocked
factorization's register-tiled update depends on. A build missing either
compiles the same kernel to slower code.

## Notes

* **Defaults:** 512 matrices / 3 reps for every benchmark.
* **Flags (factorization benchmarks).** `--simdlen=2|4|8` forces a narrower
  interleave width than the host default (a wider-than-native width is rejected);
  `--size-sweep=nmin:nmax[:stride]` switches to a cbk-only throughput scan (no
  cross-check) to resolve the SIMD "staircase" finely. `bench_qr_compact` takes
  `--simdlen` and `--nrhs` but has no sweep.
* **Size list.** One list, `bench_sizes` in `bench_util.hpp`, shared by every
  benchmark but `bench_qr_compact` (five paths per size keep it on `10..120`), so
  the set of orders moves in one place. It deliberately mixes sizes that are *not*
  multiples of the interleave width `V` (30, 45, 60, 105, 168) with round powers,
  so the SIMD remainder handling stays visible.
* **It ends at `256`.** Larger orders make a run long and the batched gains are
  hard to realize there; the regime this library is about is below `128`.
  `--size-sweep` is not capped, for a deliberate scan past `256`.
* **`gels` vs the three-step chain.** The two run the same group kernels in
  the same order, so the column should read `~1.0x` at every `nrhs`; the
  column is there to show the one-call driver costs nothing over the chain.
  Measured `1.00-1.01x` at `nrhs = 1..64`. (A fused factor-and-apply kernel
  that measured `0.92-0.99x` here was retired; PLANS.md, gels.) Against
  per-matrix `LAPACKE_dgels` it measured `2.8x` (geometric mean, 4 threads,
  AVX-512, `n = 10..120`; `7.7x` at `n = 10` down to `1.4x` at `n = 120`),
  within a few percent of its ratio to the unbatched chain -- `dgels`'s own
  bookkeeping costs little at these sizes.
* **Reading the numbers.** On a `-march=native` build over the small-size range,
  the compact paths outrun per-matrix LAPACK and are competitive with MKL's
  compact kernels; LAPACK's cache-blocked algorithm crosses ahead only at larger
  orders, where the unblocked compact kernels stop being the right tool.
