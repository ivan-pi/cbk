# Benchmarks

> Assisted-by: Claude:claude-opus-4.8

Throughput benchmarks for the compact batched kernels, alongside the programs
they drive. The first three run the *same* math three ways -- this project's
open compact kernels, MKL's own compact kernels, and the conventional
one-matrix-at-a-time LAPACK path -- over pools of many small matrices,
reporting per-size throughput and a geometric-mean speedup; the fourth has no
MKL yardstick (MKL ships no compact `sytrf`) and compares the fused compact
solver with per-matrix LAPACK alone; the fifth is the AArch64 edition of the
first, against Arm Performance Libraries (no MKL exists on ARM). Each also
cross-checks its result against per-matrix
LAPACK, so it doubles as an integration test (CTest-registered on a small pool).

| Program | Measures | Compares |
|---------|----------|----------|
| [`bench_geqrf_compact`](#bench_geqrf_compact) | QR *factorization* | `cqr_mkl_dgeqrf_compact` vs `mkl_dgeqrf_compact` vs `LAPACKE_dgeqrf` |
| [`bench_potrf_compact`](#bench_potrf_compact) | Cholesky *factorization* (SPD) | `cqr_mkl_dpotrf_compact` vs `mkl_dpotrf_compact` vs `LAPACKE_dpotrf` |
| [`bench_qr_compact`](#bench_qr_compact) | end-to-end QR *solve* `AX = B` | fully-open compact pipeline (three steps, and the one-call `gels`) vs MKL's pipeline vs per-matrix LAPACK (the three-step chain, and `LAPACKE_dgels`) |
| [`bench_sysvnp_compact`](#bench_sysvnp_compact) | end-to-end symmetric *solve* `AX = B` (indefinite) | `cqr_mkl_dsysvnp_compact` (fused unpivoted LDL^T) vs per-matrix `LAPACKE_dsysv` |
| [`bench_geqrf_armpl`](#bench_geqrf_armpl) | QR *factorization* (AArch64, `-DCQR_WITH_ARMPL=ON`) | `dgeqrf_compact` (portable C API) vs per-matrix `LAPACKE_dgeqrf` (Arm Performance Libraries) |

The worked, self-validating solver `solve_qr_compact` (not a benchmark) lives in
the same folder; see the top-level [README](../README.md).

## Running them

The benchmarks are built by the standard MKL build (see the
[README](../README.md) / [CLAUDE.md](../.claude/CLAUDE.md)). From a configured tree:

```sh
cmake --build build -j
./build/bench_geqrf_compact      # QR factorization
./build/bench_potrf_compact      # Cholesky (SPD) factorization
./build/bench_qr_compact         # end-to-end QR solve
./build/bench_sysvnp_compact     # end-to-end symmetric (LDL^T) solve
```

**Build with `-march=native` for a fair comparison.** This library sets no
`-march` of its own, so a default build emits only the baseline ISA while MKL's
compact kernels dispatch to the host's widest vectors (AVX-512) at runtime -- an
unfair matchup. Pass host-tuned flags so the open kernels emit the full width:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"
```

Correctness (the gate each benchmark carries) is independent of these flags; only
throughput changes. Threading: the factorization benchmarks' cqr paths hand the
whole pool to one call and let the library thread its loop over groups (README,
"Threading"). The MKL compact paths link sequential MKL (no internal threading;
`mkl_set_num_threads(1)` pins it regardless), so they and the per-matrix LAPACK
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

## `bench_qr_compact`

Throughput of the end-to-end solve of many systems `A_v X_v = B_v` via QR
(`X = R^-1 Q^T B`, single RHS), five ways:

* **MKL batched** -- `mkl_dgeqrf_compact` -> `cqr_mkl_dormqr_compact` -> `mkl_dtrsm_compact`
* **cqr batched** -- `cqr_mkl_dgeqrf_compact` -> `cqr_mkl_dormqr_compact` -> `cqr_mkl_dtrsm_compact`
* **cqr gels** -- `cqr_mkl_dgels_compact`, the three steps as one call per group
* **unbatched** -- `LAPACKE_dgeqrf` -> `LAPACKE_dormqr` -> `cblas_dtrsm`
* **dgels** -- `LAPACKE_dgels`, LAPACK's own one-call driver, per matrix

The two three-step batched paths share `cqr_mkl_dormqr_compact` (MKL ships no
compact `ormqr`), so the cqr path runs the whole solve with *no* MKL compute
kernel and the cqr-vs-MKL ratio is the end-to-end open-vs-MKL comparison. The
gels path runs the same open kernels fused into one call -- the apply-`Q^T`
folded into the factorization, no separate sweep over the reflectors -- between
the same pack and unpack, so gels-vs-cqr-batch is what the fusion buys.
`LAPACKE_dgels` is the like-for-like baseline for the one-call routine (it runs
the same three steps inside, blocked, plus its norm scaling and rank test), so
gels-vs-dgels is the headline one-call-vs-one-call batched win; the unbatched
chain stays as the hand-rolled LAPACK reference. Every path is checked against
the known solution `X == 1`, so the reported error is a forward error, not a
comparison to LAPACK.

```
bench_qr_compact [nmat] [reps]
```

## `bench_sysvnp_compact`

Throughput of the end-to-end solve of many symmetric *indefinite* systems
`A_v X_v = B_v`, two ways:

* **cqr fused** -- `cqr_mkl_dsysvnp_compact`: the unpivoted LDL^T factorization
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

Indicative run (4-core AVX-512 container, gcc `-O2 -march=native`, 512 matrices,
one RHS): the fused compact solve outran per-matrix `LAPACKE_dsysv` by `7-9x` at
orders `8-32`, `3-5x` at `45-64`, `1.1-2x` at `96-256`, and fell behind at
`384` and `500` (`0.6x`, `0.35x`), where LAPACK's blocked, pivoted factorization
is the better tool -- a geometric mean of `2.7x` over the default size list.
Both paths recovered the known solution to `~6e-15`.

## `bench_geqrf_armpl`

The AArch64 counterpart of `bench_geqrf_compact`: the same pre-packed QR
factorization benchmark -- same pool fill, size list, timing discipline and
LAPACK correctness gate -- built on the portable C API against per-matrix
`LAPACKE_dgeqrf` from [Arm Performance Libraries](https://learn.arm.com/install-guides/armpl/).
ArmPL ships no compact-format kernels, so there is no batched yardstick
column, and the pool is packed by the benchmark itself (MKL's packing
routines being equally absent); ArmPL is linked serial and the per-matrix
loop OpenMP-parallelized, so both paths run on the same thread count, as in
the MKL benchmarks.

```
bench_geqrf_armpl [--simdlen=2|4|8] [nmat] [reps]
```

Built only with `-DCQR_WITH_ARMPL=ON` (`cmake/FindArmPL.cmake` locates the
library from `$ARMPL_DIR`, as `module load armpl` sets, or `/opt/arm`).
Unlike the MKL benchmarks, `--simdlen` is not capped at the host's native
vector width: the portable kernels run any interleave width anywhere (a wider
`V` becomes unrolled 128-bit bursts on NEON), and no width is the obvious
winner a priori, so sweep it. The manual-dispatch workflow
`.github/workflows/armpl-bench.yml` does exactly that on GitHub's free
AArch64 runners, one run per width. Build with `-mcpu=native` for the fair
comparison, as above.

## Notes

* **Defaults:** 512 matrices / 3 reps for the factorization benchmarks and
  `bench_sysvnp_compact`, 1000 / 3 for `bench_qr_compact`.
* **Flags (factorization benchmarks).** `--simdlen=2|4|8` forces a narrower
  interleave width than the host default (a wider-than-native width is rejected);
  `--size-sweep=nmin:nmax[:stride]` switches to a cqr-only throughput scan (no
  cross-check) to resolve the SIMD "staircase" finely.
* **Size list.** The default deliberately mixes sizes that are *not* multiples of
  the interleave width `V` (30, 45, 60, 105, 168) with round powers, so the SIMD
  remainder handling stays visible across the target small-to-medium range.
* **`gels` vs the three-step chain.** With the benchmark's single right-hand
  side the two are equal (`1.00x` geometric mean over `n = 10..100`): the
  `O(n^3)` factorization dominates and fusing the apply-`Q^T` into it saves an
  `O(n^2)` sweep, so the fusion pays in proportion to `nrhs`, not `n`. The
  column is there to show the one-call routine costs nothing over the chain.
  Against per-matrix `LAPACKE_dgels` it measured `3.2x` (geometric mean, 4
  threads, AVX-512, `n = 10..100`; `8.5x` at `n = 10` down to `1.9x` at
  `n = 100`), within a few percent of its ratio to the unbatched chain --
  `dgels`'s own bookkeeping costs little at these sizes.
* **Reading the numbers.** On a `-march=native` build over the small-size range,
  the compact paths outrun per-matrix LAPACK and are competitive with MKL's
  compact kernels; LAPACK's cache-blocked algorithm crosses ahead only at larger
  orders, where the unblocked compact kernels stop being the right tool.
