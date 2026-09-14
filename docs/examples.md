# Examples and benchmarks

One program under `examples/` builds in the default configure, on the portable
C API alone (it needs OpenMP); the rest need the MKL build (`-DCBK_WITH_MKL=ON`, off by
default), since they pack with MKL's compact routines and compare against
MKL's compact kernels and LAPACKE.

## Portable example

* `qr_workflow_compact <nm> <V> <nrows> <ncols>` - the interleave-batch QR
  workflow example Arm ships with Arm Performance Libraries, transcribed onto
  the C API of `cbk.h` in plain C99 with no internal helpers: a batch of `nm`
  square or tall `m x n` matrices is packed into the compact layout with
  interleave width `V` (2, 4, 8 or 16; a last group `V` does not fill is
  padded with identities), QR-factored with `dgeqrf_compact`, its `R`
  extracted, rebuilt as `Q R` with `dormqr_compact`, unpacked, and checked,
  `norm1(A - QR) <= 5 eps m n norm1(A)` for every matrix, after three warm-up
  runs; each phase is timed with `omp_get_wtime` and the pack, extract and
  unpack loops are OpenMP loops over the groups, as in the original. The
  batch is described the way this library (and MKL's compact API) does, by
  `nm` and `V` and the layout formula of `cbk.h`, in place of ArmPL's
  `nbatch`, `ninter` and explicit strides (`ninter` is `V`, `nbatch` the
  number of groups); the source (`examples/qr_workflow_compact.c`) writes
  the pack and unpack out as two static functions, the LAPACK comparison is
  left out (the portable build links no LAPACK), and ArmPL's column pivoting
  has no counterpart. Arm's 32768-matrix runs are `32768 8 <n> <n>` or
  `32768 4 <n> <n>` here. Needs OpenMP for C; skipped without.

## Worked example

* `solve_qr_compact` - a batch of square systems `A_v X_v = B_v` solved end to
  end with the compact pipeline (`mkl_dgeqrf_compact` -> `cbk_dormqr_compact`
  -> `cbk_dtrsm_compact`) and with the one-call `cbk_dgels_compact`,
  both cross-checked against per-matrix `LAPACKE_dgels`. The source
  (`examples/solve_qr_compact.cpp`) walks through packing, workspace queries,
  the three calls, and unpacking.

## Benchmarks

* `bench_qr_compact [--nrhs=k] [--simdlen=2|4|8] [nmat] [reps]` - throughput
  of the fully open compact *solve* pipeline (the three-step chain and the
  one-call `cbk_dgels_compact`) vs. MKL's batched pipeline and the
  one-matrix-at-a-time LAPACK paths (the chain and `LAPACKE_dgels`), over pools
  of small matrices (order 10-120), reporting geometric-mean speedups. `--nrhs`
  sets the right-hand sides per system (default 1); a large value weighs the
  apply-`Q^T` and triangular solve rather than the factorization. All paths are
  checked against the known solution.
* `bench_geqrf_compact [nmat] [reps]` - throughput of the QR *factorization*:
  `cbk_dgeqrf_compact` vs `mkl_dgeqrf_compact` vs per-matrix
  `LAPACKE_dgeqrf`, across the target square-size range, reporting GFLOP/s and a
  geometric-mean speedup, checked against LAPACK.
* `bench_potrf_compact [nmat] [reps]` - the Cholesky counterpart: throughput of
  the SPD *factorization* `cbk_dpotrf_compact` vs `mkl_dpotrf_compact` vs
  per-matrix `LAPACKE_dpotrf`, over the same square-size range (tuned col-major
  lower, `A = L L^T`), reporting GFLOP/s and a geometric-mean speedup, checked
  elementwise against LAPACK (the SPD factor is unique).
* `bench_posv_compact [--nrhs=k] [nmat] [reps]` - the end-to-end SPD *solve*:
  `cbk_dposv_compact` (the fused Cholesky factor + solve, one call on the
  pool) vs the same kernels as two whole-pool calls (`potrf` + `potrs` -- what
  the fusion buys) vs MKL's compact pipeline (`mkl_dpotrf_compact` +
  `mkl_dtrsm_compact` twice; MKL has no compact `potrs`/`posv`) vs per-matrix
  `LAPACKE_dposv`, over the same square-size range on SPD diagonally dominant
  pools, every path checked against the known solution.
* `bench_sysvnp_compact [--nrhs=k] [nmat] [reps]` - the end-to-end symmetric
  *solve*: `cbk_dsysvnp_compact` (the fused unpivoted LDL^T factor + solve,
  one call on the pool) vs per-matrix `LAPACKE_dsysv` (Bunch-Kaufman), over the
  same square-size range on indefinite but diagonally dominant pools, both paths
  checked against the known solution. No MKL yardstick: MKL has no compact
  `sytrf`.

The factorization and solve benchmarks share the `--size-sweep=nmin:nmax[:stride]`
and `--simdlen=2|4|8` flags. For a fair comparison against MKL, build with
host-tuned flags (`-DCMAKE_CXX_FLAGS="-O3 -march=native"`) so the compact
kernels use the full vector width, as MKL's runtime dispatch does (see
[building.md](building.md)).

What each benchmark measures, how to run it, the flags, and the measured
results are documented in detail in
[`examples/BENCHMARKS.md`](../examples/BENCHMARKS.md).

## CTest registration

All programs are registered with CTest, on a small pool, so they double as
integration tests: `example_qr_workflow_square` and
`example_qr_workflow_tall` in every build with OpenMP, and with the MKL build
`example_solve_qr_compact`, `bench_qr_compact_integration`,
`bench_geqrf_compact_integration`, `bench_potrf_compact_integration`,
`bench_posv_compact_integration`, `bench_sysvnp_compact_integration`.
