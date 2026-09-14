# Examples and benchmarks

The worked example and the benchmarks of the MKL-style API need the MKL build
(`-DCBK_WITH_MKL=ON`, off by default), since they pack with MKL's compact
routines and compare against MKL's compact kernels and LAPACKE. The benchmarks
of the portable C API (`bench_trsm_compact` so far) measure a `cbk.h` routine
against a per-matrix BLAS/LAPACK: `-DCBK_BUILD_BENCHMARKS=ON` builds them in a
tree without MKL, against the library `find_package(LAPACK)` finds
(`-DBLA_VENDOR` selects it; see [building.md](building.md)); the MKL build
builds them too, with MKL's compact kernel added to the comparison.

## Worked example

* `solve_qr_compact` - a batch of square systems `A_v X_v = B_v` solved end to
  end with the compact pipeline (`mkl_dgeqrf_compact` -> `cbk_dormqr_compact`
  -> `cbk_dtrsm_compact`) and with the one-call `cbk_dgels_compact`,
  both cross-checked against per-matrix `LAPACKE_dgels`. The source
  (`examples/solve_qr_compact.cpp`) walks through packing, workspace queries,
  the three calls, and unpacking.

## Benchmarks

* `bench_qr_compact [nmat] [reps]` - throughput of the fully open compact *solve*
  pipeline (the three-step chain and the one-call `cbk_dgels_compact`) vs.
  MKL's batched pipeline and the one-matrix-at-a-time LAPACK paths (the chain
  and `LAPACKE_dgels`), over pools of small matrices (order 10-120), reporting
  geometric-mean speedups. All paths are checked against the known solution.
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
* `bench_trsm_compact [--nrhs=k] [--side=L|R] [--uplo=U|L] [--transa=N|T]
  [--diag=N|U] [nmat] [reps]` - the triangular *solve* `op(A) X = B` (by
  default the `R X = Q^T B` back-substitution of a QR solve): `dtrsm_compact`,
  the portable C API, vs per-matrix BLAS `dtrsm` -- MKL's compact
  `mkl_dtrsm_compact` joins in the MKL build -- over the same square-size
  range on diagonally dominant triangular pools, every path checked against
  the known solution. The host's interleave width is detected at run time
  (`--simdlen` overrides it).

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
integration tests: `example_solve_qr_compact`, `bench_qr_compact_integration`,
`bench_geqrf_compact_integration`, `bench_potrf_compact_integration`,
`bench_posv_compact_integration`, `bench_sysvnp_compact_integration`, and, in
whichever build carries it, `bench_trsm_compact_integration`.
