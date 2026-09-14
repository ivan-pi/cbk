# PLANS

Status of each routine against its design document (this folder), and what
remains. Every routine ships both API surfaces -- the portable `?*_compact`
C API (LAPACK-style `info = -j`) and the MKL-style `cbk_?*_compact` (no
argument checking, scalar `info`) -- in FP64 and FP32, over one `BatchView`
kernel that covers every layout (and side, for `ormqr`/`trsm`).

Complex precisions, pivoting, and overflow/underflow-safe scaling are out of
scope throughout; each design document's section 6 says what else its routine
leaves out, and its section 7 what the test suites gate. The benchmarks are
listed in `examples.md`.

## geqrf

- **Done:** design 6-8; unblocked, both layouts through one strided kernel.
- **Open:** the factorization is unblocked: each reflector makes its own pass
  over the trailing block. A blocked (compact-WY) factorization is the
  standard next step, for `geqrf` and `gels` alike; not tried.

## ormqr

- **Done:** design 2-6 and 8.1; `side = 'R'` is the kernel over the
  transposed view, `trans = 'C'` folds to `'T'`.
- **Gap:** the stress structures of design 7.3 (`cond` knob, banded /
  row-scaled / clustered-scale inputs) are only partly covered; `geqrf`'s
  suite has the rank-deficient and near-collinear cases. The LAPACK-style
  generators planned below would close it.
- **Open:** the one-column tail of `larf` (all of `nrhs < 4`) reduces `v^T c`
  into a single accumulator and runs at FMA latency; split it into several.
  `larf_block` takes `tau` by reference to a pack, which is safe only while it
  inlines (`.claude/CLAUDE.md`); pass the pointer and index instead.

## orgqr

- **Done:** design 6 and 8; in place on the `?geqrf_compact` output
  (`m >= n >= k`), no scratch.
- **Open:** nothing. `?orglq` (this kernel over the transposed view) waits for
  a use case.

## potrf / potrs / posv

- **Done:** design 6-8; the factorization recursively blocked (6.1) with the
  pivot's `sqrt` off the pivot-to-pivot chain (6.2) and the four
  `(layout, uplo)` cases one kernel over transposed views (6.3); `potrs` as
  two non-unit `trsm` group sweeps; `posv` the fused per-group driver (6.8).
  MKL has a compact `potrf` but no compact `potrs` or `posv`.
- **Open:** the smallest orders are behind `mkl_?potrf_compact`: the pivot's
  divide and sqrt on the one divider port bound both, so only an approximate
  reciprocal off the pivot chain would move it, and that is AVX-512-specific
  and not bit-reproducible against the divide. The rows below a panel's
  diagonal block are a triangular solve done as the rank-1 sweep,
  store-bound; solving them against the finished block in registers measured
  a further gain at moderate orders but changes the rounding order (LAPACK's
  potf2 + trsm), so it waits on a numerics decision. The strided
  (column-major upper / row-major lower) cases run the same blocked code but
  were not tuned or measured separately.

## sytrfnp / sytrsnp / sysvnp

- **Done:** design 6-8; blocked recursively like `potrf`, sharing its
  register-tiled trailing update with the pivot folded into the tile's
  weights; `sytrsnp` as two unit-diagonal `trsm` sweeps around a diagonal
  solve; `sysvnp` the fused per-group driver. The upper convention is
  `A = U^T D U` (design 6.3), not `?sytrf`'s `U D U^T`. MKL has no compact
  `sytrf`; the `np` naming follows its `mkl_?getrfnp_compact`.
- **Open:** the blocking was tuned on the contiguous case; the strided cases
  run the same code untuned. The solve is the plain sweeps. No
  factorization-only benchmark (the `potrf` harness would port).

## trsm

- **Done:** design 2-6 and 8.1; the full `side x uplo x transa x diag`
  matrix and the `alpha = 0` fast path. Column-major `side = 'L'` is the
  tuned 4/2/1 register-blocked path; the group kernel routes on views, so a
  transposed column-major view reaches it too (`gels` does). Every other
  combination is the strided kernel.
- **Open:** per-group overhead at the smallest orders, where it is behind
  `mkl_?trsm_compact`; reciprocal-multiplying the diagonal in the blocked
  paths; tuning the strided kernel.

## gels

- **Done:** design 6; both layouts, `trans in {N, T}`, any `m x n`, as one
  `for_each_group` body over the `geqrf`/`ormqr`/`trsm` group kernels (the LQ
  case is `geqrf` over the transposed view). The fused factor-and-apply
  kernel for the least-squares case was tried and retired (design 6.2): no
  arithmetic saved, and it evicted `B` from L1 at mid sizes.
- **Open:** row-major runs through the strided kernels. A whole-batch,
  library-threaded `gels` call has not been benchmarked against the
  caller-threaded per-group loop. The blocked factorization is `geqrf`'s open
  item.

## Project-wide

- **Done: test against a real BLAS/LAPACK (issue #27).** The suites validate
  against LAPACKE + CBLAS (`tests/test_lapack_util.hpp`): the `ref_*`
  procedures forward to `LAPACKE_?geqr2` / `?ormqr` / `?orgqr` / `?geqp3` /
  `?potrf` / `?gels` and `cblas_?trsm` / `?trmm` / `?gemm`; the unpivoted
  LDL^T (`ref_sytf2np`, which LAPACK lacks) is the one hand-rolled reference
  left, validated against `cblas_?trmm` in its suite.
  `cmake/FindLAPACKE.cmake` takes MKL's LAPACKE with the MKL extension and
  otherwise `find_package(LAPACK)`'s pick under `BLA_VENDOR` plus its
  LAPACKE; CI runs OpenBLAS and Netlib under gcc and clang, and
  Accelerate + accelerate-lapacke on the macOS runners
  (`.github/workflows/macos.yml`, written without a macOS machine at hand).
  The dense cross-checks of the `geqrf` and `potrf` MKL suites (design 7.1)
  moved into the portable suites, on every stack. Where a reference is
  LAPACK's blocked driver the gate is relative to the operand norms at a
  multiple of `n eps`; only `?geqr2` stays an elementwise `~eps` comparison.
- **LAPACK-style test coverage.** Adopt the testing approaches of the
  reference LAPACK repository (its `TESTING/LIN` drivers): `?latms`-style
  generators with prescribed condition number and spectral distribution, and
  the standard scaled residual gates, extending the ad-hoc generators of
  `test_compact_util.hpp`. Would close `ormqr`'s design-7.3 stress-structure
  gap along the way.
