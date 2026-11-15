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

- **Done:** design 6-8; blocked: `geqr2` panels of four columns, the trailing
  update as one register-tiled block reflector (`ormqr`'s `larft` +
  `larfb`), unblocked below three panels; both layouts through one strided
  kernel.
- **Open:** the panel is four columns wide, the width of the block-reflector
  tile; a wider panel (two tiles' worth of `Y`, or a recursive panel) would
  cut the trailing passes further at the larger orders. The in-panel
  `larf` is the one-reflector update, store-bound; it is a small share at
  four columns. At the smallest orders (around `n <= 12`, one right-hand
  side) the blocked path is within noise of the unblocked one. The
  strided (row-major) case runs the same tiles untuned.

## ormqr

- **Done:** design 2-6 and 8.1; `side = 'R'` is the kernel over the
  transposed view, `trans = 'C'` folds to `'T'`; the reflectors applied four
  at a time as register-tiled block reflectors (design 6.5), one at a time
  for a single slice. The portable `?ormqr_compact` takes layout, side and
  trans like the MKL-style entry point, and its suite runs the layout x side
  x trans product against LAPACKE `?ormqr` and the explicit Q of `?orgqr`
  (LAPACK's dqrt03) on every stack.
- **Gap:** the stress structures of design 7.3 (`cond` knob, banded /
  row-scaled / clustered-scale inputs) are only partly covered; `geqrf`'s
  suite has the rank-deficient and near-collinear cases. The LAPACK-style
  generators planned below would close it.
- **Open:** `Tf` is rebuilt for every call; `gels` (which forms the same
  blocks in `geqrf`) could keep them. A single right-hand side is the
  one-reflector sweep, bound by its dot product's dependency chain and the
  store port; splitting the dot over several accumulators measured slower
  (the loop is load-bound), so it stays.

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
  store-bound; `sytrfnp`'s left-looking panel solves them against the
  finished block in registers and would port (its tile has the switch), but
  for Cholesky that changes the rounding order (LAPACK's potf2 + trsm), so
  it waits on a numerics decision. The strided
  (column-major upper / row-major lower) cases run the same blocked code but
  were not tuned or measured separately.

## sytrfnp / sytrsnp / sysvnp

- **Done:** design 6-8; left-looking register-tiled panels (each tile
  loaded and stored once, its update and its own pivots' steps in between),
  split recursively above the panel width with `potrf`'s tiled trailing
  update, the pivot folded into the tile's weights; `sytrsnp` as two
  unit-diagonal `trsm` sweeps with the diagonal solve folded into the second;
  `sysvnp` the fused per-group driver. The upper convention is
  `A = U^T D U` (design 6.3), not `?sytrf`'s `U D U^T`. MKL has no compact
  `sytrf`; the `np` naming follows its `mkl_?getrfnp_compact`.
- **Open:** the factorization's tiles are `4 x 4` whatever the target; with
  16 SIMD registers that tile does not fit (`trsm` sizes its own from
  `simd_registers`). Each tile multiplies its column weights by the pivot
  per panel column; keeping the panel's columns unscaled until the panel is
  done would drop that multiply. The same left-looking panel would serve
  `potrf` (its open item). No factorization-only benchmark (the `potrf`
  harness would port).

## trsm

- **Done:** design 2-6 and 8.1; the full `side x uplo x transa x diag`
  matrix and the `alpha = 0` fast path, as one register-tiled forward
  substitution that every side, uplo, trans and layout reaches through view
  transposes and reversals; the tile sized to the target's register count;
  the diagonal applied as a reciprocal multiply with the divide off the
  accumulation path; the narrow blocks' per-tile overhead cut (design 6.2),
  which brings the one-right-hand-side solve level with or ahead of MKL's
  compact `trsm`, behind it before.
- **Open:** a tile's update reads the whole solved part of `X` above it and
  the matching rows of `A`, which stops being cache-resident for large
  orders; a recursive split of the rows would bound it. A power-of-two
  `ldap` puts a row of `A` on few cache sets (the no-transpose sweeps). No
  `trsm` benchmark; it is measured only through the solve benchmarks. Under
  icpx the narrow-block measures cost a few percent with several right-hand
  sides, where gcc and clang gain; why is not yet known.

## gels

- **Done:** design 6; both layouts, `trans in {N, T}`, any `m x n`, as one
  `for_each_group` body over the `geqrf`/`ormqr`/`trsm` group kernels (the LQ
  case is `geqrf` over the transposed view). The fused factor-and-apply
  kernel for the least-squares case was tried and retired (design 6.2): no
  arithmetic saved, and it evicted `B` from L1 at mid sizes.
- **Open:** row-major `geqrf`/`ormqr` run through the strided kernels. A whole-batch,
  library-threaded `gels` call has not been benchmarked against the
  caller-threaded per-group loop.

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
