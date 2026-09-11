# PLANS

Status of each routine against its design document (`docs/`), plus open items.
Every routine ships both API surfaces -- the MKL-style `cqr_mkl_?*_compact`
(no argument checking, scalar `info`) and the portable `?*_compact` C API
(LAPACK-style `info = -j`) -- in FP64 and FP32, over one `BatchView` kernel
that covers every layout (and side, for `ormqr`/`trsm`). Complex precisions,
pivoting, and overflow/underflow-safe scaling are out of scope throughout.

## geqrf

- **Implemented (design 6-8):** vectorized unblocked `geqr2` with a branch-free
  masked `larfg`; the trailing update is `ormqr`'s `larf`. One strided kernel
  for both layouts (column-major contiguous).
- **Validated (design 7):** BLAS-free test vs a scalar `geqr2`; MKL/LAPACK
  test gating factorization residual (`20 n eps`) and orthogonality
  (`100 n eps`), cross-checking `mkl_dgeqrf_compact` in both layouts, closing
  `AX = B`, and covering rank-deficient / near-collinear inputs.
- **Benchmarked:** `bench_geqrf_compact`.
- **Scoped out (design 6.6):** `dlarfg` rescaling, column pivoting, blocked
  (`larft`/`larfb`) factorization at the target sizes.
- **Done:** the one-pass `[A | B]` reduction, as `geqrf_panel_compact_group`
  (see gels). The plain kernel's signature and generated code are unchanged: a
  flag parameter, even compiled out, cost it 8-13% under GCC.
- **Deferred:** a benchmark against `batmat`'s `geqrf` (same interleaved
  format).

## ormqr

- **Implemented (design 2-6, 8.1):** vectorized unblocked `dorm2r`, both
  sides, both layouts, `trans in {N, T}` (`C` folds to `T`). The shared `larf`
  is register-blocked four slices at a time; `side = 'R'` is the same kernel
  over the transposed view. Padded slots carry `tau = 0`, so they are no-ops.
- **Validated (design 7):** BLAS-free test vs a scalar `dorm2r` (including a
  column-pivoted QR + back-permutation solve); MKL test of `op(Q) C` vs dense
  `LAPACKE_dormqr` over `layout x side x trans` (`20 s eps`) plus the
  `mkl_dgeqrf_compact -> cqr_mkl_dormqr_compact -> mkl_dtrsm_compact` solve
  (`100 n eps`).
- **Known gap:** the stress structures of design 7.3 (`cond` knob, banded /
  row-scaled / clustered-scale inputs) are only partly covered; `geqrf`'s
  suite has the rank-deficient and near-collinear cases.

## potrf

- **Implemented (design 6-8):** vectorized unblocked `potf2`, unconditional
  `sqrt` pivot, `JB = 4` register-blocked rank-1 update. The four
  `(layout, uplo)` cases are one kernel over transposed views (design 6.3).
- **Validated (design 7):** BLAS-free test vs a scalar `potf2` (both `uplo`,
  both layouts, padding, non-SPD lane isolation); MKL/LAPACK test gating the
  reconstruction residual (`20 n eps`), the untouched triangle (bit-for-bit),
  the factor vs `LAPACKE_dpotrf` (`20 n eps`), the cross-check vs
  `mkl_dpotrf_compact` (bit-exact), and the SPD solve.
- **Benchmarked:** `bench_potrf_compact`.
- **Scoped out (design 6.6):** positive-definiteness is assumed (a non-SPD lane
  poisons itself with `NaN`/`Inf`); no blocked factorization.

## sytrfnp / sytrsnp / sysvnp

The unpivoted LDL^T factorization, its solve, and the fused `?sysv`-style
driver (`docs/cqr_mkl_dsytrfnp_compact_design.md`); MKL has no compact
`sytrf`, and the `np` naming follows its `mkl_?getrfnp_compact`.

- **Implemented (design 6-8):** the square-root-free sweep with the `JB = 4`
  trailing update, structured like `potrf` (one kernel over transposed views;
  column-major lower and row-major upper contiguous); the solve as two
  unit-diagonal `trsm` group sweeps around a diagonal solve; `sysvnp` factoring
  and solving each group while its factor is cache-resident, bit-identical to
  the two calls. The upper convention is `A = U^T D U` (design 6.3), not
  `?sytrf`'s `U D U^T`.
- **Validated (design 7):** BLAS-free test over `(T, V, uplo, layout)` with
  padding, the end-to-end indefinite solve (two-step and fused), the zero-pivot
  semantics of design 6.2, and C-API validation of all three entry points;
  MKL test gating reconstruction (`20 n eps`), the untouched triangle, an
  elementwise `(L, D)` cross-check vs `mkl_?getrfnp_compact` (`~3e-15` FP64),
  the indefinite solve over both `uplo` and layouts, and the fused driver's
  bit-identity.
- **Benchmarked:** `bench_sysvnp_compact` (fused solve vs per-matrix
  `LAPACKE_dsysv`).
- **Scoped out (design 6.2, 6.7):** no pivoting (a singular leading minor
  poisons its lane; Bunch-Kaufman does not vectorize per lane); no complex
  Hermitian variants; the strided sweep is correctness-first.
- **Open:** no factorization-only benchmark (the potrf harness would port);
  no `sysvnp` vs `sytrfnp + sytrsnp` measurement on out-of-cache pools, the
  comparison that would quantify the fusion (design 6.8).

## trsm

- **Implemented (design 2-6, 8.1):** full `side x uplo x transa x diag`,
  `alpha = 0` as the BLAS `B := 0` fast path. Column-major `side = 'L'` is
  tuned (4/2/1 register-blocked row-dot, contiguous axpy for the leftover
  column); the group kernel routes on views, so a transposed column-major view
  reaches the tuned path too (used by gels). Other combinations are strided.
- **Validated (design 7):** BLAS-free test vs a scalar `?trsm` (forward error
  and `||op(A) X - alpha B||`); MKL cross-check vs `mkl_?trsm_compact` over the
  full feature matrix plus the end-to-end solve.
- **Performance vs `mkl_?trsm_compact`** (single thread, AVX-512, orders
  10-148, measured before the view-based routing): `nrhs = 1` ~`1.0x`, `nrhs`
  a multiple of 4 ~`1.3-1.5x`, mixed ~`1.0-1.3x`. Open: small-`n` per-group
  overhead (~`0.6-0.9x` at `n ~ 10`), reciprocal-multiplying the diagonal in
  the blocked paths, tuning the strided kernel.
- **Scoped out:** no singularity check (a zero non-unit diagonal divides to
  `Inf`/`NaN`, as in BLAS).

## gels

- **Implemented (design 6):** LAPACK `?gels` for the compact format, both
  layouts, `trans in {N, T}`, any `m x n`. The four cases are one kernel: QR
  of the tall orientation of `A` (its view, transposed when `m < n`, which is
  `?gelqf` storage for free), `B := Q^T B` fused into the factorization and
  `R X = B` (least squares), or `R^T Y = B`, `B := Q [Y; 0]` (minimum norm),
  one `for_each_group` body over the `geqrf`/`ormqr`/`trsm` group kernels. The
  MKL-style `work` is the per-group `tau` scratch (`lwork >= min(m,n) * V *
  ceil(nm/V)`) and holds `tau` on exit; the portable C API takes it as an
  explicit `taup` output. `(ap, tau)` feed `ormqr` for further right-hand
  sides.
- **Validated (design 7):** BLAS-free test vs a scalar `ref_gels` (`X`,
  factorization and `tau` elementwise) plus the defining properties formed
  independently, C-API validation, and the `min(m,n) = 0` quick return; MKL
  test vs per-matrix `LAPACKE_?gels` over every `(layout, trans)` and
  square/tall/wide shape (forward error `100 max(m,n) eps`, residual
  contracts, workspace query, factorization vs `LAPACKE_?geqrf`/`?gelqf`); and
  a cross-check vs the `mkl_?geqrf_compact -> cqr_mkl_?ormqr_compact ->
  mkl_?trsm_compact` pipeline on the same packed input.
- **Benchmarked:** the `cqr-gels` path of `bench_qr_compact` vs per-matrix
  `LAPACKE_dgels`: `3.2x` geometric mean over `n = 10..100` (4 threads,
  AVX-512; `8.5x` at `n = 10`, `1.9x` at `n = 100`). At `nrhs = 1` it matches
  the three-step chain (`1.00x`): the fused apply-`Q^T` saves an `O(n^2)`
  sweep against an `O(n^3)` factorization, so the fusion's gain scales with
  `nrhs`.
- **Scoped out (design 6.6):** no rank-deficiency test (a zero diagonal of `R`
  divides to `Inf`/`NaN` in that lane), no `?lascl` rescaling, no pivoting.
- **Open:** row-major runs through the strided kernels. A whole-batch,
  library-threaded `gels` call has not been benchmarked against the
  caller-threaded per-group loop. The unblocked factorization is
  bandwidth-bound across the target range (~`10 GF/s` per AVX-512 core, flat
  from `n = 40` to `120`; it streams the trailing block per reflector), so a
  blocked (compact-WY) factorization is the next lever for `gels` and `geqrf`
  and the one that would make multiple right-hand sides pay.

## Project-wide

- **Precision coverage.** Every suite is templated on the scalar type and runs
  in FP64 and FP32 through the `cqr_mkl<T>` / `mkl<T>` / `lapack<T>` dispatch
  of `tests/test_mkl_util.hpp`; FP32 cross-checks agree with `mkl_s*_compact`
  to ~1e-6, gated at 1e-4.
- **Threading.** Each routine's group loop is an OpenMP `parallel for`
  (static, at most one thread per group), active for two or more groups and a
  work estimate above `CQR_OMP_MIN_FLOPS` (`5e4`, a compromise between two
  measured fork/join break-evens). Inside a caller's parallel region it stays
  serial unless nesting is enabled (`OMP_NUM_THREADS=8,2`). The factorization
  benchmarks hand the whole pool to one cqr call and drive the sequential MKL
  and LAPACK references from an equivalent outer loop; the solve benchmark
  keeps its pipeline per group (whole-pool passes measured 15-55% slower).
  `gels` and `sysvnp` are the fused per-group drivers that give library-side
  threading of a whole solve.
- **MKL Compact contract.** `.claude/mkl-compact-behavior.md` records what
  MKL's own compact routines were measured to do (`info` and `work` mandatory,
  `lwork` unchecked, `n*V` scratch per thread, internal threading only under
  the threaded layer). The wrappers follow it; cqr's own workspace policy for
  `gels` may still change.
- **No install/export.** No `install()`/package-config rules; the project is
  not consumable via `find_package(cqr)`.
- **Alignment.** Compact buffers are correct at any `T` alignment on GCC and
  clang (issue #34); pack-width alignment is a performance recommendation only.
