# PLANS

Status of each routine against its design document (this folder), plus open items.
Every routine ships both API surfaces -- the MKL-style `cbk_?*_compact`
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
- **Scoped out (design 6.6):** `dlarfg` rescaling (the portable test pins the
  underflow behavior: `tau = 0`, diagonal kept, body zeroed), column pivoting, blocked
  (`larft`/`larfb`) factorization at the target sizes.

## ormqr

- **Implemented (design 2-6, 8.1):** vectorized unblocked `dorm2r`, both
  sides, both layouts, `trans in {N, T}` (`C` folds to `T`). The shared `larf`
  is register-blocked four slices at a time; `side = 'R'` is the same kernel
  over the transposed view. Padded slots carry `tau = 0`, so they are no-ops.
- **Validated (design 7):** BLAS-free test vs a scalar `dorm2r` (including a
  column-pivoted QR + back-permutation solve); MKL test of `op(Q) C` vs dense
  `LAPACKE_dormqr` over `layout x side x trans` (`20 s eps`) plus the
  `mkl_dgeqrf_compact -> cbk_dormqr_compact -> mkl_dtrsm_compact` solve
  (`100 n eps`).
- **Benchmarked:** through `bench_qr_compact --nrhs=k` (both batched chains
  share it; at large `nrhs` it and the triangular solve are most of the chain).
- **Known gap:** the stress structures of design 7.3 (`cond` knob, banded /
  row-scaled / clustered-scale inputs) are only partly covered; `geqrf`'s
  suite has the rank-deficient and near-collinear cases.
- **Open:** the one-column tail of `larf` (all of `nrhs < 4`) reduces `v^T c`
  into a single accumulator and runs at FMA latency; split it into several.
  `larf_block` takes `tau` by reference to a pack, which is safe only while it
  inlines (`.claude/CLAUDE.md`); pass the pointer and index instead.

## orgqr

- **Implemented (design 6, 8):** vectorized unblocked `dorg2r`: columns
  `k..n-1` unit-seeded, then a backward sweep on the shared `larf` over the
  trailing columns only, with each column `kk` formed sweep-free
  (`1 - tau` diagonal, `-tau`-scaled reflector body, zeros above). In place on
  the `?geqrf_compact` output (`m >= n >= k`); both layouts through the one
  strided kernel; no scratch (the MKL-style `lwork = -1` query answers 1).
- **Validated (design 7):** BLAS-free test vs a scalar `dorg2r` plus the
  independent invariants (`Q^T Q = I`, `Q R = A`, padded lanes exactly
  identity), covering `k < n`, `k = 0` and both layouts; MKL test vs dense
  `LAPACKE_dorgqr` (cross-check tolerance) gating orthogonality (`100 n eps`)
  and reconstruction (`100 m eps`), plus bit-exact agreement with
  `cbk_?ormqr_compact` applied to a packed identity.
- **Scoped out (design 6.5):** complex (`?ungqr`); `?orglq` (this kernel over
  the transposed view) until a use case asks; blocked accumulation.

## potrf / potrs / posv

The Cholesky factorization, its solve, and the fused `?posv`-style driver
(`docs/cbk_dpotrf_compact_design.md`); MKL has a compact `potrf` but no
compact `potrs` or `posv`.

- **Implemented (design 6-8):** vectorized `potf2`, recursively blocked
  (design 6.1): 8-column leaf panels with the `JB = 4` rank-1 update confined
  to the panel, and a `4 x 4` register-tiled rank-K update of the trailing
  columns at every split, so a trailing element is stored and reloaded once
  per split level rather than once per pivot; the pivot's unconditional
  `sqrt` moved off the pivot-to-pivot chain (design 6.2). The four
  `(layout, uplo)` cases are one kernel over transposed views (design 6.3).
  The solve as two non-unit `trsm` group sweeps (`?sytrsnp` minus the diagonal
  step); `posv` factoring and solving each group while its factor is
  cache-resident, bit-identical to the two calls.
- **Validated (design 7):** BLAS-free test vs a scalar `potf2` (both `uplo`,
  both layouts, padding, non-SPD lane isolation), the end-to-end SPD solve
  (two-step and fused) and C-API validation of all three entry points;
  MKL/LAPACK test gating the
  reconstruction residual (`20 n eps`), the untouched triangle (bit-for-bit),
  the factor vs `LAPACKE_dpotrf` (`20 n eps`), the cross-check vs
  `mkl_dpotrf_compact` (bit-exact), the SPD solve through `mkl_?trsm_compact`
  and through `potrs` (`100 n eps`), and the fused driver's bit-identity.
- **Benchmarked:** `bench_potrf_compact` (the factorization, against
  `mkl_?potrf_compact` and per-matrix `LAPACKE_?potrf`) and
  `bench_posv_compact` (the end-to-end solve: fused vs its own two-step calls
  vs MKL's compact `potrf + trsm x2` pipeline vs per-matrix `LAPACKE_dposv`).
  The latter is the fusion measurement design 6.8 called for: the fused
  driver gains over its own two calls only once the pool is out of cache,
  and modestly there (`examples/BENCHMARKS.md`).
- **Performance vs `mkl_?potrf_compact`:** the blocked kernel is behind MKL
  at the smallest orders, where the pivot's divide and sqrt on the one
  divider port bound both, level with it in the low tens, and ahead from
  there on by a margin that grows with the order (the plain sweep it replaced
  was behind at every order). Ahead of per-matrix `LAPACKE_?potrf` at every
  size of the benchmark list.
- **Scoped out (design 6.6):** positive-definiteness is assumed (a non-SPD lane
  poisons itself with `NaN`/`Inf`, and propagates through a `potrs` solve with
  that factor).
- **Open:** the smallest orders (behind MKL; the divider port, so only an
  approximate reciprocal off the pivot chain would move it, and that is
  AVX-512-specific and not bit-reproducible against the divide); the rows
  below a panel's diagonal block are a triangular solve done as the rank-1
  sweep, store-bound -- solving them against the finished block in registers
  measured a further gain at `n >= 24` but changes the rounding order
  (LAPACK's potf2 + trsm), so it waits on a numerics decision; the strided
  (column-major upper / row-major lower) cases run the same blocked code but
  were not tuned or measured separately.

## sytrfnp / sytrsnp / sysvnp

The unpivoted LDL^T factorization, its solve, and the fused `?sysv`-style
driver (`docs/cbk_dsytrfnp_compact_design.md`); MKL has no compact
`sytrf`, and the `np` naming follows its `mkl_?getrfnp_compact`.

- **Implemented (design 6-8):** the square-root-free sweep, blocked
  recursively like `potrf` (8-column leaf panels with the `JB = 4` rank-1
  update confined to the panel, the trailing matrix updated per split level by
  the register-tiled rank-K update the two kernels share, with the pivot
  folded into the tile's weights here), structured like `potrf` (one kernel
  over transposed views; column-major lower and row-major upper contiguous);
  the solve as two unit-diagonal `trsm` group sweeps around a diagonal solve;
  `sysvnp` factoring and solving each group while its factor is cache-resident,
  bit-identical to the two calls. The upper convention is `A = U^T D U`
  (design 6.3), not `?sytrf`'s `U D U^T`.
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
- **Open:** the blocking was tuned on the contiguous case; the strided
  (column-major upper / row-major lower) cases run the same code untuned. The
  solve is the plain sweeps. No factorization-only benchmark (the potrf
  harness would port);
  no `sysvnp` vs `sytrfnp + sytrsnp` measurement on out-of-cache pools, the
  comparison that would quantify the fusion (design 6.8) -- the
  `bench_posv_compact` harness, which carries exactly that column for the
  Cholesky pair, would port directly.

## trsm

- **Implemented (design 2-6, 8.1):** full `side x uplo x transa x diag`,
  `alpha = 0` as the BLAS `B := 0` fast path. Column-major `side = 'L'` is
  tuned (4/2/1 register-blocked row-dot, contiguous axpy for the leftover
  column); the group kernel routes on views, so a transposed column-major view
  reaches the tuned path too (used by gels). Other combinations are strided.
- **Validated (design 7):** BLAS-free test vs a scalar `?trsm` (forward error
  and `||op(A) X - alpha B||`); MKL cross-check vs `mkl_?trsm_compact` over the
  full feature matrix plus the end-to-end solve.
- **Performance vs `mkl_?trsm_compact`** (measured before the view-based
  routing): level at `nrhs = 1`, ahead when `nrhs` is a multiple of the
  4-column block, in between for other counts; behind at the smallest orders,
  where the per-group overhead shows. Open: that small-`n` overhead,
  reciprocal-multiplying the diagonal in the blocked paths, tuning the strided
  kernel.
- **Scoped out:** no singularity check (a zero non-unit diagonal divides to
  `Inf`/`NaN`, as in BLAS).

## gels

- **Implemented (design 6):** LAPACK `?gels` for the compact format, both
  layouts, `trans in {N, T}`, any `m x n`. The four cases are one kernel: QR
  of the tall orientation of `A` (its view, transposed when `m < n`, which is
  `?gelqf` storage for free), `B := Q^T B` and `R X = B` (least squares), or `R^T Y = B`, `B := Q [Y; 0]` (minimum norm),
  one `for_each_group` body over the `geqrf`/`ormqr`/`trsm` group kernels. The
  MKL-style `work` is the per-group `tau` scratch (`lwork >= min(m,n) * V *
  ceil(nm/V)`) and holds `tau` on exit; the portable C API takes it as an
  explicit `taup` output. `(ap, tau)` feed `ormqr` for further right-hand
  sides. A fused factor-and-apply kernel for the least-squares case was
  retired (design 6.2): no arithmetic saved, and it evicted `B` from L1 at
  mid sizes.
- **Validated (design 7):** BLAS-free test vs a scalar `ref_gels` (`X`,
  factorization and `tau` elementwise) plus the defining properties formed
  independently, C-API validation, and the `min(m,n) = 0` quick return; MKL
  test vs per-matrix `LAPACKE_?gels` over every `(layout, trans)` and
  square/tall/wide shape (forward error `100 max(m,n) eps`, residual
  contracts, workspace query, factorization vs `LAPACKE_?geqrf`/`?gelqf`); and
  a cross-check vs the `mkl_?geqrf_compact -> cbk_?ormqr_compact ->
  mkl_?trsm_compact` pipeline on the same packed input.
- **Benchmarked:** the `cbk-gels` path of `bench_qr_compact` vs per-matrix
  `LAPACKE_dgels`: ahead at every order of the list, by the most at the
  smallest, where the dense routine's per-call overhead dominates. It runs the
  chain's group kernels, so gels-vs-cbk-batch there is the driver's overhead
  alone (BENCHMARKS.md).
- **Scoped out (design 6.6):** no rank-deficiency test (a zero diagonal of `R`
  divides to `Inf`/`NaN` in that lane), no `?lascl` rescaling, no pivoting.
- **Open:** row-major runs through the strided kernels. A whole-batch,
  library-threaded `gels` call has not been benchmarked against the
  caller-threaded per-group loop. The unblocked factorization is
  bandwidth-bound across the target range (its throughput is flat over the
  middle of the size list, where a compute-bound kernel's would still rise;
  it streams the trailing block per reflector), so a
  blocked (compact-WY) factorization is the next lever for `gels` and `geqrf`
  and the one that would make multiple right-hand sides pay.

## Project-wide

- **Precision coverage.** Every suite is templated on the scalar type and runs
  in FP64 and FP32 through the `compat<T>` / `mkl<T>` / `lapack<T>` dispatch
  of `tests/test_mkl_util.hpp`; FP32 cross-checks agree with `mkl_s*_compact`
  to ~1e-6, gated at 1e-4.
- **Planned: test against a real BLAS/LAPACK (issue #27).** The portable
  suites hand-roll their scalar reference routines. Instead, assume a library
  is present for testing -- `find_package(LAPACK REQUIRED)` -- and validate
  against it, rather than maintaining our own reference versions. This also
  opens the dense cross-checks (today MKL-only) to any BLAS/LAPACK stack.
- **Planned: LAPACK-style test coverage.** Adopt the testing approaches of the
  reference LAPACK repository (its `TESTING/LIN` drivers): `?latms`-style
  generators with prescribed condition number and spectral distribution, and
  the standard scaled residual gates, extending the ad-hoc generators of
  `test_compact_util.hpp`. Would close ormqr's design-7.3 stress-structure
  gap along the way.
- **Threading.** Each routine's group loop is an OpenMP `parallel for`
  (static, at most one thread per group), active for two or more groups and a
  work estimate above `CBK_OMP_MIN_FLOPS` (`5e4`, a compromise between two
  measured fork/join break-evens). Inside a caller's parallel region it stays
  serial unless nesting is enabled (`OMP_NUM_THREADS=8,2`). The factorization
  benchmarks hand the whole pool to one cbk call and drive the sequential MKL
  and LAPACK references from an equivalent outer loop; the solve benchmark
  keeps its pipeline per group (whole-pool passes measured slower).
  `gels`, `posv` and `sysvnp` are the fused per-group drivers that give
  library-side threading of a whole solve.
- **MKL Compact contract.** `.claude/mkl-compact-behavior.md` records what
  MKL's own compact routines were measured to do (`info` and `work` mandatory,
  `lwork` unchecked, `n*V` scratch per thread, internal threading only under
  the threaded layer). The wrappers follow it.
- **Alignment.** Compact buffers are correct at any `T` alignment on GCC and
  clang (issue #34); pack-width alignment is a performance recommendation only.
