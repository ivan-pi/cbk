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
  transposed view, `trans = 'C'` folds to `'T'`. The portable `?ormqr_compact`
  takes layout, side and trans like the MKL-style entry point, and its suite
  runs the layout x side x trans product against LAPACKE `?ormqr` and the
  explicit Q of `?orgqr` (LAPACK's dqrt03) on every stack.
- **Gap:** the stress structures of design 7.3 (`cond` knob, banded /
  row-scaled / clustered-scale inputs) are only partly covered; `geqrf`'s
  suite has the rank-deficient and near-collinear cases. The matrix-type
  table planned under "Testing" below would close it.
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

## Testing: the LAPACK practices still to adopt

The portable suites follow the reference LAPACK `TESTING/LIN` drivers where
it counts (`.claude/CLAUDE.md`, "Test ratios, one threshold"): every check
is the corresponding checker's dimensionless ratio against one `THRESH`,
every routine sweeps the small-dimension cross product, related routines
are checked against each other, the argument validation is table-tested
like `?errqr`, seeds are fixed, and a run reports every failure rather than
stopping at the first. What LAPACK does that the suites still do not, in
order of value:

- **The matrix-type table, on LAPACK's own generator.** LAPACK runs every
  path over a per-path table of matrix types (`dlatb4`, `dlattr`): diagonal,
  triangular, full; condition number 2, `sqrt(0.1/eps)` and `0.1/eps`; and
  singular or rank-deficient types that exercise the failure paths. The
  suites' inputs are benign by construction (`gen_boosted`, `gen_spd`,
  `gen_tri`, `gen_sym_ldlt`), so a kernel that is only correct on
  well-conditioned input passes them.
  The generator is not ours to write or maintain: LAPACK's `?latms`
  (singular values from `?latm1`'s `MODE`/`COND` table, random orthogonal
  transformations on both sides, `sym = 'P'` for SPD, `kl`/`ku` for
  triangular or banded) is reachable through LAPACKE. `LAPACKE_?latms` is
  declared in Netlib's `lapacke.h` and in MKL's `mkl_lapacke.h`; Netlib's
  `liblapacke.so.3` links `libtmglib.so.3` (Debian's `libtmglib-dev`, pulled
  in by `liblapack-dev`), and Debian's OpenBLAS stack uses that same
  LAPACKE, so it is there on every Linux stack CI runs; MKL exports the
  routine from its interface libraries. It is absent on Accelerate
  (accelerate-lapacke has no `tmglib` behind it) and on a self-contained
  OpenBLAS that ships its own LAPACKE (OpenBLAS does not build `MATGEN`
  into `libopenblas`). So: `cmake/FindLAPACKE.cmake` adds `LAPACKE_dlatms`
  to its link test and defines `CBK_HAVE_LATMS` on the target when it
  links; the type-table cases compile only then and a stack without it
  reports them skipped -- macOS today, everything else runs them. The
  fallback, should macOS coverage ever matter, is what `?latms` does
  inside -- `U diag(sigma) V^T` with `U`, `V` from `?geqrf` + `?orgqr` of
  Gaussian matrices, `sigma` from `?latm1`'s six modes -- about forty lines
  over calls the suites already make; recorded, not planned.
  The types, per path: QR and least squares (`geqrf`, `orgqr`, `ormqr`,
  `gels`) `dlatb4`'s eight -- diagonal, upper and lower triangular
  (`kl = 0` / `ku = 0`), full at the three condition numbers -- of which the
  two norm-scaled ones (`anorm` near underflow and overflow) are out of
  scope here (no overflow/underflow-safe reflector rescaling): run them as a
  boundary case that documents where the kernel stops, as `test_underflow`
  does, not as a gate. Cholesky (`potrf`, `potrs`, `posv`): `sym = 'P'` at
  the same condition ladder, plus the singular type LAPACK uses for
  `info > 0` (a lane whose trailing pivot is zero, beside the non-SPD lane
  the suite has). Unpivoted LDL^T (`sytrfnp`): conditioning is not its
  adverse axis -- bounded element growth is, and a near-zero leading minor
  is what breaks it -- so `gen_sym_ldlt` and the zero-pivot lane stay the
  contract-defining cases; a `sym = 'S'` condition ladder only where the
  minors stay bounded. Triangular solve (`trsm`): `?latms` with `kl = 0` or
  `ku = 0` gives a triangular factor with prescribed singular values;
  `dlattr`'s remaining types (unit diagonal, badly scaled rows, one large
  off-diagonal entry) are a few lines on top.
  What gates them needs almost nothing: the reconstruction and residual
  ratios are backward-stable and conditioning-independent, `forward_ratio`
  already discounts by `rcond`, and the elementwise comparisons against
  LAPACK's blocked drivers are skipped on rank-deficient types as
  `run_invariants` already does. The one prerequisite: `gels`'s comparison
  of `X` against LAPACK's `X` is undiscounted (fine on well-conditioned
  input), so it needs `rcond` of the triangular factor (`?trcon`, or
  `rcond1` on `R`) before ill-conditioned types go in. Closes `ormqr`'s
  design-7.3 gap.
- **Poison the outputs, check the padding lanes.** `compact_buffer` zero-fills
  the buffers a kernel writes, so a kernel that forgets an element whose
  correct value is 0 passes. LAPACK's `dqrt01` fills `Q` with
  `ROGUE = -1e10` before `?orgqr`; do the same for `tau`, the right-hand
  sides and the factor's other triangle where a routine must write, and
  after every call check that the padding lanes of a partial last group
  still hold what `pack_compact` put there (the identity for a factor, 0 for
  `tau`; finite values at least for an output). `orgqr`'s check 4 is the
  template; no other suite looks at the padding lanes after the call, and
  they are the one place the unmasked SIMD contract can go wrong silently.
- **Force the knobs.** LAPACK's `xlaenv` overrides `ilaenv` so the drivers
  exercise the unblocked, blocked and crossover paths deterministically. The
  analogue here is `for_each_group`'s gate: the suites reach the OpenMP path
  only through their `nm = 40` cases. One test, through any routine, should
  cover the shared driver on both sides of `parallel_min_flops`, at a single
  group, and from inside a caller's `omp parallel` region (the nested-level
  refusal), checking each result bit-identical to the serial run.
- **Heterogeneous lanes in one group.** The Cholesky and LDL^T suites put one
  poisoned lane beside good siblings; the QR path has no case where the
  lanes of a group differ in conditioning or structure (a rank-deficient or
  badly scaled lane beside well-conditioned ones). That is the batched
  analogue of LAPACK's type sweep, and with `?latms` per lane it is a loop
  over the type table inside one group.
- **The MKL suites onto the ratio convention.** They keep their relative
  gates, `solve_errors` and `orth_error`; move them to the three ratio forms
  and retire those two helpers.

Not planned: `dtest.in`-style configuration files (the sweeps are code, and
`small_dims` is the one list), the inverse, condition-estimate and
refinement-bound checks (`dpot03`, `dtrt01`, `?con`, `?rfs`: no cbk routine
to test), and LAPACK's `xlintst?` output format and `lapack_testing.py`.
