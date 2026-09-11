# API Design Document: `cqr_mkl_dpotrf_compact` / `cqr_mkl_dpotrs_compact` / `cqr_mkl_dposv_compact`

> Assisted-by: Claude:claude-opus-4.8 Claude

## 1. Overview

This extension provides a fast batched **Cholesky factorization** of a set of
symmetric positive-definite `n x n` matrices stored in Intel MKL's Compact
(interleaved-batch) format, the **solve** that completes a batched SPD linear
system from that factor, and the **fused driver** that does both in one pass
over the batch. The factorization mirrors `mkl_?potrf_compact` in signature and
semantics but is a fully portable, open implementation built on GNU vector types
-- specialized for SSE, AVX, and AVX-512 registers -- so it can be used,
studied, and tuned without depending on MKL's closed compact kernels. MKL ships
no compact `potrs` or `posv`, so -- like `cqr_mkl_?ormqr_compact` -- the solve
and the fused driver fill a gap in the compact ecosystem rather than shadowing
an MKL routine.

They join the QR routines already in this project (`cqr_mkl_?geqrf_compact`,
`cqr_mkl_?ormqr_compact`), sharing the same `pack<T,V>` GNU-vector machinery and
the same MKL-style API surface (`MKL_LAYOUT` + `MKL_COMPACT_PACK`). The
factorization pairs with either the solve companion or MKL's own
`mkl_?trsm_compact` to solve batched symmetric-positive-definite systems in the
compact format:

```
cqr_mkl_dpotrf_compact('L', A -> L);          // A = L L^T
cqr_mkl_dpotrs_compact('L', L, B);           // B := A^{-1} B = X
                                             //   (L z = B; L^T X = z)

cqr_mkl_dposv_compact ('L', A -> L, B);      // both, fused per group
```

(`cqr_mkl_dpotrs_compact` packages the two `mkl_dtrsm_compact('L','L','N')` /
`('L','L','T')` sweeps of the manual pipeline; section 6.7.)

The primary target is many small-to-medium matrices, with order in `3..500` and
the emphasis on sizes below 170. The tuned path is column-major, lower triangle
-- matching LAPACK and the natural Cholesky data flow.

## 2. Syntax

```c
void cqr_mkl_dpotrf_compact (
    MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n,
    double * ap, MKL_INT ldap, MKL_INT * info,
    MKL_COMPACT_PACK format, MKL_INT nm
);

void cqr_mkl_dpotrs_compact (
    MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, MKL_INT nrhs,
    const double * ap, MKL_INT ldap, double * bp, MKL_INT ldbp,
    MKL_INT * info, MKL_COMPACT_PACK format, MKL_INT nm
);

void cqr_mkl_dposv_compact (
    MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, MKL_INT nrhs,
    double * ap, MKL_INT ldap, double * bp, MKL_INT ldbp,
    MKL_INT * info, MKL_COMPACT_PACK format, MKL_INT nm
);
```

The factorization signature is identical to
[`mkl_dpotrf_compact`](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/mkl-potrf-compact.html),
so the routine is a drop-in alternative within the MKL Compact ecosystem. The
solve is LAPACK `?potrs` plus the three arguments MKL's compact routines add
(`layout`, `format`, `nm`), with `info` as a scalar status; the fused driver is
LAPACK `?posv`, i.e. the solve's signature with a non-const `ap` -- the same
relation `cqr_mkl_?sytrsnp_compact` / `cqr_mkl_?sysvnp_compact` bear to their
factorization. Both real precisions are provided (`d`/`s`). Unlike `?geqrf`,
none of the routines needs workspace, so -- like MKL's compact `potrf` -- there
are no `work`/`lwork` arguments.

## 3. Description

`cqr_mkl_?potrf_compact` forms the Cholesky factorization of each symmetric positive-definite
`n x n` matrix `A` in the batch,

```
A = U^T U,   if uplo = MKL_UPPER  (U upper triangular),
A = L L^T,   if uplo = MKL_LOWER  (L lower triangular),
```

with `L` (or `U`) having a positive diagonal. On exit each matrix is overwritten
exactly as `mkl_?potrf_compact` (and LAPACK `?potrf`) leave it:

* if `uplo = MKL_LOWER`, the lower triangle of `A` is overwritten by `L`; the
  strictly upper triangle is neither referenced nor modified;
* if `uplo = MKL_UPPER`, the upper triangle of `A` is overwritten by `U`; the
  strictly lower triangle is neither referenced nor modified.

`A` must be symmetric; only the triangle named by `uplo` is read, so the caller
need only populate that triangle. Before calling this routine, pack the matrices
with `mkl_?gepack_compact`; after it, call `mkl_?geunpack_compact` unless another
compact routine (e.g. `cqr_mkl_?potrs_compact` or `mkl_?trsm_compact`) will
consume the factors first.

`cqr_mkl_?potrs_compact` then solves `A X = B` for each matrix from that
factor, overwriting the `n x nrhs` RHS block `B` with `X` via two in-place
substitution sweeps:

```
L z = B;    L^T X = z      (MKL_LOWER)
U^T z = B;  U   X = z      (MKL_UPPER)
```

Both sweeps are the compact `trsm` with a non-unit diagonal (section 6.7).

`cqr_mkl_?posv_compact` performs the factorization and the solve for each group
of `V` matrices before moving to the next group (section 6.8). On exit `ap`
holds the factor exactly as `cqr_mkl_?potrf_compact` leaves it and `bp` holds
`X`; the result is bit-identical to the two separate calls.

**Constraint note.** As with all Compact routines, every matrix in the call
shares the same order `n`, leading dimension `ldap`, storage `layout`, and
`format`. The batch is processed one *pack* (group of `V` interleaved matrices)
at a time; `V` is derived from `format`.

Like ArmPL's `armpl_?potrf_interleave_batch`, this is an interleave-batch
Cholesky that -- unlike LAPACK -- does not verify that the inputs are SPD (section
6.2). Where ArmPL exposes the interleaved layout through explicit
`ninter`/`bstrd`/`istrd`/`jstrd` strides, this API abstracts it behind the
`MKL_COMPACT_PACK` parameter and `MKL_LAYOUT`, preserving symmetry with MKL's
native compact API: the batch is moved into the compact layout by
`mkl_?gepack_compact` and back out by `mkl_?geunpack_compact`, and this routine
consumes the packed buffer directly in between (as does `mkl_?trsm_compact` when
the factor feeds a solve, so no unpack is needed until the final result is read).

## 4. Input Parameters

Factorization (`cqr_mkl_?potrf_compact`):

* **`layout`** (`MKL_LAYOUT`): the in-memory storage order of each matrix --
  `MKL_COL_MAJOR` (tuned path) or `MKL_ROW_MAJOR`.
* **`uplo`** (`MKL_UPLO`): `MKL_LOWER` (factor and store the lower triangle `L`;
  tuned path) or `MKL_UPPER` (upper triangle `U`). The other triangle is not
  referenced.
* **`n`** (`MKL_INT`): the order of each `A` (`n >= 0`).
* **`ap`** (`double *`): the compact buffer of `nm` matrices `A`, packed with
  `mkl_?gepack_compact`. Overwritten in place with the Cholesky factor.
* **`ldap`** (`MKL_INT`): leading dimension of each matrix within the compact
  buffer (column stride for column-major, row stride for row-major), `>= n`.
* **`format`** (`MKL_COMPACT_PACK`): the pack format from
  `mkl_get_format_compact()`; selects the interleave width `V`
  (SSE/AVX/AVX-512 -> 2/4/8 for FP64, 4/8/16 for FP32).
* **`nm`** (`MKL_INT`): total number of matrices in the batch (`nm >= 0`).

Solve (`cqr_mkl_?potrs_compact`), in addition:

* **`nrhs`** (`MKL_INT`): number of right-hand sides (columns of `B`,
  `nrhs >= 0`).
* **`ap`** (`const double *`): the factored compact batch from
  `cqr_mkl_?potrf_compact` (not modified); `uplo` must match the factorization
  call.
* **`bp`** (`double *`): the compact RHS batch `B` (`n x nrhs` per matrix),
  overwritten with `X`.
* **`ldbp`** (`MKL_INT`): leading dimension of each `B` (`>= n` column-major,
  `>= nrhs` row-major).

Fused solve (`cqr_mkl_?posv_compact`): the solve's arguments, with **`ap`**
(`double *`) the symmetric positive-definite input batch, as for the
factorization.

**Buffer alignment.** Any base alignment is correct. For full speed, align the
bases to the pack width (64 B covers every format) so each SIMD access stays on
one cache line instead of splitting across two -- worth up to ~40% on small,
cache-resident sizes. `mkl_malloc(bytes, 64)` (the default of this project's
`mkl_alloc_bytes`) already does this.

## 5. Output Parameters

* **`ap`** (factorization, fused solve): the named triangle is overwritten with
  its Cholesky factor `L` or `U`, in Compact format; the other triangle is left
  untouched.
* **`bp`** (solve, fused solve): overwritten with the solution `X`, in Compact
  format.
* **`info`** (`MKL_INT *`): a single scalar status, `0` on success. MKL leaves
  the compact `info` reserved rather than reporting a non-SPD leading minor the
  way LAPACK `?potrf` does, and these routines do the same (section 6.2). The one
  value any of them can set is dispatch-level: an unrecognized `format` selects
  no kernel and sets `info = -1`.

## 6. Design Considerations & Compatibility

### 6.1 The algorithm: vectorized unblocked Cholesky (`potf2`)

The batch is factored with the unblocked LAPACK algorithm (`dpotf2`), executed
`V` matrices at a time. Because Compact format interleaves the `V` matrices so
that element `(i,j)` of all `V` is contiguous, the scalar algorithm lifts almost
verbatim with `double -> V`-wide vector: every `+`, `-`, `*`, `/`, and `sqrt`
becomes a lane-wise SIMD operation over `V` independent matrices. The
right-looking form is used, which for the lower triangle is, per pivot column
`j`:

```
d       = sqrt(A(j,j))                          // pivot (vector sqrt)
A(j,j)  = d
invd    = 1 / d
A(i,j) *= invd            for i > j             // scale the pivot column
A(i,jj)-= A(i,j)*A(jj,j)  for jj > j, i >= jj   // symmetric rank-1 trailing update
```

Only the lower trapezoid is ever touched, so the strictly-upper triangle passes
through untouched as `?potrf` requires. The trailing update -- the `O(n^3)` bulk
of the work -- is register-blocked `JB = 4` trailing columns at a time so each
pivot-column entry `A(i,j)` load is reused across four columns, exactly as the
`geqrf` trailing update. Blocked (`potrf`) factorization with `syrk`/`trsm`
panels is deliberately *not* used: for the target sizes the panels are short, and
the interleaved batch is already likely to saturate the vector units without the
extra blocking bookkeeping. `vsqrt<T,V>` (a short lane loop that GCC and Clang
lower to a single `vsqrt*`) is the only special function needed; it runs once per
column, negligible next to the `O(n^2)` scaling and `O(n^3)` update.

### 6.2 The pivot: `sqrt`, and no positive-definiteness check

Cholesky needs no branch-free trickery: unlike QR's `larfg`, its math has no
data-dependent branch. Scalar `dpotf2` has exactly one test -- `if (ajj <= 0 ||
isnan(ajj))` set `info = j` and stop, flagging a non-SPD leading minor -- and that
is the only thing that would diverge per lane across a pack. Both vendors'
interleave-batch Cholesky drop it: MKL leaves `info` "reserved for future use"
(its compact routines "skip error checking for performance reasons"), and ArmPL
"does not check that the input matrices are SPD; no error will be returned if any
`A_i` are not SPD." This routine does the same: it computes `d = sqrt(A(j,j))`
unconditionally.

The consequence is graceful "garbage in, garbage out": a genuinely non-SPD lane
has some pivot `A(j,j) <= 0`, so `sqrt` yields `NaN` (or the following `1/d`
yields `Inf`), and the poison propagates through that lane's factor. The caller
detects it by inspecting the unpacked diagonal, exactly as with MKL's compact
`potrf`. Early-exit with `info = j` is intentionally not provided -- it is
precisely the per-lane branch that does not vectorize across a pack.

### 6.3 Layouts and triangles: one kernel over transposed views

The kernel factors the *lower* triangle of a strided view
`A(i,j) = data[i*si + j*sj]` (strides in packs). Since `A` is symmetric, the four
`(layout, uplo)` combinations pair by transpose duality and are all the same code
with different strides:

* **Column-major + `MKL_LOWER`** is the view itself (`si = 1`, `sj = ldap`): a
  factor column is contiguous in the compact buffer (consecutive rows one
  `V`-wide pack apart), so the pivot sqrt, the column scaling, and the rank-1
  trailing update all walk contiguous packs. Its transpose dual, **row-major +
  `MKL_UPPER`**, has the same memory picture (`U^T` stored row-major *is* `L`
  stored column-major), so it takes the same strides.
* **Column-major + `MKL_UPPER`** and **row-major + `MKL_LOWER`** are the
  transposed view (`si = ldap`, `sj = 1`): the factor runs across the
  non-contiguous axis. Supported for MKL compatibility, correctness-first; the
  strided sweep is not separately SIMD-tuned.

Only the view's lower trapezoid is read or written, so the strictly-opposite
triangle of the named storage passes through untouched, as `?potrf` requires.

### 6.4 Padding and SIMD semantics

When `nm` is not a multiple of `V`, `mkl_?gepack_compact` fills the unused slots
of the last pack with identity matrices. The Cholesky factor of the identity is
the identity (`L = I`, all pivots `1`, no off-diagonal fill), so the padded lanes
compute a mathematical no-op and the kernel runs the whole final pack unmasked at
full width without corrupting real data. Because the pivot path is
unconditional, the identity flows through it with no lane mask -- unlike `geqrf`,
whose `larfg` needs a mask to neutralize padded columns. The same holds through
the solve: padded factor slots are identities (unit diagonal, so the non-unit
diagonal divide never hits zero), and the padded columns of `B` pass through the
sweeps unchanged (and are never read back).

### 6.5 No argument checking (Compact convention)

Like MKL's own compact routines -- which "skip error checking for performance
reasons" and make "the user responsible for passing correct parameters" --
`cqr_mkl_?potrf_compact` validates no arguments and writes a single scalar
`info = 0`. The only failure it can report is dispatch-level: an unrecognized
`format` has no kernel to run and sets `info = -1` (section 5).

### 6.6 Numerical scope

The unblocked `potf2` is backward stable for symmetric positive-definite input to
working precision across the target range, matching LAPACK `?potf2` element for
element (the SPD Cholesky factor with positive diagonal is unique, so agreement
is expected far below the backward-error bound -- see 7.1). Three limits are the
deliberate scope of this routine:

* **Positive definiteness is assumed, not enforced.** Non-SPD and
  borderline-semidefinite inputs -- where rounding can drive a true-zero pivot
  slightly negative -- poison their lane with `NaN`/`Inf` instead of taking a
  safeguarded path or reporting `info = j`.
* **Matrices scaled near underflow/overflow.** Following Intel's stated
  [numerical limitations for Compact BLAS and Compact LAPACK
  routines](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/numerical-limits-compact-blas-compact-lapack.html),
  the compact factorization does not provide safe handling of values near
  underflow/overflow; such inputs are out of scope.
* **Real precisions only; no pivoting.** Single and double (`s`/`d`); the
  Hermitian complex variants (`c`/`z`, `A = U^H U`) are out of scope, as they are
  for the QR routines. The factorization is unpivoted -- LAPACK `?potrf` is
  unpivoted too, and symmetric (diagonal) pivoting (`?pstrf`) needs comparisons
  that do not vectorize across a pack.

### 6.7 The solve: two non-unit `trsm` sweeps

`cqr_mkl_?potrs_compact` completes the batched solve from the factor. Its two
triangular sweeps are the existing compact `trsm` *group* kernels invoked with a
*non-unit* diagonal -- `L z = B; L^T X = z` for the lower factor, `U^T z = B;
U X = z` for the upper -- so column-major runs on `trsm`'s tuned side-left
row-dot path and row-major on its strided kernel (a single right-hand side,
contiguous in either layout, reaches the tuned path in both), exactly as for
`cqr_mkl_?sytrsnp_compact` (whose diagonal-solve middle step Cholesky does not
need: the factor's diagonal is the divisor of the sweeps themselves). No
workspace is needed and `B` is overwritten in place.

A caller who prefers to compose the solve manually can equally run
`cqr_mkl_?trsm_compact` (or `mkl_?trsm_compact`) twice, as in section 1 --
`potrs` packages exactly that composition behind the LAPACK `?potrs` argument
list. Consistent with the factorization's no-check contract, a factor lane
poisoned by non-SPD input (NaN/Inf on the diagonal) propagates into that lane's
solution rather than reporting an error.

### 6.8 The fused solve: factor and solve per group

`cqr_mkl_?posv_compact` is the `?posv` of the pair, and it exists for
throughput, not for arithmetic: called separately, `potrf` streams the whole
batch once and `potrs` streams it again, so for batches that exceed the cache
every factor is written out and read back -- the 15-55% whole-pool penalty the
project's solve benchmark measured (`PLANS.md`). The fused driver runs, for
each group of `V` matrices, the factorization group kernel immediately followed
by the solve group kernel, while the group's factor is still in cache, and
threads the whole solve as one `for_each_group` loop with the combined flop
estimate. Because it calls the *same* group kernels in the *same* order on the
*same* data, its factor and its `X` are bit-identical to the two separate calls
-- a property the test suites gate (section 7.5). The design (and its
rationale) is `cqr_mkl_?sysvnp_compact`'s, applied to the Cholesky pair; see
`docs/cqr_mkl_dsytrfnp_compact_design.md` section 6.8. `bench_posv_compact`
measures exactly this fused-vs-two-step column: `~1.0x` while the pool is
cache-resident, `1.1-1.3x` once it is not (orders 32-96 on 134-300 MB pools;
`examples/BENCHMARKS.md`).

## 7. Testing and Validation Methodology

Correctness is checked against standard dense LAPACK. SIMD, blocking, and the
choice of view strides are internal strategies only: the returned factor must
satisfy the same invariants as an unbatched `?potrf`.

### 7.1 Suite 1 -- Factorization invariants vs dense LAPACK

For each `(V, uplo, layout)`, a batch of random SPD `A` is generated with a
controlled condition number (`A = Q diag(logspace(0, -cond, n)) Q^T`, or the
cheaper `A = M^T M + n*I` with a `cond` diagonal-scaling knob), factored by the
routine under test, unpacked, and checked per matrix against the LAPACK Cholesky
contract:

* **Reconstruction residual.** Form the factor product (`L L^T` for lower,
  `U^T U` for upper) and gate `|| L L^T - A ||_1 / ||A||_1 <= 20 * n * eps`.
* **Triangularity / untouched triangle.** Confirm the strictly-opposite triangle
  of `ap` is bit-for-bit unchanged from the input (the routine must not reference
  or write it), which simultaneously gates that the factor is triangular.
* **Elementwise vs LAPACK (gated).** Unlike QR reflectors, the SPD Cholesky
  factor is *unique* (positive diagonal), so the elementwise difference of the
  factor vs `LAPACKE_dpotrf` is a sharp, meaningful signal and is gated, not just
  printed: `|| L_cqr - L_lapack ||_1 / ||L_lapack||_1 <= 20 * n * eps`.

### 7.2 Suite 2 -- Cross-check vs `mkl_dpotrf_compact`

The same packed batch is factored by both `cqr_mkl_dpotrf_compact` and the native
`mkl_dpotrf_compact`, and the two compact `ap` buffers are compared elementwise at
a small fixed tolerance (`1e-9`). The two implementations do not share an order of
arithmetic operations -- MKL's is closed -- so exact agreement is neither expected
nor required; the tolerance simply confirms they compute the *same* Cholesky
factor. On well-conditioned SPD inputs the observed agreement is far tighter, at
the `1e-14` level.

### 7.3 Suite 3 -- End-to-end solve `AX = B`

Closing the pipeline: `B = A X` for a known SPD `A` and known `X`, then
`cqr_mkl_dpotrf_compact('L') -> mkl_dtrsm_compact('L','L','N') ->
mkl_dtrsm_compact('L','L','T')` (forward then back substitution) must recover
`X`. Gate the forward error `Xhat - X` and the residual `A Xhat - B` at
`100 * n * eps` (relative to the matrix L1 norm). This validates `?potrf` in situ
with the rest of the compact toolkit.

### 7.4 Portable self-test

A self-contained test validates the templated kernels directly against a scalar
reference (`ref_potf2`) across `(T, V)` combinations, both `uplo`, and partial
(padded) final packs, plus the end-to-end portable solve (`?potrf_compact` +
`?potrs_compact` recovering a known `X`), the fused `?posv_compact`
bit-identical to it, and the LAPACK-style argument validation of the three
portable C APIs. It needs no external libraries at all; only the suites above
require an MKL installation (for the Compact API). BLAS and LAPACK themselves
are assumed available, as they are on most platforms -- it is the MKL Compact
extension that must be installed separately.

### 7.5 Suite 4 -- The packaged solve: `?potrs` and the fused `?posv`

The end-to-end contract of section 7.3, run through the packaged routines over
both `uplo` and both layouts: `B = A X` for known `X`, then
`cqr_mkl_?potrf_compact -> cqr_mkl_?potrs_compact` must recover `X` (forward
error and system residual gated at `100 * n * eps`, as in suite 3), including
padded partial groups and RHS counts that exercise `trsm`'s 4/2/1 column
blocks. On the same packed input `cqr_mkl_?posv_compact` must reproduce the
two-step factor and `X` **bit-for-bit** over the whole compact buffers, padded
lanes included (section 6.8).

## 8. Implementation Strategy

Modern C++ (C++17) templated on scalar type `T` and interleave width `V`, exposed
through `extern "C"` for the FFI-stable surfaces, built on the project's
`cqr::detail::pack<T,V>` GNU-vector machinery and a lane-wise `vsqrt<T,V>` helper.

### 8.1 API boundary

* **MKL-style API** (`cqr_mkl_ext.h`, the primary surface):
  `cqr_mkl_?potrf_compact` / `cqr_mkl_?potrs_compact` / `cqr_mkl_?posv_compact`,
  unwrapping `MKL_COMPACT_PACK -> V` and `MKL_UPLO`/`MKL_LAYOUT`, instantiated
  on `MKL_INT` so ILP64 dimensions are not narrowed.
* **Portable C API** (`cqr_compact.h`): `?potrf_compact` / `?potrs_compact` /
  `?posv_compact`, taking `char layout` (`'C'`/`'R'`), `char uplo`
  (`'L'`/`'U'`), an explicit interleave width `V`, and no MKL dependency, with
  LAPACK-style `info = -j` argument validation.
* **Templated kernels**: the per-group kernel `potrf_compact_group<T,V>`
  factors the lower triangle of a `BatchView` (the driver builds the view so
  that the named triangle's storage appears as that lower triangle, section
  6.3); `potrs_compact_group<T,V>` runs the two sweeps on one group; the
  all-groups drivers `potrf_compact` / `potrs_compact` / `posv_compact` (the
  entry points both C adapters call) loop over groups with `for_each_group`.

### 8.2 Source layout

| File | Role |
|------|------|
| `src/cqr_potrf_compact.hpp` | Templated SIMD Cholesky kernel (vectorized `potf2`; scalar `T`, width `V`). |
| `src/cqr_potrs_compact.hpp` | Templated solve: two non-unit `trsm` group sweeps, and its driver. |
| `src/cqr_posv_compact.hpp` | The fused factor-and-solve driver over both group kernels. |
| `src/cqr_compact.cpp` | Portable `?potrf_compact` / `?potrs_compact` / `?posv_compact` C entry points (runtime `V` -> compile-time dispatch, `info = -j`). |
| `src/cqr_mkl_ext.cpp` | Unwraps `MKL_COMPACT_PACK` -> `V` and `MKL_UPLO`/`MKL_LAYOUT`, calls the kernels. |
| `tests/test_cqr_potrf_compact.cpp` | Self-contained correctness test vs a scalar `potf2` reference (no BLAS), incl. the potrs/posv solve. |
| `tests/test_cqr_potrf_mkl.cpp` | MKL + dense-LAPACK validation (residual, uniqueness, cross-check, solves, fused bit-identity). |
