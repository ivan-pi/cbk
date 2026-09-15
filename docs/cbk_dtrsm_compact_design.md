# API Design Document: `cbk_dtrsm_compact`

> Assisted-by: Claude:claude-opus-4.8

## 1. Overview

This extension provides a batched **triangular solve with multiple right-hand
sides** for a set of matrices stored in Intel MKL's Compact (interleaved-batch)
format. It mirrors `mkl_?trsm_compact` in signature and semantics but is a fully
portable, open implementation built on GNU vector types.

Batched triangular solves are a core building block of dense linear algebra: every
direct `A X = B` solve finishes with one or two of them, and which triangle,
transpose, and diagonal it needs depends on the factorization --

* **LU** (`getrf`/`getrs`): forward-substitute the unit lower factor `L`
  (`L Y = P B`), then back-substitute the upper factor `U` (`U X = Y`).
* **Cholesky** (`potrf`/`potrs`): for `A = R^T R`, solve `R^T Y = B` then
  `R X = Y` (the lower `A = L L^T` variant is the mirror image).
* **QR** (`geqrf`/`ormqr`): form `Q^T B`, then back-substitute the upper factor
  `R` (`R X = Q^T B`).

Between them these span the whole `uplo x transa x diag` matrix, which is why
`cbk_?trsm_compact` implements all of it. Paired with cbk's compact
`geqrf`/`ormqr`, for instance, it closes a batched QR solve entirely in the open
compact stack:

```
cbk_dgeqrf_compact(A -> H, tau);          // A = Q R
cbk_dormqr_compact('L','T', H, tau, B);   // B := Q^T B
cbk_dtrsm_compact ('L','U','N','N', R, B) // B := R^{-1} Q^T B = X
```

MKL's compact pack/unpack helpers (`mkl_?gepack_compact`,
`mkl_get_format_compact`, ...) are still used to move data in and out of the
interleaved layout; only the *computation* is now cbk's. The primary target is
many small-to-medium matrices (both dimensions in `3..500`, emphasis below 170),
matching the rest of the toolkit; the tuned path is column-major, `side = 'L'`,
upper-triangular non-unit -- the back-substitution that every one of the
factorizations above ends with (`U X = Y`, `R X = Y`, `R X = Q^T B`).

## 2. Syntax

```c
void cbk_dtrsm_compact (
    MKL_LAYOUT layout, MKL_SIDE side, MKL_UPLO uplo,
    MKL_TRANSPOSE transa, MKL_DIAG diag,
    MKL_INT m, MKL_INT n, double alpha,
    const double * ap, MKL_INT ldap,
    double * bp, MKL_INT ldbp,
    MKL_COMPACT_PACK format, MKL_INT nm
);
```

The signature is identical to
[`mkl_dtrsm_compact`](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/mkl-trsm-compact.html),
so the routine can be used as a drop-in replacement for MKL's routines. Both
real precisions are provided: `cbk_dtrsm_compact` (double) and
`cbk_strsm_compact` (single).

## 3. Description

For each matrix in the batch the routine solves, in place, one of the triangular
systems

```
op(A) X = alpha B     (side = MKL_LEFT)
X op(A) = alpha B      (side = MKL_RIGHT)
```

where `alpha` is a scalar, `A` is a unit or non-unit, upper or lower triangular
matrix, `B` is the `m x n` right-hand-side matrix, and
`op(A) = A` (`MKL_NOTRANS`) or `op(A) = A^T` (`MKL_TRANS`). `A` has order
`s = m` when `side = MKL_LEFT` and `s = n` when `side = MKL_RIGHT`. On exit `B`
is overwritten by the solution `X`.

`op(A) = A^H` (`MKL_CONJTRANS`) is accepted and, for the real types handled
here, is identical to `A^T`; it is folded to the transpose path. When
`alpha = 0`, `A` is not referenced and `B` need not be set on entry -- `B` is set
to `0` on exit, exactly as reference BLAS `?trsm` defines it.

**Constraint note.** As with all Compact routines, every matrix in the call
shares the same dimensions (`m`, `n`), leading dimensions (`ldap`, `ldbp`),
storage `layout`, `format`, and the same `side`/`uplo`/`transa`/`diag`. The batch
is processed one *pack* (group of `V` interleaved matrices) at a time; `V` is
derived from `format`.

**Relationship to MKL and ArmPL.** The behavior captured here is the intersection
of the two production batched triangular solvers, expressed through MKL's compact
interface:

* **Intel MKL `mkl_?trsm_compact`** -- same equation set and the same
  `layout / side / uplo / transa / diag / m / n / alpha` parameters over the
  Compact (interleaved) format, selected by an opaque `MKL_COMPACT_PACK`.
  `cbk_?trsm_compact` matches its signature verbatim.
* **Arm Performance Libraries `armpl_dtrsm_interleave_batch`** -- the same
  mathematical operation (`op(A^i) X^i = alpha B^i` and the right-side variant,
  with upper/lower and unit/non-unit `A^i`) over Arm's interleave-batch format.
  ArmPL exposes the interleaving explicitly through `ninter`/`nbatch` and
  arbitrary batch/row/column strides (`bstrd_`, `istrd_`, `jstrd_`); as with
  `cbk_?ormqr_compact`, this API keeps MKL's abstraction instead, hiding the
  interleave width and strides behind `MKL_COMPACT_PACK` + `MKL_LAYOUT` + the
  compact leading dimensions, for perfect symmetry with MKL's native compact
  ecosystem. ArmPL's real interface has no conjugate-transpose; neither do the
  real types here.

## 4. Input Parameters

* **`layout`** (`MKL_LAYOUT`): the in-memory storage order of every matrix in the
  batch, `MKL_COL_MAJOR` or `MKL_ROW_MAJOR`.
* **`side`** (`MKL_SIDE`): `MKL_LEFT` solves `op(A) X = alpha B`; `MKL_RIGHT`
  solves `X op(A) = alpha B`.
* **`uplo`** (`MKL_UPLO`): `MKL_UPPER` -- `A` is upper triangular; `MKL_LOWER` --
  `A` is lower triangular. The opposite triangle of `A` is never referenced.
* **`transa`** (`MKL_TRANSPOSE`): `MKL_NOTRANS` -> `op(A) = A`; `MKL_TRANS` ->
  `op(A) = A^T`; `MKL_CONJTRANS` -> `A^H` (== `A^T` for the real types, folded to
  the transpose path).
* **`diag`** (`MKL_DIAG`): `MKL_NONUNIT` -- the diagonal of `A` is used;
  `MKL_UNIT` -- `A` is assumed to have a unit diagonal and its diagonal entries
  are not referenced.
* **`m`** (`MKL_INT`): rows of `B` (and the order of `A` when `side = MKL_LEFT`),
  `m >= 0`.
* **`n`** (`MKL_INT`): columns of `B` (and the order of `A` when
  `side = MKL_RIGHT`), `n >= 0`.
* **`alpha`** (`double`): scalar multiplying `B`. When `alpha = 0`, `A` is not
  referenced and `B` is set to `0`.
* **`ap`** (`const double *`): the compact buffer of `nm` triangular matrices
  `A` (order `s`, with `s = m` for `MKL_LEFT` and `n` for `MKL_RIGHT`), packed
  with `mkl_?gepack_compact`.
* **`ldap`** (`MKL_INT`): leading dimension of each `A` within the compact
  buffer, `>= max(1, s)`.
* **`bp`** (`double *`): the compact buffer of `nm` right-hand-side matrices `B`
  (`m x n`); overwritten in place with `X`.
* **`ldbp`** (`MKL_INT`): leading dimension of each `B` within the compact
  buffer, `>= max(1, m)` (column-major) or `>= max(1, n)` (row-major).
* **`format`** (`MKL_COMPACT_PACK`): the pack format from
  `mkl_get_format_compact()`; selects the interleave width `V`
  (SSE/AVX/AVX-512 -> 2/4/8 for FP64, 4/8/16 for FP32).
* **`nm`** (`MKL_INT`): total number of matrices in the batch.

## 5. Output Parameters

* **`bp`**: overwritten with the solution `X` (of `op(A) X = alpha B` or
  `X op(A) = alpha B`), in Compact format.

An unrecognized `format` selects no kernel, so the call is a silent no-op.

## 6. Design Considerations & Compatibility

### 6.1 The algorithm: vectorized batched substitution

Each system is solved by the standard BLAS `?trsm` substitution -- scale
`B := alpha B`, then forward- or back-substitute one row (`side = 'L'`) or one
column (`side = 'R'`) of the solution at a time -- executed `V` matrices at once.
Because Compact format interleaves the `V` matrices so that element `(i,j)` of
all `V` is contiguous, the scalar algorithm lifts almost verbatim with
`double -> V`-wide vector: every `*`, `-`, and `/` becomes a lane-wise SIMD
operation over `V` independent matrices. There is no data-dependent branch to
mask, so the lift is direct.

The eight `side x uplo x transa` combinations collapse to a choice of sweep
direction and which index of `A` is read:

| side | uplo  | transa | sweep over the solved index | reads      |
|------|-------|--------|-----------------------------|------------|
| L    | lower | N      | forward `i = 0 .. m-1`      | `A(i, r)`  |
| L    | upper | N      | backward `i = m-1 .. 0`     | `A(i, r)`  |
| L    | upper | T      | forward                     | `A(r, i)`  |
| L    | lower | T      | backward                    | `A(r, i)`  |

`side = 'R'` mirrors these along the columns of `B` (solving one column of `X`
per step and reading `A(r, j)` / `A(j, r)`), so the arithmetic is shared. The
diagonal divide (`/ A(i,i)`) is skipped entirely when `diag = MKL_UNIT`.

### 6.2 Tail-blocked register blocking, and the 1-column axpy tail

For `op(A) = A` the row-dot reads `A(i, r)` -- a *row* of `A`, strided by `ldap`
in column-major storage -- so once `A` spills out of cache it prefetches poorly.
Register-blocking the RHS columns amortizes that: each `A` load serves the whole
block. The `side = 'L'` column-major path therefore sweeps the RHS in **descending
blocks of 4, then 2, then 1** (since `n mod 4 ∈ {0,1,2,3}`, the tail is at most
one 2-block plus one 1-block), so the 2- and 3-column tails reuse each `A` load
rather than streaming it per column as a 4-only block with a one-column remainder
would. `JB` and the `uplo/trans/diag` config are compile-time, so the
accumulators land in registers, the loops unroll, and the config branches vanish.
This lifts the awkward counts (`n = 2, 3, 5, 6, 7, ...`) to parity-or-better with
the clean multiples of 4.

The exception is a single **`op(A) = A` leftover column** (`n = 1` -- a single-RHS
solve such as QR's `R x = Q^T b` -- and the 1-tail of odd `n`): with one column
there is no reuse to amortize the strided read, so it is instead solved by a
**column-oriented (gaxpy) sweep** (`trsm_axpy_col`) reading *column* `k` of `A`
contiguously. `op(A) = A^T`
reads `A(r, i)` = a column already, so it always uses the row-dot. Every path
computes the same result -- this is a throughput choice only.

Two further points decide throughput at the larger orders and at many
right-hand sides.

**One reciprocal per pivot row.** The non-unit diagonal used to divide once per
`(row, RHS column)`. The divider is the one port that is not pipelined, so at
`nrhs` columns that is `n * nrhs` unpipelined divides against `n^2 * nrhs / 2`
fused multiply-adds -- a ratio of `2/n`, which at the small orders this library
is for is not a rounding error but the kernel. The diagonal is now inverted
once per row and the row's columns multiplied, in both the row-dot and the
axpy tail so a column's answer does not depend on which block of the 4/2/1
split it landed in. It costs one extra rounding per solved entry, which the
suites' test ratios cover.

**A blocked sweep above `trsm_block_min`.** Unblocked, the substitution is a
chain of rank-1 passes: every pivot row touches all the rows still to come,
once. From order 40 up the sweep instead takes the pivots in blocks of
`trsm_nb`, solving each block's own small triangle with the row-dot and
applying its whole effect on the rows still to come as one **register-tiled
rank-`NB` update** -- the same structure the blocked `potrf` uses for its
trailing update. The tile is **2 rows x 4 RHS columns**: `2*4` accumulators
plus `2 + 4` operands is 14 vectors live, which fits the *16* architectural
vector registers of SSE and AVX as well as AVX-512's 32, and the kernel is
instantiated at all three widths. A 4x4 tile is faster where 32 registers
exist and spills where only 16 do (built for an AVX2 target it lost about 40%
at `V = 4`, 16 right-hand sides, order 128), so the smaller tile is the one
that holds across CPU generations; tuning a tile per micro-architecture is
explicitly out of scope. The blocked sweep is used when there is something for
the tile to amortize over -- a second RHS column, or `op(A) = A^T`, whose
unblocked form reduces each row into a single accumulator and runs at FMA
latency. A single column of `op(A) = A` is already the contiguous axpy above
and stays there.

### 6.3 Layouts and sides: one kernel, four routes

Every `(side, layout)` combination reaches the tuned kernel; the strided
`BatchView` kernel remains only as the fallback for a view with no unit stride
at all.

**Layout** is a template parameter of the tuned body, not a separate kernel.
Column-major makes a matrix column contiguous in the compact buffer,
row-major a row; that is the *whole* difference, and it is one index
expression (`trsm_ix`), selected at compile time so neither layout pays for the
other. The substitution, the blocking, the register tile and the reciprocal are
written once. Previously row-major fell to the stride-generic kernel, which has
no blocking or register tiling at all and ran several times slower than the
tuned path at many right-hand sides.

**Side** needs no kernel of its own either. `X op(A) = alpha B` transposes to
`op(A)^T X^T = alpha B^T`, a *left* solve on the transposed views with the
extents swapped -- and in compact storage a transposed view is a stride swap,
which `BatchView::transposed()` already expresses. `op(A)^T` is `A^T` when
`transa = 'N'` and `A` when it is `'T'`, which is exactly `op(A^T)` for the
*same* `transa`, and `A^T` is upper where `A` is lower. So `side = 'R'` reduces
to "transpose both views, swap `m` and `n`, flip `uplo`, keep `transa`" -- the
same identity that makes `ormqr`'s `side = 'R'` and `gels`'s LQ case the
kernels they already had.

The routing is then four branches on which of each operand's strides is unit
(B column-major or row-major, A stored the same way or as its transpose), all
landing in the same body. One branch per group, outside every kernel loop.

### 6.4 Padding and SIMD semantics

When `nm` is not a multiple of `V`, `mkl_?gepack_compact` fills the unused slots
of the last pack with identity matrices. For a triangular operand that identity
is itself triangular with a **unit diagonal** (`A(i,i) = 1`), so solving
`I X = alpha B` in a padded lane simply yields `X = alpha B`, and -- crucially --
the diagonal divide `/ A(i,i)` in those lanes divides by `1`, never by `0`. The
kernel can therefore run unmasked across the whole final pack at full width
without producing a NaN in the padding or corrupting real lanes. (This is the
`?trsm` analogue of the `geqrf` observation that the identity factors to
`tau = 0`.)

Alignment is a free performance lever: any alignment is correct, but aligning
each compact buffer's base to the pack width (64 B covers every format) keeps the
`V`-wide loads and stores off cache-line splits.

### 6.5 No argument checking (Compact convention)

Like MKL's own compact routines -- which "skip error checking for performance
reasons" and make "the user responsible for passing correct parameters"
(Intel MKL, *Numerical Limitations for Compact BLAS and Compact LAPACK
Routines*) -- `cbk_?trsm_compact` validates nothing. Because `?trsm` has no
`info`, there is not even a dispatch-level status: an unrecognized `format` is a
silent no-op. It is the caller's responsibility to pass valid parameters, and to
validate them beforehand if defensive checking is wanted.

### 6.6 Numerical scope

The solve is the direct substitution, backward stable to working precision for
the well-conditioned triangular factors a QR, Cholesky, or LU factorization
produces (e.g. `R` from a well-conditioned `A`). Consistent with BLAS
`?trsm` and `mkl_?trsm_compact`, the routine performs **no singularity check**:
a zero (or tiny) diagonal entry of a non-unit `A` divides through to `Inf`/`NaN`
exactly as reference BLAS would, and it is the caller's responsibility to pass a
non-singular triangular factor. As with the rest of the toolkit, no
overflow/underflow-safe scaling is attempted, and complex precisions (`c`/`z`)
are out of scope.

## 7. Testing and Validation Methodology

Matching standard BLAS `?trsm` to working precision is the minimum bar. SIMD,
blocking, and layout handling are internal strategies only: the returned `X`
must satisfy the same numerical invariants as an unbatched `?trsm`. Tolerances
are purely relative to the working precision. All suites below are
CTest-registered.

### 7.1 Suite 1 -- Portable, vs `cblas_?trsm`

`tests/test_trsm_compact.cpp` validates the templated kernel directly against
the library's `?trsm` (`cblas_?trsm` through `ref_trsm`, on any LAPACKE +
CBLAS stack: the same algorithm, one matrix at a time) over the full
`side x uplo x transa x diag` matrix, across precisions (FP32/FP64), interleave
widths, and padded final packs, column-major. The row-dot kernels run the
identical operation sequence one lane per matrix and so match the reference to
the last bit (relative error `0`); the column-axpy no-transpose kernel
accumulates in a different (but equally backward-stable) order and agrees to
working precision (~`1e-16`).
Two gates, both at a generous multiple of `eps` on the diagonal-boosted (well
conditioned) factors: the relative forward error of `X` against the reference
solve, and the solve's own residual `||op(A) X - alpha B||` formed with the
library's triangular multiply (`cblas_?trmm`) -- so a bug shared by the
reference and the kernel cannot pass unseen (the `?trsm` analogue of the reconstruction check
the `geqrf`/`potrf` self-tests apply).

### 7.2 Suite 2 -- MKL cross-check, vs `mkl_?trsm_compact`

`tests/test_trsm_mkl.cpp` packs a diagonal-boosted triangular batch and a random
RHS with the genuine MKL Compact API, then has `cbk_?trsm_compact` and
`mkl_?trsm_compact` each solve their own copy. The two compact result buffers are
compared elementwise over the full feature matrix
(`layout x side x uplo x transa x diag`), gated at a small relative tolerance.
This is the direct analogue of the `geqrf` suite's cross-check against
`mkl_dgeqrf_compact`.

### 7.3 Suite 3 -- End-to-end solve `AX = B`

The capstone: `B = A X` for a known `X`, solved by the fully open pipeline
`cbk_dgeqrf_compact -> cbk_dormqr_compact('L','T') ->
cbk_dtrsm_compact('L','U','N','N')`, which must recover `X`. Gate the forward
error `Xhat - X` and the residual `A Xhat - B` at `100 * n * eps` (relative to
the matrix L1 norm). This exercises `cbk_?trsm_compact` closing a complete
batched QR solve -- an open, independent implementation of the compact triangular
solve.

## 8. Implementation Strategy

Modern C++ (C++17) templated on scalar type `T` and interleave width `V`,
exposed through `extern "C"` for FFI-stable surfaces, reusing the existing
`cbk::detail::pack<T,V>` / `BatchView` GNU-vector machinery.

### 8.1 API boundary

Two C-linkage interfaces wrap the same templated kernel:

* **MKL-style API** (`cbk_compat.h`): `cbk_dtrsm_compact` /
  `cbk_strsm_compact` -- a drop-in for `mkl_?trsm_compact`, taking the MKL
  enums and `MKL_COMPACT_PACK`, with no argument checking (Compact convention).
* **Portable C API** (`cbk.h`): `dtrsm_compact` / `strsm_compact` -- an
  MKL-independent surface taking an explicit interleave width `V` and `char`
  selectors, with LAPACK/BLAS-style `info = -j` argument validation.

Both instantiate the kernel on `MKL_INT` (or `int`) so ILP64 dimensions are not
narrowed; the algorithm and its internal routines are described in section 6.
