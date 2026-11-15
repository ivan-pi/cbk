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

Each system is solved by the standard BLAS `?trsm` substitution, executed `V`
matrices at once. Because Compact format interleaves the `V` matrices so that
element `(i,j)` of all `V` is contiguous, the scalar algorithm lifts almost
verbatim with `double -> V`-wide vector: every `*` and `-` becomes a lane-wise
SIMD operation over `V` independent matrices. There is no data-dependent
branch to mask, so the lift is direct.

There is one kernel: the forward substitution `L X = alpha B` for a lower
triangular `L` (`trsm_lower_group`), written over `BatchView`s. The sixteen
`side x uplo x transa x layout` combinations reach it by choosing the views
(`trsm_compact_group`), with no arithmetic of their own:

| case | view |
|------|------|
| `side = 'R'`: `X op(A) = alpha B` | the left solve of the transposed system `op(A)^T X^T = alpha B^T`: `B` transposed (a stride swap) |
| the operator `M` of that left solve is `A^T` | `A` transposed |
| `M` upper triangular | `A` and `B` with their index ranges reversed (`BatchView::reversed`: negative strides from the last element), which reads `M` as lower triangular and turns the back substitution into the forward sweep |
| row-major | the strides the views were built with |

Reversing `B` also reverses the order of its columns; they are independent, so
that is harmless.

### 6.2 Register tiles

The sweep is done in `IB x JB` tiles of `X` (rows `i .. i+IB-1` of the columns
`j .. j+JB-1`): a tile is loaded once (scaled by `alpha` unless it is 1),
accumulates its dot products with the rows already solved,

```
X(i+r, j+c) -= L(i+r, k) * X(k, j+c)      for k < i
```

in registers -- each pack of `L` feeds `JB` FMAs and each pack of `X` feeds
`IB`, against one of each per FMA for a column-at-a-time sweep -- is finished
against its own `IB x IB` corner of `L`, and is stored once. The inner loop
then runs at the FMA units' rate rather than the load ports'. The tile must fit
the register file (`IB * JB` accumulators, `IB` packs of `L`, one of `X`): it is
`4 x 6` with 32 SIMD registers and `3 x 4` with 16 (`simd_registers`); a tile
that does not fit is staged through the stack and measured slower than a
smaller one that does. The leftover rows and columns run as one narrower tile
of the same kernel (`for_remainder`), so a single right-hand side is a
`4 x 1` tile with four independent accumulators.

Column blocks are the outer loop: a block's columns stay cache-resident while
`L` streams past them once.

**Narrow blocks.** With one right-hand side a tile does only a few dozen
FMAs, each on its own pack of `L`, so what surrounds them counts. Every pack
of `L` is used once: the solve is bound by streaming `L`, and only the
per-tile overhead is left to cut. Two measures do that. The diagonal mode is
a template argument of the tile, dispatched once per group: run-time tests
of it inside the unrolled tile cost as many instructions as the tile's
arithmetic. And when `L`'s rows are adjacent (row stride `+-1`: a
column-major lower `L`, or the reversal of an upper one) the update walks one
pointer through `L`, with the tile's `IB` packs at constant offsets from it,
so each FMA takes a base + displacement operand instead of a base + index
one, which Intel cores split into two uops; other strides keep the indexed
form, which the compiler versions well for `L` contiguous along `k`. Two
alternatives measured slower: taller tiles for the narrow blocks (more
independent FMA chains), and forcing every tile inline, which slowed the
wide tiles at `V = 4`. Under icpx the two measures cost a few percent with
several right-hand sides (gcc and clang gain throughout).

**The diagonal** of a non-unit `A` is applied as a multiply by its reciprocal:
one divide per row of a tile, not one per element, issued before the tile's
accumulation so the divider runs underneath it instead of in the path of the
result. (Dividing each element kept the whole solve waiting on the divider.)
The result differs from the dividing substitution's in the last bit, within
the same backward-error bound (section 7). `TrsmDiag::scale` is an internal
third diagonal mode for the LDL^T solve: the tile scales its rows by the
reciprocal of the stored diagonal as it loads them and then sweeps as unit.

### 6.3 Layouts

Every layout and side runs the same tiles; they differ only in which operand
is walked contiguously. In the column-major left solve the right-hand side
columns are contiguous; `op(A) = A` reads rows of `A` (strided by `ldap`) and
`op(A) = A^T` its columns.

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
widths, and padded final packs, column-major, at shapes below, at and above
the register tile in both directions. The kernel accumulates in the
reference's order but multiplies by the diagonal's reciprocal where the
reference divides, so the two agree to working precision, not to the last
bit.
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
