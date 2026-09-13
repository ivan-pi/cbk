# API Design Document: `cbk_dgels_compact`

> Assisted-by: Claude:claude-fable-5

## 1. Overview

A batched least-squares / minimum-norm solve for full-rank systems
`op(A) X = B` in Intel MKL's Compact (interleaved-batch) format: the compact
form of LAPACK `?gels`, which MKL does not ship. It is the one-call form of the
toolkit's three-step pipeline

```
cbk_dgeqrf_compact(A -> H, tau);          // A = Q R
cbk_dormqr_compact('L','T', H, tau, B);   // B := Q^T B
cbk_dtrsm_compact ('L','U','N','N', R, B) // B := R^{-1} Q^T B = X
```

generalized to any `m x n` shape and to `A^T`, and run per group of `V`
interleaved matrices from factorization to solution while the group's buffers
are cache-resident. Three whole-batch calls stream the batch three times;
`bench_qr_compact` measured that 15-55% slower than keeping a group's pipeline
together. Target: many matrices with both dimensions in `3..500`, emphasis
below 170; column-major is the tuned path.

## 2. Syntax

```c
void cbk_dgels_compact (
    MKL_LAYOUT layout, char trans,
    MKL_INT m, MKL_INT n, MKL_INT nrhs,
    double * ap, MKL_INT ldap,
    double * bp, MKL_INT ldbp,
    double * work, MKL_INT lwork, MKL_INT * info,
    MKL_COMPACT_PACK format, MKL_INT nm
);
```

LAPACK [`?gels`](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/gels.html)'s
argument list with MKL's compact arguments added as `cbk_?ormqr_compact`
adds them: `layout` in front, `format` and `nm` at the back. `d` (double) and
`s` (single) precisions.

## 3. Description

For each `A` and its right-hand sides `B`, one of `?gels`'s four problems,
assuming `op(A)` has full rank:

| `trans` | shape | problem |
|---|---|---|
| `'N'` | `m >= n` | least squares, `min || B - A X ||_F` |
| `'N'` | `m < n` | minimum-norm solution of `A X = B` |
| `'T'` | `m >= n` | minimum-norm solution of `A^T X = B` |
| `'T'` | `m < n` | least squares for `A^T X = B` |

`B` is `max(m, n) x nrhs` per matrix (`ldb >= max(m, n)`, as `?gels` declares
it): on entry its first `rows(op(A))` rows hold the right-hand sides, on exit
its first `cols(op(A))` rows hold `X`. In the least-squares cases the remaining
rows hold the residual; a column's sum of squares over them is its residual sum
of squares, as `?gels` reports it. On exit `ap` holds the QR factorization
(`m >= n`, `?geqrf` storage) or the LQ factorization (`m < n`, `?gelqf`
storage), and `work` the reflector scalars `tau`, in compact format.

All matrices in a call share `trans`, `m`, `n`, `nrhs`, `ldap`, `ldbp`,
`layout` and `format`; the batch is processed one group of `V` matrices at a
time, `V` derived from `format`.

## 4. Input Parameters

* **`layout`**: `MKL_COL_MAJOR` (tuned) or `MKL_ROW_MAJOR`.
* **`trans`**: `'N'` solves `A X = B`; `'T'` (or `'C'`) solves `A^T X = B`.
* **`m`, `n`**: rows and columns of each `A` (`>= 0`).
* **`nrhs`**: columns of each `B` and `X` (`>= 0`).
* **`ap`**: compact buffer of the `nm` matrices `A` (`mkl_?gepack_compact`);
  overwritten with the factorization. Any alignment works; a 64-byte base
  keeps the SIMD sweeps off cache-line splits.
* **`ldap`**: leading dimension of each `A`, `>= m` (column-major) or `>= n`
  (row-major).
* **`bp`**: compact buffer of the `nm` matrices `B`, each `max(m, n) x nrhs`,
  packed with `mkl_?gepack_compact(layout, max(m,n), nrhs, ...)` and sized
  with `mkl_?get_size_compact(max(m,n), nrhs, format, nm)`. Rows beyond
  `rows(op(A))` need not be set on entry.
* **`ldbp`**: leading dimension of each `B`, `>= max(m, n)` (column-major) or
  `>= nrhs` (row-major).
* **`work`, `lwork`**: `lwork >= max(1, min(m, n) * V * ceil(nm / V))`, the
  size in scalars of a compact `tau` buffer for the batch
  (`mkl_?get_size_compact(min(m,n), 1, format, nm) / sizeof(double)`).
  `lwork = -1` is a workspace query: `work[0]` receives the required `lwork`
  and nothing else is touched. Unlike the other routines here, whose queries
  return `1`, `?gels` uses `work` as the `tau` scratch of its factorization,
  one slot per group so groups can run in parallel. Size it from this
  routine's own query; `lwork` is not otherwise checked.
* **`format`**: the pack format from `mkl_get_format_compact()`; selects `V`
  (SSE/AVX/AVX-512 -> 2/4/8 for FP64, 4/8/16 for FP32).
* **`nm`**: number of matrices in the batch.

## 5. Output Parameters

* **`bp`**: `X` in the first `cols(op(A))` rows of each `B`; in the
  least-squares cases the residual in the remaining rows.
* **`ap`**: the QR (`m >= n`) or LQ (`m < n`) factorization in LAPACK's
  `?geqrf` / `?gelqf` storage: `R` on and above (`L` on and below) the
  diagonal, the Householder vectors below (to the right of) it.
* **`work`**: the `min(m, n)` scalars `tau` per matrix, in compact format.
  `(ap, work)` is the `(H, tau)` that `cbk_?ormqr_compact` accepts, so
  further right-hand sides can be solved without refactoring (for `m < n` the
  LQ reflectors are the QR reflectors of `A^T`: the same buffer in the other
  layout, same `ldap`). On a query, `work[0]` is the required `lwork`.
* **`info`**: a scalar status, `0` on success (MKL leaves the compact `info`
  reserved). No argument checking and no rank-deficiency detection (section
  6.6); an unrecognized `format` selects no kernel and sets `info = -1`.

## 6. Design

### 6.1 Four cases, one tall QR over a view

`?gels` factors by QR when `m >= n` and by LQ when `m < n`. In the compact
kernels that is one code path, because the LQ factorization of `A` is the QR
factorization of `A^T`, and a `BatchView` transposes for free (its strides
swap). With `F` the tall orientation of `A` -- `F = A` if `m >= n`, `F = A^T`
if `m < n`; `p = max(m, n)`, `q = min(m, n)` -- `geqrf_compact_group` over
`F`'s view produces exactly `?gelqf`'s storage in the `m < n` case, and with
`F = Q [R; 0]` the four cases become two:

| cases | system | solved as |
|---|---|---|
| `m >= n, 'N'`; `m < n, 'T'` | `F X = B` (overdetermined) | `B := Q^T B`; `R X = B(0:q)`; rows `q..p-1` of `B` are the residual |
| `m >= n, 'T'`; `m < n, 'N'` | `F^T X = B` (underdetermined) | `R^T Y = B(0:q)`; `B(q:p) := 0`; `B := Q B` (`X = Q [Y; 0]`) |

Every step is an existing group kernel over views: `geqrf` for the
factorization, `ormqr` (backward sweep) for `Q B`, `trsm` for the triangular
solves. `trsm`'s group kernel routes on the views and reaches its tuned
row-dot path whenever `B`'s rows are contiguous -- for column-major `F`
directly, and for `m < n` through the transposed view, where `R` is the lower
triangle `L = R^T` of `A`'s own storage and the solve flips `uplo` and
`transa`. Row-major goes through the strided kernel.

### 6.2 The fused `[F | B]` reduction

In the overdetermined case `Q^T B` is not a separate sweep.
`geqrf_panel_compact_group`, a sibling of `geqrf_compact_group` sharing the
one-reflector step `geqrf_reflector`, applies each reflector, right after it
is built from column `kk` of `F`, to the trailing columns of `F` and to all
columns of `B`: the QR factorization of `[F | B]` truncated to `F`'s `q`
reflectors. The arithmetic is that of `ormqr('L','T')` (same reflectors, same
ascending order), so the result matches the separate sweep to rounding, but
each reflector is loaded once for both panels while it is in cache. `B` keeps
its own buffer and leading dimension: the panel is a second view, not an
augmented matrix.

It is a separate kernel rather than a flag on `geqrf_compact_group` because
an extra parameter, even one compiled out, changed GCC's code generation for
the plain path (frame and register allocation; measured 8-13% slower). With
the plain signature untouched its assembly matches the pre-`gels` kernel.

The underdetermined case cannot fuse: `Q` is applied after the triangular
solve, in descending order, so the factorization must be complete first. It
runs the three steps in sequence, still per group.

The fusion absorbs an `O(p q nrhs)` sweep into an `O(p q^2)` factorization,
so its value grows with `nrhs`. At `nrhs = 1`, `bench_qr_compact` measures the
one-call routine at parity with the three-step chain driven group by group
(`1.00x` geometric mean, `n = 10..100`, 4 threads, AVX-512); there the
routine's value is the interface, the rectangular cases, and threading the
whole solve inside the library.

### 6.3 Threading

One `for_each_group` over the batch, the body solving one group, threaded as
the toolkit's routines are ([threading.md](threading.md)): static-schedule
OpenMP, at most one thread per group, active only for two or more groups and
enough work, with the per-group estimate `(2 p q^2 + 4 p q nrhs + q^2 nrhs) * V`.

### 6.4 Workspace

`?gels` needs `min(m, n)` scalars of `tau` scratch per matrix (LAPACK's keeps
them in `work` too). Because groups run in parallel each group gets its own
slot: `lwork >= min(m, n) * V * ceil(nm / V)`, one compact `tau` buffer for the
batch. The routine does not allocate -- no routine here does, and a per-call
allocation inside a threaded loop would be the wrong place to start. On exit
the scratch is the `tau` of the factorization left in `ap`.

### 6.5 Padding and no argument checking

When `nm` is not a multiple of `V`, `mkl_?gepack_compact` fills the last
pack's unused slots with identity matrices, of `A` and `B` alike. An identity
`A` factors to `R = I`, `tau = 0`, so the fused reduction and the apply-`Q`
sweep are no-ops in those lanes and the triangular solve divides by `1`: the
kernel runs unmasked at full width without a `NaN` or a corrupted real lane.
Nothing is validated, as for every compact routine here and MKL's own; the
portable `?gels_compact` of `cbk.h` is the checked surface.

### 6.6 Numerical scope

Backward stable for a full-rank, well-conditioned `op(A)`, as the unblocked
`?gels` path is. `?gels`'s rank test (`info = i > 0` on an exactly zero
diagonal of `R`) does not vectorize across a pack and the compact `info` is a
scalar, so it is not performed: a zero diagonal divides through to `Inf`/`NaN`
in that lane, as in `?trsm`. A nearly rank-deficient `op(A)` gives the same
inaccurate solution it would from `?gels` (no pivoting; use a rank-revealing
method). `?gels`'s overflow/underflow rescaling (`?lascl`) and `?larfg`'s are
omitted, as throughout the toolkit; complex precisions are out of scope.

## 7. Testing

Matching LAPACK `?gels` to working precision is the bar; fusion, views and the
group-at-a-time driver are internal. All suites are CTest-registered and run
in FP64 and FP32.

* **Suite 1, vs dense `LAPACKE_?gels`**: every `(layout, trans)` over square,
  tall and wide `A` (all four cases in both layouts), diagonal-boosted `A` and
  random `B`, solved by `cbk_?gels_compact` and per matrix by
  `LAPACKE_?gels`. Gated: forward error of `X` (`100 * max(m,n) * eps`,
  relative to `||X||_1`); the defining property formed independently (residual
  sums of squares against `||B - op(A) X||^2`, or `op(A) X - B` for minimum
  norm); the workspace query; and the factorization in `ap` and `work`
  elementwise against `LAPACKE_?geqrf` or `LAPACKE_?gelqf`, which pins the
  `?gelqf` storage of the transposed-view case.
* **Suite 2, vs the three-step pipeline**: on the same packed square batch
  with `B = A X` for a known `X`, `cbk_?gels_compact` and
  `mkl_?geqrf_compact -> cbk_?ormqr_compact -> mkl_?trsm_compact` must
  agree in `ap`, `tau` (elementwise, cross-check tolerance) and `X`
  (`100 * n * eps`), and both must match the known `X`.
* **Portable self-test** (no BLAS): the templated kernel across
  `(T, V, layout, trans, shape)`, padded final packs, `nrhs = 1` and the
  smallest sizes, against a scalar `ref_gels` of the same unblocked steps
  (`X`, factorization and `tau` gated at `~eps`); the defining properties
  formed without the reference (normal equations and residual rows for least
  squares; `op(A) X = B` and agreement with `X = op(A)^T Z`,
  `(op(A) op(A)^T) Z = B`, for minimum norm); the portable C API's argument
  validation; the workspace query; and `min(m, n) = 0` (`B := 0`).

## 8. Implementation

* **MKL-style API** (`cbk_compat.h`): `cbk_?gels_compact`, unwrapping
  `MKL_COMPACT_PACK -> V`, instantiated on `MKL_INT`; no checking; `lwork = -1`
  answers the query.
* **Portable C API** (`cbk.h`): `?gels_compact` with an explicit `V`,
  no MKL dependency, LAPACK-style `info = -j` validation, and -- like the rest
  of that surface -- an explicit `taup` output sized as `?geqrf_compact`'s
  instead of the `work`/`lwork` protocol.
* **Kernel** (`src/cbk_gels_compact.hpp`): `gels_compact_group<T,V>` over the
  tall view `F` and the right-hand-side view `B`, driven by `gels_compact<T,V>`
  (one `for_each_group`), which both adapters call. It owns no arithmetic:
  `geqrf_panel_compact_group` / `geqrf_compact_group`, `ormqr_compact_group`,
  and `trsm`'s group kernel do the work; `gels_lwork` is the workspace rule.

The routine is `?gels` minus its scaling and rank test, over the unblocked
kernels the toolkit already validates, and the `(ap, work)` it leaves behind
interoperate with `cbk_?ormqr_compact` and `cbk_?trsm_compact` for
anything the one call does not cover.
