# API Design Document: `cqr_mkl_dorgqr_compact`

> Assisted-by: Claude

## 1. Overview

Generate the explicit orthogonal factor `Q` of a compact QR factorization:
the compact form of LAPACK `?orgqr`, supplied by the `cqr` project on the
precedent of `cqr_mkl_?ormqr_compact`. It completes the standard QR trio
-- factor (`?geqrf`), apply `Q` implicitly (`?ormqr`), form `Q` explicitly
(`?orgqr`) -- for batches of small matrices in Intel MKL's Compact
(interleaved) format.

## 2. Syntax

```c
void cqr_mkl_dorgqr_compact (
    MKL_LAYOUT layout,
    MKL_INT m, MKL_INT n, MKL_INT k,
    double * ap, MKL_INT ldap,
    const double * taup,
    double * work, MKL_INT lwork, MKL_INT * info,
    MKL_COMPACT_PACK format, MKL_INT nm
);
```

LAPACK [`?orgqr`](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/orgqr.html)'s
argument list with MKL's compact arguments added as `cqr_mkl_?ormqr_compact`
adds them: `layout` in front, `format` and `nm` at the back. `work` and
`lwork` are kept because the reference LAPACK routine has them (the kernel
itself needs no scratch; the query answers `1`). `d` (double) and `s`
(single) precisions.

## 3. Description

For each matrix of the batch, the routine generates the first `n` columns
of the `m x m` orthogonal matrix

```
Q = H(0) H(1) ... H(k-1)
```

defined by the `k` elementary reflectors and scalars `tau` that
`?geqrf_compact` (or `mkl_?geqrf_compact`, or the factorization half of
`cqr_mkl_?gels_compact`) left in `ap` and `taup` -- in place: on exit the
`m x n` matrix in `ap` holds `n` orthonormal columns, `Q^T Q = I_n`.
LAPACK's shape constraint applies, `m >= n >= k >= 0`.

The two common calls:

| goal | call | buffer |
|---|---|---|
| thin `Q` (`m x q`, `q = min(m,n)` of the factorization) | `n = k = q` | the `?geqrf_compact` output of a tall/square `A`, consumed directly |
| full `Q` (`m x m`) | `n = m` | the `k` reflector columns copied into an `m x m` compact buffer first |

The factorization is destroyed: `R` (on and above the diagonal) is
overwritten along with the reflectors. A caller who needs `R` as well --
`A = Q R` reconstruction, a subsequent triangular solve against the
explicit `Q` -- extracts the upper triangle before calling, exactly as
with dense `?orgqr`.

All matrices in a call share `m`, `n`, `k`, `ldap`, `layout` and `format`;
the batch is processed one group of `V` interleaved matrices at a time,
`V` derived from `format`.

## 4. Input Parameters

* **`layout`**: `MKL_COL_MAJOR` (tuned) or `MKL_ROW_MAJOR`.
* **`m`**: rows of each matrix `Q` (`>= 0`).
* **`n`**: columns of `Q` to generate (`m >= n >= 0`).
* **`k`**: number of elementary reflectors (`n >= k >= 0`).
* **`ap`**: compact buffer of the `nm` matrices, packed `m x n`
  (`mkl_?gepack_compact`). On entry, columns `0..k-1` hold the reflectors
  below the diagonal as `?geqrf_compact` returned them; everything else --
  the upper triangle, and columns `k..n-1` entirely -- need not be set.
  Any alignment works; a 64-byte base keeps the SIMD sweeps off
  cache-line splits.
* **`ldap`**: leading dimension of each matrix, `>= m` (column-major) or
  `>= n` (row-major). The per-matrix group stride is `ldap * n * V`
  (column-major), so the buffer must be packed with exactly `n` columns:
  a `?geqrf_compact` output packed `m x n` is consumed directly when
  `n = k = min(m, n)` (the thin-`Q` row of the table above).
* **`taup`**: compact buffer of the reflector scalars, `k` per matrix
  (`tau_v(kk) = taup[g*k*V + kk*V + v]`), as `?geqrf_compact` wrote them.
* **`work`, `lwork`**: `lwork = -1` is a workspace query: `work[0]`
  receives the required `lwork` and nothing else is touched. The kernel is
  scratch-free (the reflector-application accumulators live in registers),
  so the query answers `1`; the arguments exist because LAPACK `?orgqr`'s
  do. As everywhere in the compact ecosystem, `lwork` is not otherwise
  checked and `work` must be a valid pointer.
* **`format`**: the pack format from `mkl_get_format_compact()`; selects
  `V` (SSE/AVX/AVX-512 -> 2/4/8 for FP64, 4/8/16 for FP32).
* **`nm`**: number of matrices in the batch.

## 5. Output Parameters

* **`ap`**: the first `n` columns of `Q`, one `m x n` matrix with
  orthonormal columns per compact lane.
* **`work`**: on a query, `work[0]` is the required `lwork` (`1`);
  otherwise untouched.
* **`info`**: a scalar status, `0` on success (MKL leaves the compact
  `info` reserved). No argument checking (section 6.4); an unrecognized
  `format` selects no kernel and sets `info = -1`.

## 6. Design

### 6.1 The algorithm: `?org2r`'s backward accumulation, vectorized

LAPACK's unblocked `?org2r` builds `Q` backward, and the backward order is
what lets it exploit structure. Columns `k..n-1` are initialized to unit
vectors `e_j`; every reflector `H(kk)` has `kk < k <= j`, so all of them
are still to come and act on these columns through step 1 below. Then for
`kk = k-1` down to `0`:

1. **Apply `H(kk)` to the trailing columns** `kk+1 .. n-1` -- the columns
   of `H(kk+1)...H(k-1) [e_{kk+1} .. e_{n-1}]` accumulated so far. This is
   the same rank-1 two-pass update as `?ormqr`'s, over rows `kk..m-1`
   only: rows above `kk` of those columns are zero at this point and stay
   untouched.
2. **Form column `kk` explicitly**: it is `H(kk) e_kk`, which needs no
   sweep -- `A(kk,kk) := 1 - tau(kk)`, `A(i,kk) := -tau(kk) * v_i` for
   `i > kk` (a scale of the reflector body already in place), zeros
   above.

Versus applying all `k` reflectors to all `n` columns of a packed
identity (`?ormqr('L','N')`), the trailing-columns-only sweep and the
sweep-free column formation cut the arithmetic to LAPACK's `?orgqr`
count -- the per-reflector updates `sum_{j=0}^{k-1} 4(m - j)(n - j)` in
closed form, `4mnk - 2(m + n)k^2 + (4/3)k^3` flops per matrix (at `k = n`
it reduces to `?geqrf`'s `2mn^2 - (2/3)n^3`, as it must: accumulating `Q`
costs what factoring did): about a third
less for a square `Q` (`m = n = k`), about half for a tall thin one
(`m >> n = k`) -- on top of not packing, storing, or streaming the
identity batch at all.

The compact vectorization is the toolkit's usual lift: scalar -> `V`-wide
GNU vector, one lane per matrix, no data-dependent branch anywhere in the
loop (step 2 is branch-free by construction; `tau = 0` simply reproduces
`e_kk`).

### 6.2 Reuse: `larf`, nothing else

Step 1 is exactly `larf` (`cqr_ormqr_compact.hpp`), the register-blocked
one-reflector update that `?ormqr_compact` and `?geqrf_compact` already
share -- applied to `A`'s own trailing columns through the same
`BatchView`, precisely as `geqrf_compact_group` applies it during
factorization. `larf`'s implicit-unit convention (it never reads the
pivot row of the reflector column) means the diagonal does not need the
temporary `A(kk,kk) = 1` store dense `?org2r` performs.

Step 2 is new code, but small: one pack broadcast of `tau(kk)`, one scale
loop, one store, one zero loop. Following the one-kernel-per-routine rule
the routine gets its own group kernel, `orgqr_compact_group`, in its own
header `src/cqr_orgqr_compact.hpp`; nothing is added to any existing
kernel's signature (the `?gels` experience: an extra parameter, even
compiled out, cost the plain `?geqrf` path 8-13%).

`ormqr_compact_group` is *not* the building block, despite the shared
sweep direction: its per-reflector slice range is fixed (`0..npanel`)
where `?orgqr`'s shrinks with `kk`, and it has no column-formation step.

### 6.3 Threading

One `for_each_group` over the batch, the body generating one group's `Q`.
The gate is the toolkit's (README, "Threading"): static-schedule OpenMP,
at most one thread per group, active only for two or more groups and
enough work, with the per-group estimate
`(4 m n k - 2 (m + n) k^2 + (4/3) k^3) * V`.

### 6.4 Padding and no argument checking

When `nm` is not a multiple of `V`, the padded identity lanes of the last
group carry zero reflector bodies and (through `?geqrf_compact`) `tau = 0`.
Every step is then the identity map on those lanes: the `larf` sweeps are
no-ops (`tau = 0`), and column formation writes `A(kk,kk) = 1 - 0 = 1`
over a `1`, scales zeros by zero, and zeros the already-zero upper part --
`Q = I` comes out, and the kernel runs unmasked at full width. Note this
holds for a *padded* lane, whose `tau` slot the factorization wrote; a
lane whose `taup` is uninitialized garbage produces garbage, as in every
compact routine.

Nothing is validated, as for every compact routine here and MKL's own;
the portable `?orgqr_compact` of `cqr_compact.h` is the checked surface.

### 6.5 Numerical scope

The columns of the computed `Q` are orthonormal to working precision
(`||Q^T Q - I|| = O(k eps)`), as for dense `?orgqr` -- the accumulation is
a product of exactly representable Householder applications and does not
amplify the factorization's error. `?larfg`-style overflow/underflow
rescaling is omitted, as throughout the toolkit; complex precisions
(`?ungqr`) are out of scope with the rest of the complex domain. The LQ
counterpart `?orglq` -- rows of `Q` from a `?gelqf`-storage factorization,
e.g. the `m < n` output of `cqr_mkl_?gels_compact` -- is this kernel over
the transposed view, and stays out of scope until a use case asks for it.

## 7. Testing

Matching LAPACK `?orgqr` to working precision is the bar. All suites are
CTest-registered and run in FP64 and FP32.

* **Suite 1, vs dense `LAPACKE_?orgqr`** (the dispatch already exists in
  `test_mkl_util.hpp` and is used by the `?geqrf` suite): square and tall
  shapes in both layouts, factored by `mkl_?geqrf_compact` where MKL
  provides the shape and `cqr_mkl_?geqrf_compact` otherwise, then `Q`
  formed both ways -- `cqr_mkl_?orgqr_compact` on the compact batch,
  `LAPACKE_?orgqr` per unpacked matrix. Gated: `Q` elementwise at a
  cross-check tolerance (LAPACKE's blocked accumulation orders the
  rounding differently, so the gate is a modest multiple of `eps`, not
  exactness), plus the properties formed independently: orthogonality
  `||Q^T Q - I||_1 <= c * m * eps` and reconstruction
  `||Q R - A||_1 <= c * m * eps * ||A||_1` with `R = triu` of the
  factorization saved before the call. The `k < n` case (extra unit-seeded
  columns) and the workspace query are covered here.
* **Suite 2, vs `cqr_mkl_?ormqr_compact` on a packed identity**: the two
  routes to `Q` must agree elementwise to a small multiple of `eps` --
  same reflectors, same application order, different structure
  exploitation. This pins the accumulation against an already-validated
  routine without any dense reference in the loop.
* **Portable self-test** (no BLAS): the templated kernel across
  `(T, V, layout, shape)` including `k < n`, `k = 0` (`Q` = leading
  columns of `I`), `min(m, n) = 0`, and padded final packs (padded lanes
  come out identity, real lanes unperturbed), against a scalar
  `ref_org2r` of the same steps (elementwise at `~eps`); orthogonality
  and `Q R = A` formed without the reference; and the portable C API's
  argument validation, `-j` per the header.

## 8. Implementation

* **MKL-style API** (`cqr_mkl_ext.h`): `cqr_mkl_?orgqr_compact` as in
  section 2, unwrapping `MKL_COMPACT_PACK -> V`, instantiated on
  `MKL_INT`; no checking; `lwork = -1` answers the query with `1`.
* **Portable C API** (`cqr_compact.h`):

  ```c
  int dorgqr_compact(char layout, int m, int n, int k, double *ap, int ldap,
                     const double *taup, int V, int nm);
  ```

  No MKL dependency and -- like the rest of that surface -- no
  `work`/`lwork` protocol, since there is nothing to query. LAPACK-style
  validation returning `-j`: `-1` layout, `-2` `m < 0`, `-3` `n` (`< 0` or
  `> m`), `-4` `k` (`< 0` or `> n`), `-6` `ldap` (`< max(1, m)`
  column-major, `< max(1, n)` row-major), `-8` `V` (not 2/4/8/16), `-9`
  `nm < 0`.
* **Kernel** (`src/cqr_orgqr_compact.hpp`): `orgqr_compact_group<T,V>`
  over the layout-agnostic `BatchView` -- unit-seed columns `k..n-1`, then
  the backward sweep of section 6.1 built on `larf` -- driven by
  `orgqr_compact<T,V>` (one `for_each_group`), which both adapters call.

The routine is `?orgqr` in `?org2r`'s unblocked form -- the right form at
these sizes, and the toolkit's throughout -- minus the rescaling LAPACK
also omits there, over the one reflector primitive the QR family already
shares.
