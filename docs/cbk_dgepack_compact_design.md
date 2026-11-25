# API Design Document: `?gepack_compact`, `?geunpack_compact`, `?get_size_compact`

> Assisted-by: Claude:claude-fable-5

## 1. Overview

The conversion between a batch of conventionally stored matrices and the
compact (interleaved) format, and the size of the buffer that holds the
latter. Intel MKL ships these as `mkl_?gepack_compact`,
`mkl_?geunpack_compact` and `mkl_?get_size_compact`; the portable C API had no
counterpart, so a caller without MKL had to write the interleaving by hand,
and get the padding of a partial last group right, before calling anything
else in `cbk.h`. The test suites carried exactly such a pair of helpers; these
routines are those helpers, made part of the library.

There is no `cbk_`-prefixed MKL-style variant: with MKL present, MKL's own
routines are the ones to call, and the two produce the same buffer (section 5).

## 2. Syntax

```c
size_t dget_size_compact(int ld, int sd, int V, int nm);

int dgepack_compact  (char layout, int m, int n, const double *const *a, int lda,
                      double *ap, int ldap, int V, int nm);
int dgeunpack_compact(char layout, int m, int n, double *const *a, int lda,
                      const double *ap, int ldap, int V, int nm);
```

and the `s` forms for `float`. The argument lists are MKL's with the
`MKL_LAYOUT` enum replaced by the layout character and `MKL_COMPACT_PACK` by
the interleave width `V`, as everywhere in `cbk.h`.

## 3. Parameters

* **`layout`**: `'C'` column-major or `'R'` row-major, for the dense matrices
  and the compact batch alike.
* **`m`, `n`**: rows and columns of each matrix.
* **`a`**: an array of `nm` pointers, `a[idx]` the `idx`-th matrix, stored in
  `layout` with leading dimension `lda`. The matrices need not be contiguous
  or equally spaced, which is why this is a pointer array and not a base and
  a stride.
* **`lda`**, **`ldap`**: leading dimensions of the dense matrices and of the
  compact batch (`>= max(1, m)` column-major, `>= max(1, n)` row-major).
* **`ap`**: the compact batch, `?get_size_compact(ldap, n, V, nm)` bytes
  column-major and `(ldap, m, V, nm)` row-major.
* **`V`, `nm`**: interleave width (2, 4, 8 or 16) and number of matrices.

`?get_size_compact(ld, sd, V, nm)` returns
`ld * sd * V * ceil(nm / V) * sizeof(scalar)` bytes, `sd` being the extent
that is not the leading dimension's. It returns `0` for an illegal argument:
its return value is a size, so it has no `-j` to give.

The pack and unpack routines validate LAPACK-style and return `-j` for an
illegal `j`-th argument (`-1 layout`, `-2 m`, `-3 n`, `-5 lda`, `-7 ldap`,
`-8 V`, `-9 nm`); the pointers are not inspected.

## 4. Padding

A compact batch is `ceil(nm / V)` whole groups, and every kernel computes on
all `V` lanes of every group. `?gepack_compact` therefore fills the slots past
`nm` in the last group with the identity -- ones on the diagonal, zeros
elsewhere -- which each factorization and solve maps to a finite result (the
identity factors to itself, with `tau = 0`). `?geunpack_compact` does not read
those slots.

The scalars between the matrix extent and the leading dimension (rows `m` to
`ldap - 1` column-major) are padding of a different kind: neither routine
writes them, on either side.

A `tau` batch is the one operand for which the identity is not the right
padding: packed as `k x 1` matrices it gets a one in `tau(0)` of the padded
slots, where the reflectors of an identity have zero. `?geqrf_compact`
produces `tau` in compact form, so packing one is needed only when the
factorization came from elsewhere; clear those scalars then (the suites'
`pack_tau` does).

## 5. Compatibility with MKL

Both padding choices are `mkl_?gepack_compact`'s, measured rather than read
from the documentation, and `?get_size_compact` returns MKL's byte count.
`tests/test_gepack_mkl.cpp` holds the routines to that: the packed buffers of
the two libraries are compared with `memcmp` from the same sentinel fill, each
library unpacks the other's buffer to the same matrices, and the sizes agree,
over both layouts and precisions and MKL's three formats.

## 6. Implementation

`src/cbk_gepack_compact.hpp`: one `for_each_group` driver per direction over
the `BatchView` of the group, so both layouts are one body. Each pack is `V`
scalar loads (or stores) through the matrix pointers; that gather is not
vectorized and does not need to be, since the conversion is bound by memory
traffic. The groups are independent and run in parallel under the usual gate.

## 7. Testing

`tests/test_gepack_compact.cpp` checks the routines against the layout formula
of `cbk.h`, written out by hand in the test: every other suite packs and
unpacks through these routines, so this one must not. All checks are exact:
every scalar of the packed buffer is the element the formula names, the padded
slots hold the identity, the scalars beyond the extent keep their sentinel
(including a guard past the end of the buffer), the unpacked matrices are the
originals with their own beyond-extent entries untouched, and the padded slots
are poisoned with NaN before unpacking to show they are not read. It runs both
layouts and precisions, the four widths, leading dimensions at and beyond the
extent, full and partial last groups, a batch large enough to thread, and the
small-dimension cross product, plus the argument validation and the size.
