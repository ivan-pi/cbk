# Changelog

All notable changes to cbk are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the version
numbers [Semantic Versioning](https://semver.org/): until 1.0.0, a minor
release may change the API.

## [Unreleased]

### Changed

- **Breaking:** `?ormqr_compact` and `cbk_?ormqr_compact` take `na`, the
  number of columns of the batch in `ap` as `?geqrf_compact` factored it,
  after `k`: `(layout, side, trans, m, n, k, na, ap, ldap, ...)`. The group
  strides of `ap` and `taup` were derived from `k`, so the output of a wide
  factorization (`m < n`), or of any factorization of which fewer than
  `min(m, n)` reflectors were applied, was read from the wrong place in every
  group after the first, with a zero return code. A caller that packed exactly
  the `k` reflector columns passes `na = k` and is otherwise unchanged; the
  portable routine's argument validation renumbers from `na` on (`-7 na`,
  `-9 ldap`, `-12 ldcp`, `-13 V`, `-14 nm`).
- The shared library's soname is `libcbk.so.<major>.<minor>` until 1.0.0
  (`libcbk.so.0.2`), since a minor release may change the ABI; it was
  `libcbk.so.0` for 0.1.0 and 0.2.0 alike, across the `?ormqr_compact`
  signature change.
- The library is built with hidden symbol visibility and exports only the
  entry points of its public headers (marked `CBK_API`); a shared `libcbk`
  used to export every kernel template instantiation as well.

### Fixed

- `?geqrf_compact` (and `?gels_compact`, which factors with it) rescales a
  reflector whose column's sum of squares underflows or overflows, as LAPACK's
  `?larfg` does. A column of small entries (below about `1e-162` in double,
  `1e-23` in single precision) used to read as already triangular and return a
  finite, wrong `R`; one of large entries returned `Inf`. Inputs of ordinary
  magnitude take the same path as before and factor to the same bits.
- A batch within `V` of `INT_MAX` matrices no longer overflows the group
  count, which made every routine return 0 without computing anything.

## [0.2.0] - 2026-09-15

### Changed

- **Breaking:** the portable `?ormqr_compact` takes `layout`, `side` and
  `trans` -- `(layout, side, trans, m, n, k, ap, ldap, taup, cp, ldcp, V, nm)`,
  LAPACK `?ormqr`'s argument order behind the layout, as the MKL-style
  `cbk_?ormqr_compact` already did -- so Q can be applied from either side in
  either layout; it was side `'L'`, column-major only. A caller of the old
  signature `(trans, m, nrhs, k, ...)` now writes `('C', 'L', trans, m, nrhs,
  k, ...)`. Its argument validation renumbers accordingly (`-1 layout`,
  `-2 side`, `-3 trans`, ..., `-13 nm`).
- Tests: the portable suites gate every numerical check on a dimensionless
  test ratio in the exact form of the corresponding LAPACK checker (dqrt01,
  dqrt03, dpot01, dsyt01, dget02, dtrt02, dget04, ...) against one threshold,
  `THRESH = 30`, instead of per-check tolerances; sweep the small-dimension
  cross product (`m, n` from 0 to 5, the empty operand included) in every
  routine; and check related routines against each other (`potrs` and
  `sytrsnp` against the `trsm` sweeps they are built from, `ormqr` in every
  layout, side and trans against the explicit Q of `orgqr`).
- Tests: every suite validates against a real LAPACKE + CBLAS stack
  (OpenBLAS, Netlib, or MKL's own with the MKL extension;
  `cmake/FindLAPACKE.cmake` over `find_package(LAPACK)` and `BLA_VENDOR`)
  instead of hand-rolled
  scalar references (issue #27). The dense-LAPACK cross-checks of the MKL
  suites run in the portable suites, on any stack; the unpivoted LDL^T keeps
  its scalar reference, itself validated against CBLAS. Building the tests
  now needs such a stack; the library does not. A macOS workflow tests
  against Accelerate through accelerate-lapacke.

## [0.1.0] - 2026-09-13

The first release: batched QR, Cholesky and unpivoted LDL^T factorizations,
and the solves built on them, for many small matrices in the compact
(interleaved) format, in single and double precision.

### Added

- Portable C API (`cbk.h`), no external dependencies, LAPACK-style argument
  validation: `?geqrf`, `?ormqr`, `?orgqr`, `?potrf`, `?potrs`, `?posv`,
  `?sytrfnp`, `?sytrsnp`, `?sysvnp`, `?trsm` and `?gels`, each `_compact`, with
  an explicit interleave width `V` of 2, 4, 8 or 16 and column- or row-major
  layout.
- `cbk_get_version()` and the `CBK_VERSION_*` macros of `cbk.h`.
- Optional Intel MKL-style API (`cbk_compat.h`, `-DCBK_WITH_MKL=ON`): the same
  routines as `cbk_?*_compact` entry points taking MKL's `MKL_LAYOUT` and
  `MKL_COMPACT_PACK`, interoperable with MKL's own compact routines and
  pack/unpack helpers; `cbk_mkl_alloc.h`, C++-only RAII `mkl_malloc` helpers.
- SIMD kernels written with GNU vector types, one source the compiler lowers to
  whatever vector ISA it targets (SSE, AVX, AVX-512, or non-x86 extensions);
  OpenMP threading of each routine's loop over groups (`-DCBK_WITH_OPENMP`),
  composable with a caller's outer parallel loop.
- CMake package: `find_package(cbk CONFIG)` and the `cbk::cbk` target, static or
  shared (`BUILD_SHARED_LIBS`), verified by an install CI job.
- Tests: BLAS-free suites against scalar references for every routine, and
  MKL-backed suites cross-checked against MKL's compact kernels and dense
  LAPACK/LAPACKE; a worked example and five benchmarks with the MKL build.
- One design document per routine under `docs/`.

### Known limitations

- Real precisions only; no column pivoting (`?geqrf`) and no pivoting in the
  LDL^T (`?sytrfnp` requires a well-behaved factorization).
- No overflow/underflow-safe scaling (LAPACK's `dlarfg` rescaling): inputs are
  assumed well conditioned and within range. Deferred to a later release.
- Tested on x86-64 Linux with GCC and Clang, against Intel MKL 2020.4 (lp64 and ilp64).

[0.2.0]: https://github.com/ivan-pi/cbk/releases/tag/v0.2.0
[0.1.0]: https://github.com/ivan-pi/cbk/releases/tag/v0.1.0
