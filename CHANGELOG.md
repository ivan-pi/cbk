# Changelog

All notable changes to cbk are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the version
numbers [Semantic Versioning](https://semver.org/): until 1.0.0, a minor
release may change the API.

## [Unreleased]

### Changed

- `?trsm_compact` is one register-tiled kernel for every side, uplo, trans
  and layout (the other cases are transposed or reversed views of the same
  forward substitution), with the tile sized to the target's SIMD register
  count; row-major, `side = 'R'` and few-right-hand-side solves no longer
  take a slower path. A non-unit diagonal is applied as a multiply by its
  reciprocal instead of a divide, so results can differ from the previous
  release in the last bit. `?potrs_compact`, `?posv_compact`,
  `?sytrsnp_compact`, `?sysvnp_compact` and `?gels_compact` inherit it.
- `?geqrf_compact` is blocked: four-column `geqr2` panels, the trailing
  columns updated by the panel's reflectors as one register-tiled block
  reflector (compact WY), unblocked below three panels. `?ormqr_compact`
  applies its reflectors four at a time the same way (one at a time for a
  single right-hand side). `?gels_compact` inherits both. Throughput only; the
  factors differ from the previous release's in rounding.
- `?sytrfnp_compact` (and `?sysvnp_compact`) factor left-looking in register
  tiles, each tile loaded and stored once; `?sytrsnp_compact` folds its
  diagonal solve into the second triangular sweep. Throughput only; the
  factor differs from the previous release's in rounding.
- A call that is not worth threading -- a batch below the work gate, or any
  call from inside the caller's own parallel region -- runs its loop over
  groups without entering the OpenMP runtime. The serialized parallel region
  it used to open cost as much as a small group's work on every call, most
  under clang's and Intel's OpenMP runtimes. Every routine benefits; results
  are unchanged.

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
