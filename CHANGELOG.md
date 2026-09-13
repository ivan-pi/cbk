# Changelog

All notable changes to cbk are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the version
numbers [Semantic Versioning](https://semver.org/): until 1.0.0, a minor
release may change the API.

## [0.1.0] - unreleased

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
- Tested on x86-64 Linux with GCC and Clang.

[0.1.0]: https://github.com/ivan-pi/cbk/releases/tag/v0.1.0
