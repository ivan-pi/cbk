# cbk - Compact Batch Kernels

Batched QR, Cholesky and LDL^T factorizations, and the solves built on them, for
many small matrices stored in the **compact** (interleaved) format: element
`(i,j)` of `V` consecutive matrices sits side by side, so one SIMD instruction
advances all `V` factorizations in lockstep. **cbk** (Compact Batch Kernels,
libcbk) is a standalone library
with a portable C API and no external dependencies; its kernels are written with
GNU vector types, which the compiler lowers to SSE, AVX, or AVX-512 -- one
source for every width. An optional Intel MKL-style API drops the same kernels
into MKL's Compact ecosystem on Intel CPUs.

> **Generated with Claude.** Most of this library -- the kernels, the tests,
> the benchmarks and the design documents -- was written by Anthropic's Claude
> models, working from and against per-routine design documents, with human
> direction and review. Each design document and benchmark page records the
> models that assisted it, and the commit history records the rest.

| Routine (`s`/`d`) | What it does |
|-------------------|--------------|
| [`?geqrf_compact`](docs/cbk_dgeqrf_compact_design.md) | QR factorization |
| [`?ormqr_compact`](docs/cbk_dormqr_compact_design.md) | apply `Q` or `Q^T` to a batch |
| [`?orgqr_compact`](docs/cbk_dorgqr_compact_design.md) | form the explicit (thin) `Q` |
| [`?potrf` / `?potrs` / `?posv_compact`](docs/cbk_dpotrf_compact_design.md) | Cholesky factorization, solve, fused factor-and-solve |
| [`?sytrfnp` / `?sytrsnp` / `?sysvnp_compact`](docs/cbk_dsytrfnp_compact_design.md) | unpivoted LDL^T factorization, solve, fused factor-and-solve |
| [`?trsm_compact`](docs/cbk_dtrsm_compact_design.md) | triangular solve |
| [`?gels_compact`](docs/cbk_dgels_compact_design.md) | least-squares / minimum-norm solve `op(A) X = B` in one call |

Each routine links to its design document.

The C API (`include/cbk.h`) takes the interleave width `V` explicitly
and validates its arguments LAPACK-style. Every routine threads its loop over
groups of `V` matrices with OpenMP (`-DCBK_WITH_OPENMP=OFF` disables it).

## Getting started

Requires CMake >= 3.28 and a C++17 compiler (GCC/Clang); nothing else.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`-march=native` lets the kernels use the host's widest vectors; drop it for a
generic build. Link the `cbk::cbk` target and include `cbk.h`.
`cmake --install build --prefix <dir>` installs the library, the headers and a
CMake package (`find_package(cbk CONFIG)`); see [Building](docs/building.md).

## Intel MKL extension

The project began as a set of extensions to Intel MKL's Compact API, which ships
`mkl_?geqrf_compact`, `mkl_?potrf_compact` and `mkl_?trsm_compact` but no way
to apply or form `Q`, no Cholesky or LDL^T solve, and no `gels`. That API
survives as an optional layer, `-DCBK_WITH_MKL=ON`: `cbk_?*_compact` entry
points (`include/cbk_compat.h`) that take MKL's `MKL_LAYOUT` and
`MKL_COMPACT_PACK` arguments, so they mix freely with MKL's own compact routines,
its pack/unpack helpers, and `mkl_?gemm_compact`.

The same option builds the MKL-backed test suites, the worked example `examples/solve_qr_compact.cpp`, and
the benchmarks against MKL's compact kernels and per-matrix LAPACK. It needs
Intel MKL (oneAPI, or `sudo apt-get install libmkl-dev` on Debian/Ubuntu).

## Documentation

[`docs/README.md`](docs/README.md) indexes the rest:

* [Building](docs/building.md) - the MKL build, locating MKL, the workspace
  contract, CMake options.
* [Threading](docs/threading.md) - the loop over groups and how it composes
  with a caller's own parallel loop.
* [Source layout](docs/layout.md) - the public headers, the kernels, the test
  and benchmark helpers.
* [Examples and benchmarks](docs/examples.md) - the worked solve and the five
  benchmarks; measured results in [`examples/BENCHMARKS.md`](examples/BENCHMARKS.md).
* One design document per routine, linked from the table above.
* [CHANGELOG.md](CHANGELOG.md) - what each release changed.

## Related work

Batched / compact dense linear algebra for many small matrices:

* [Batched BLAS (BBLAS)](https://icl.utk.edu/bblas/) - the proposed standard interface for batched BLAS.
* [Intel oneMKL Compact BLAS and LAPACK functions](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/compact-blas-and-lapack-functions.html) - the compact (interleaved) format this project builds on.
* [Arm Performance Libraries interleave-batch functions](https://developer.arm.com/documentation/101004/2507/Interleave-batch-functions/Interleave-batch-introduction?lang=en) - Arm's equivalent interleaved-batch API.
* [Kokkos Kernels batched API](https://kokkos.org/kokkos-kernels/docs/API/batched-index.html) - portable batched kernels.
* [MAGMA batched routines](https://icl.utk.edu/projectsfiles/magma/doxygen/group__batched.html) - batched BLAS and LAPACK for GPUs, over arrays of conventionally stored (non-interleaved) matrices.
* [batmat](https://github.com/tttapa/batmat) - batched small-matrix linear algebra.

## Contributing

The C++ is formatted with clang-format and linted with clang-tidy, both driven
by [pre-commit](https://pre-commit.com/) (`pip install pre-commit && pre-commit
install`); CI checks the same hooks. [CLAUDE.md](.claude/CLAUDE.md) has the
details and the conventions the kernels follow.

## License

Apache License 2.0; see [LICENSE](LICENSE).
