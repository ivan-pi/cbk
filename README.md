# cqr - batched compact-format factorizations

Batched QR, Cholesky and LDL^T factorizations, and the solves built on them, for
many small matrices stored in the **compact** (interleaved) format: element
`(i,j)` of `V` consecutive matrices sits side by side, so one SIMD instruction
advances all `V` factorizations in lockstep. **cqr** is a standalone library
with a portable C API and no external dependencies; its kernels are written with
GNU vector types, which the compiler lowers to SSE, AVX, or AVX-512 -- one
source for every width. An optional Intel MKL-style API drops the same kernels
into MKL's Compact ecosystem on Intel CPUs.

> **Generated with Claude.** Most of this library -- the kernels, the tests,
> the benchmarks and the design documents -- was written by Anthropic's Claude
> models, working from and against per-routine design documents, with human
> direction and review. Each design document and benchmark page records the
> models that assisted it, and the commit history records the rest.

| Routine (`s`/`d`) | What it does | Design |
|-------------------|--------------|--------|
| `?geqrf_compact` | QR factorization | [doc](docs/cqr_mkl_dgeqrf_compact_design.md) |
| `?ormqr_compact` | apply `Q` or `Q^T` to a batch | [doc](docs/cqr_mkl_dormqr_compact_design.md) |
| `?orgqr_compact` | form the explicit (thin) `Q` | [doc](docs/cqr_mkl_dorgqr_compact_design.md) |
| `?potrf` / `?potrs` / `?posv_compact` | Cholesky factorization, solve, fused factor-and-solve | [doc](docs/cqr_mkl_dpotrf_compact_design.md) |
| `?sytrfnp` / `?sytrsnp` / `?sysvnp_compact` | unpivoted LDL^T factorization, solve, fused factor-and-solve | [doc](docs/cqr_mkl_dsytrfnp_compact_design.md) |
| `?trsm_compact` | triangular solve | [doc](docs/cqr_mkl_dtrsm_compact_design.md) |
| `?gels_compact` | least-squares / minimum-norm solve `op(A) X = B` in one call | [doc](docs/cqr_mkl_dgels_compact_design.md) |

The C API (`include/cqr_compact.h`) takes the interleave width `V` explicitly
and validates its arguments LAPACK-style. Every routine threads its loop over
groups of `V` matrices with OpenMP, and composes with a caller's own parallel
loop.

## Getting started

Requires CMake >= 3.18 and a C++17 compiler (GCC/Clang); nothing else.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`-march=native` lets the kernels use the host's widest vectors; drop it for a
generic build. Link the `cqr::compact` target and include `cqr_compact.h`.

## Intel MKL extension

The project began as a set of extensions to Intel MKL's Compact API, which ships
`mkl_?geqrf_compact`, `mkl_?potrf_compact` and `mkl_?trsm_compact` but no way
to apply or form `Q`, no Cholesky or LDL^T solve, and no `gels`. That API
survives as an optional layer, `-DCQR_WITH_MKL=ON`: `cqr_mkl_?*_compact` entry
points (`include/cqr_mkl_ext.h`) that take MKL's `MKL_LAYOUT` and
`MKL_COMPACT_PACK` arguments, so they mix freely with MKL's own compact routines,
its pack/unpack helpers, and `mkl_?gemm_compact`. The same option builds the
MKL-backed test suites, the worked example `examples/solve_qr_compact.cpp`, and
the benchmarks against MKL's compact kernels and per-matrix LAPACK. It needs
Intel MKL (oneAPI, or `sudo apt-get install libmkl-dev` on Debian/Ubuntu).

## Documentation

[`docs/README.md`](docs/README.md) indexes the rest: [building](docs/building.md)
(the MKL build, locating MKL, the workspace contract, CMake options),
[threading](docs/threading.md), the [source layout](docs/layout.md), the
[examples and benchmarks](docs/examples.md) (measured results in
[`examples/BENCHMARKS.md`](examples/BENCHMARKS.md)), [related work](docs/related_work.md),
and one design document per routine.

## Contributing

The C++ is formatted with clang-format and linted with clang-tidy, both driven
by [pre-commit](https://pre-commit.com/) (`pip install pre-commit && pre-commit
install`); CI checks the same hooks. [CLAUDE.md](.claude/CLAUDE.md) has the
details and the conventions the kernels follow.

## License

Apache License 2.0; see [LICENSE](LICENSE).
