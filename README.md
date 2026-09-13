# cqr - Compact QR extensions

Batched QR, Cholesky and LDL^T for many small matrices, stored in Intel MKL's
**Compact** (interleaved) format. **cqr** provides portable, SIMD-vectorized
kernels behind an Intel MKL-style API, filling the gaps in MKL's compact
LAPACK (`ormqr`, `orgqr`, `potrs`/`posv`, `sytrf`, `gels`) and offering open
alternatives to the routines it does ship (`geqrf`, `potrf`, `trsm`). Together
they factor and solve batched systems entirely in the compact format, with no
MKL compute kernel.

| Routine (`s`/`d`) | What it does | Design |
|-------------------|--------------|--------|
| `cqr_mkl_?geqrf_compact` | QR factorization | [doc](docs/cqr_mkl_dgeqrf_compact_design.md) |
| `cqr_mkl_?ormqr_compact` | apply `Q` or `Q^T` to a batch | [doc](docs/cqr_mkl_dormqr_compact_design.md) |
| `cqr_mkl_?orgqr_compact` | form the explicit (thin) `Q` | [doc](docs/cqr_mkl_dorgqr_compact_design.md) |
| `cqr_mkl_?potrf` / `?potrs` / `?posv_compact` | Cholesky factorization, solve, fused factor-and-solve | [doc](docs/cqr_mkl_dpotrf_compact_design.md) |
| `cqr_mkl_?sytrfnp` / `?sytrsnp` / `?sysvnp_compact` | unpivoted LDL^T factorization, solve, fused factor-and-solve | [doc](docs/cqr_mkl_dsytrfnp_compact_design.md) |
| `cqr_mkl_?trsm_compact` | triangular solve | [doc](docs/cqr_mkl_dtrsm_compact_design.md) |
| `cqr_mkl_?gels_compact` | least-squares / minimum-norm solve `op(A) X = B` in one call | [doc](docs/cqr_mkl_dgels_compact_design.md) |

Every routine is also available through a portable C API (`include/cqr_compact.h`)
with an explicit interleave width and no MKL dependency. The kernels are written
with GNU vector types, which the compiler lowers to SSE, AVX, or AVX-512 -- one
portable source for every width.

## Getting started

Requires CMake >= 3.18, a C++17 compiler (GCC/Clang), and Intel MKL (oneAPI, or
`sudo apt-get install libmkl-dev` on Debian/Ubuntu).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`-march=native` lets the kernels use the host's widest vectors; drop it for a
generic build. `-DCQR_WITH_MKL=OFF` builds the portable kernels alone, without
MKL. `examples/solve_qr_compact.cpp` is a worked batched solve.

## Documentation

[`docs/README.md`](docs/README.md) indexes the rest: [building](docs/building.md)
(locating MKL, the threaded layer, the workspace contract, CMake options),
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
