# Source layout

```
include/   public headers
src/       kernels and the two adapter sources
tests/     the portable C API's suites (vs LAPACKE/CBLAS, any stack) and the
           MKL-backed ones; tests/install/ checks an installed cbk from a
           downstream project
examples/  worked solve + benchmarks (see examples/BENCHMARKS.md)
docs/      design documents and guides (see docs/README.md)
cmake/     FindMKLCompact.cmake, FindLAPACKE.cmake (the tests' stack),
           cbkConfig.cmake.in (the installed package)
```

## Public interface

The headers under `include/` are the project's API, the only files users need:

| File | Role |
|------|------|
| `include/cbk.h` | The library's C API: `?geqrf_compact`, `?ormqr_compact`, `?orgqr_compact`, `?potrf_compact`, `?potrs_compact`, `?posv_compact`, `?sytrfnp_compact`, `?sytrsnp_compact`, `?sysvnp_compact`, `?trsm_compact`, `?gels_compact` (`d`/`s`), with an explicit interleave width `V`, LAPACK-style `info = -j` validation, and no MKL dependency. The library target is `cbk::cbk`. |
| `include/cbk_compat.h` | The optional MKL-style API (`-DCBK_WITH_MKL=ON`, compiled into the same `cbk::cbk`): `cbk_?geqrf_compact`, `cbk_?ormqr_compact`, `cbk_?orgqr_compact`, `cbk_?potrf_compact`, `cbk_?potrs_compact`, `cbk_?posv_compact`, `cbk_?sytrfnp_compact`, `cbk_?sytrsnp_compact`, `cbk_?sysvnp_compact`, `cbk_?trsm_compact`, `cbk_?gels_compact`, taking `MKL_LAYOUT` and `MKL_COMPACT_PACK` formats. Also the C++ helpers `vlen_for_format` / `format_for_vlen` / `compact_format_name`. |
| `include/cbk_mkl_alloc.h` | Optional RAII buffer helpers (`mkl_alloc_bytes`, `mkl_buffer`) wrapping `mkl_malloc`/`mkl_free`. |

The two APIs share the kernels and differ in argument checking: the portable
C API validates LAPACK-style, returning `-j` for a bad j-th argument; the
MKL-style one skips validation like MKL's own compact routines (`info` is a
scalar, `0` on success).

## Internals

| File | Role |
|------|------|
| `src/cbk_common.hpp` | The `pack<T,V>` SIMD element, the `BatchView` strided group view every kernel addresses its operands through, `vsqrt`/`broadcast`, the `for_each_group` threading driver, and the runtime-`V` dispatch helper `for_vlen`. |
| `src/cbk_matrix_view.hpp` | `MatrixView<T>`, the dense strided 2-D view the tests, benchmarks and examples address their host-side matrices through -- the non-compact counterpart of `BatchView`. Internal, not part of the public API. |
| `src/cbk_matrix_batch.hpp` | `MatrixBatch<T>`, the owning batch of dense matrices (storage, per-matrix `view(v)`, and the `base_ptrs()` array MKL's pack/unpack routines take) the tests, benchmarks and examples share. |
| `src/cbk_geqrf_compact.hpp` | QR factorization kernel: vectorized `geqr2` with a branch-free `larfg`, reusing ormqr's `larf` for the trailing update. |
| `src/cbk_ormqr_compact.hpp` | Apply-Q kernel (vectorized `dorm2r`) and the shared one-reflector update `larf`. |
| `src/cbk_orgqr_compact.hpp` | Form-Q kernel (vectorized `dorg2r`): backward accumulation on `larf` over the trailing columns only, unit-seeded extra columns, sweep-free column formation. |
| `src/cbk_potrf_compact.hpp` | Cholesky kernel (vectorized `potf2`); the four `(layout, uplo)` cases are one kernel over transposed views. |
| `src/cbk_potrs_compact.hpp` | The Cholesky solve: two non-unit `trsm` group sweeps. |
| `src/cbk_posv_compact.hpp` | The fused Cholesky factor-and-solve driver: both group kernels per group, while the factor is cache-resident. |
| `src/cbk_sytrfnp_compact.hpp` | Unpivoted LDL^T kernel (the square-root-free `potf2`), over the same transposed views as `potrf`. |
| `src/cbk_sytrsnp_compact.hpp` | The LDL^T solve: two unit-diagonal `trsm` group sweeps around a diagonal solve. |
| `src/cbk_sysvnp_compact.hpp` | The fused factor-and-solve driver: both group kernels per group, while the factor is cache-resident. |
| `src/cbk_trsm_compact.hpp` | Triangular-solve kernels: the tuned column-major `side='L'` row-dot path and the general strided kernel, behind the per-group `trsm_compact_group` the fused solves compose. |
| `src/cbk_gels_compact.hpp` | The one-call least-squares / minimum-norm solve: one driver over the geqrf, ormqr and trsm group kernels, on the tall view of `A` (transposed when `m < n`, which is the LQ case). |
| `src/cbk.cpp` | The portable C API: argument validation and `V` dispatch for all twenty-two entry points. |
| `src/cbk_compat.cpp` | The MKL-style API: MKL enum / `MKL_COMPACT_PACK` unwrapping for all twenty-two entry points. |
| `tests/test_compact_util.hpp` | Shared test helpers: RNG, error metrics, input generation, Compact pack/unpack, and `compact<T>` (the portable C API dispatched on the scalar type); header-only, library-free. |
| `tests/test_lapack_util.hpp` | The dense references on a real LAPACKE + CBLAS: `lapack<T>` (the LAPACKE/CBLAS dispatch), the `ref_*` procedures over views (`ref_geqr2`, `ref_orm2r`, `ref_potf2`, `ref_trsm`, `ref_gels`, ...), `matmul`, `tri_apply`, `solve_errors`. The stack comes from `cmake/FindLAPACKE.cmake`. |
| `tests/test_mkl_util.hpp` | Scalar-type dispatch for the MKL-backed suites: `compat<T>` (routines under test), `mkl<T>` (MKL's compact API and kernels). |
| `tests/test_*_compact.cpp` | The portable C API's suites, on any LAPACKE stack: each kernel vs its LAPACK counterpart (elementwise where that is the same unblocked algorithm, residual and invariant gates elsewhere), plus C API validation, in FP64 and FP32. |
| `tests/test_*_mkl.cpp` | MKL validation, templated on the scalar type and run in FP64 and FP32: the MKL-style API through MKL's pack/unpack, cross-checks vs MKL's compact kernels, end-to-end solves. |
| `examples/bench_util.hpp` | The benchmarks' shared harness (timing, aligned storage, command line). |

The contributor conventions behind this layout -- one kernel per routine over
`BatchView`, the two views, the workspace and alignment contracts -- are in
[`.claude/CLAUDE.md`](../.claude/CLAUDE.md).
