# Building

## The portable library (default)

Requires **CMake >= 3.28**, a **C++17 compiler** (GCC/Clang) and a build tool
(Make/Ninja). Nothing else: the default configure builds the `cbk::cbk`
library (the C API of `include/cbk.h`) and its BLAS-free test suites,
which check every kernel against a scalar reference of the same algorithm.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## The Intel MKL extension (`-DCBK_WITH_MKL=ON`)

`-DCBK_WITH_MKL=ON` adds the MKL-style API (the `cbk_?*_compact` entry
points of `include/cbk_compat.h`, compiled into the same `cbk::cbk` library),
the MKL-backed
test suites (cross-checked against MKL's own compact kernels and dense
LAPACK/LAPACKE), the worked example and the benchmarks. It needs **Intel MKL**,
which provides the Compact-format API (`mkl_compact.h`, `mkl_?gepack_compact`,
`mkl_?geqrf_compact`, ...). Any MKL works:

* oneAPI MKL - `source /opt/intel/oneapi/setvars.sh` (sets `MKLROOT`), or
* Debian/Ubuntu - `sudo apt-get install libmkl-dev` (headers in
  `/usr/include/mkl`, LP64 libs in the default library path).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCBK_WITH_MKL=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Locating MKL

`cmake/FindMKLCompact.cmake` locates Intel MKL directly (the compact API is an
MKL extension no other BLAS provides): the distro package is found on its own,
and a oneAPI install is selected with `MKLROOT` or `-DMKLCompact_ROOT=<prefix>`
(headers and libraries are taken from that one installation). The sequential
MKL layer is the default; `-DMKLCompact_THREADING=threaded` links MKL's internally
threaded layer instead. `.claude/mkl-install.md` covers installing MKL from the
distro package or from Intel's oneAPI apt repository, and the remaining
`MKLCompact_*` options.

> **Workspace sizes under a threaded MKL.** Compact routines take their
> workspace from the caller so that nothing is allocated on the hot path, and
> none of them checks `lwork`: an undersized buffer is an out-of-bounds write,
> not an error. With `MKLCompact_THREADING=threaded`, MKL's own `?geqrf_compact`
> and `?getrinp_compact` size that workspace *per thread* -- their `lwork = -1`
> query returns `n * V * mkl_get_max_threads()` -- so query and call under the
> same thread count, and never reuse a buffer sized under fewer threads.
> `cbk_?gels_compact` also uses `work` (as the `tau` scratch of its
> factorization, `min(m,n) * V * ceil(nm/V)` scalars, independent of the thread
> count); size it from its own query. The other cbk routines need no scratch.
> `.claude/mkl-compact-behavior.md` records what MKL's compact routines were
> measured to do with their arguments.

## The benchmarks without MKL (`-DCBK_BUILD_BENCHMARKS=ON`)

The benchmarks of the portable C API (`bench_trsm_compact` so far) measure a
`cbk.h` routine against the per-matrix BLAS/LAPACK routine of the same math,
so they need a BLAS/LAPACK, not MKL: `-DCBK_BUILD_BENCHMARKS=ON` builds them
in the default tree against whatever `find_package(LAPACK REQUIRED)` locates.
Select the library with `-DBLA_VENDOR` (`OpenBLAS`, `Intel10_64lp_seq` for
sequential MKL, `Generic` for the reference library, ...;
`-DBLA_SIZEOF_INTEGER=8` selects an ILP64 one) -- a baseline is only as
meaningful as the BLAS behind it, so prefer a tuned one; the reference BLAS
builds and passes but is slow. The MKL build builds every benchmark, these
included, and takes their BLAS/LAPACK from MKL itself, the one MKL link line
`MKL::Compact` carries, so a second `find_package(LAPACK)` cannot pull in a
different threading layer of the same MKL.

```sh
sudo apt-get install libopenblas-dev          # Debian/Ubuntu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCBK_BUILD_BENCHMARKS=ON -DBLA_VENDOR=OpenBLAS
cmake --build build -j
OPENBLAS_NUM_THREADS=1 ./build/bench_trsm_compact   # the benchmark threads its own loop
```

The benchmarks of the MKL-style API pack with MKL's compact routines and
compare against its compact kernels, so they stay with the MKL build
([`examples/BENCHMARKS.md`](../examples/BENCHMARKS.md)).

## Performance builds

The library sets no `-march` of its own; optimization flags are the caller's to
choose. For a performance build, pass host-tuned flags through `CMAKE_CXX_FLAGS`
so the SIMD kernels target the machine's widest vectors (the same AVX-512 MKL
selects at runtime):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"
```

Correctness is independent of these flags; only throughput changes. The
benchmarks in particular need them for a fair comparison against MKL (see
[`examples/BENCHMARKS.md`](../examples/BENCHMARKS.md)).

## Installing

```sh
cmake --install build --prefix /opt/cbk
```

installs the library (`libcbk.a`, or `libcbk.so` with `-DBUILD_SHARED_LIBS=ON`),
the public headers of what was built (`cbk.h` always; `cbk_compat.h`
and `cbk_mkl_alloc.h` only with `CBK_WITH_MKL=ON`), and a CMake package under
`lib/cmake/cbk/`. A downstream project finds it with

```cmake
find_package(cbk CONFIG REQUIRED)            # -DCMAKE_PREFIX_PATH=/opt/cbk
target_link_libraries(app PRIVATE cbk::cbk)  # both APIs, as installed
```

The package reports `cbk_WITH_MKL` and, when set, re-finds MKL's headers (all
the exported target needs) through the `FindMKLCompact.cmake` installed beside
it; a consumer that also calls MKL itself finds `MKL::Compact` with that module
(`list(APPEND CMAKE_MODULE_PATH "${cbk_DIR}")`) or its own. A static cbk built
with OpenMP needs the CXX language enabled in the consumer, as linking a C++
static library does anyway. `tests/install/` is a downstream consumer project
CI builds against a fresh install for every MKL / static / shared combination
(`.github/workflows/install.yml`, through `.github/check_install.sh`).

## Options

| Option | Default | Effect |
|--------|---------|--------|
| `CBK_WITH_MKL` | `OFF` | Build the MKL-style API, the MKL-backed tests, the example and the benchmarks; requires Intel MKL. |
| `CBK_WITH_OPENMP` | `ON` | Thread the loop over groups with OpenMP; `OFF` gives single-threaded routines. See [threading.md](threading.md). |
| `CBK_BUILD_TESTS` | `ON` | Build the tests and register them with CTest. |
| `CBK_BUILD_BENCHMARKS` | `OFF` | Build the benchmarks of the portable C API in a tree without MKL, against the BLAS/LAPACK `find_package(LAPACK)` finds (`BLA_VENDOR` selects it); implied by `CBK_WITH_MKL`. |
| `BUILD_SHARED_LIBS` | `OFF` | CMake's own switch: build `libcbk` shared instead of static. |
| `CBK_INSTALL` | `ON` when top-level | Generate the install rules; off by default under a parent project's `add_subdirectory`. |
| `MKLCompact_ROOT` | -- | A oneAPI MKL prefix, when `MKLROOT` is not set (MKL build only). |
| `MKLCompact_THREADING` | `sequential` | `threaded` links MKL's internally threaded layer (MKL build only). |
| `MKLCompact_INTERFACE` | `lp64` | `ilp64` selects MKL's 64-bit integer interface and defines `MKL_ILP64` (MKL build only). |

The threading gate's flop threshold is a preprocessor macro, not a CMake
option: `-DCMAKE_CXX_FLAGS="-DCBK_OMP_MIN_FLOPS=1e5"` overrides the default
`5e4` (see [threading.md](threading.md)).
