# Building

## The portable library (default)

Requires **CMake >= 3.28**, a **C++17 compiler** (GCC/Clang) and a build tool
(Make/Ninja). The library needs nothing else. The test suites validate every
kernel against a real **LAPACKE + CBLAS** (issue #27), so the default
configure, which builds them, also needs one of

* OpenBLAS -- Debian/Ubuntu: `sudo apt-get install libopenblas-dev liblapacke-dev`
  (Debian ships OpenBLAS's LAPACKE as Netlib's `liblapacke`), or
* Netlib's reference LAPACK -- `sudo apt-get install liblapacke-dev liblapack-dev libblas-dev`,
  or
* Intel MKL's own, with the MKL extension below (then it is the only stack the
  tests may link).

`cmake/FindLAPACKE.cmake` picks the first it finds (OpenBLAS, then Netlib,
then any `lapacke` library next to what `find_package(LAPACK)` returns) and
reports it as `LAPACKE for the tests: ...`; `-DCBK_TEST_LAPACK=openblas|netlib|mkl`
pins one, which matters on Debian, where `liblapack.so` is an
`update-alternatives` symlink and either stack can otherwise stand in for the
other. `-DCBK_BUILD_TESTS=OFF` drops the requirement with the tests.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

On macOS, Accelerate provides LAPACK and CBLAS but no LAPACKE;
[accelerate-lapacke](https://github.com/lepus2589/accelerate-lapacke) builds
one on top of it (its configure needs a Fortran compiler, `brew install gcc`),
and `-DCBK_TEST_LAPACK=accelerate` with `-DCMAKE_PREFIX_PATH` at its install
(its own prefix; the workflow caches it) looks for that: the `lapacke` package plus
Accelerate's own CBLAS through `<vecLib/cblas.h>`. `.github/workflows/macos.yml`
runs exactly this on the macOS runners, one job next to the Linux matrix.

## The Intel MKL extension (`-DCBK_WITH_MKL=ON`)

`-DCBK_WITH_MKL=ON` adds the MKL-style API (the `cbk_?*_compact` entry
points of `include/cbk_compat.h`, compiled into the same `cbk::cbk` library),
the MKL-backed
test suites (cross-checked against MKL's own compact kernels), the worked
example and the benchmarks; every suite's dense references then come from
MKL's LAPACKE. It needs **Intel MKL**,
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

Two codegen settings the library does fix itself, because without them the
kernels compile to something much slower than their source says, on any
`-march`:

* **`-fno-math-errno`**, set on the `cbk` target (PRIVATE; consumers are not
  bound to it). The pivots' lane-wise square root becomes one vector
  instruction only when the compiler need not set `errno` on a negative
  input; with math errno on (GCC's and clang's default on Linux) each lane is a
  guarded scalar square root plus a libm call, under both compilers. The
  kernels never read `errno`; a non-SPD lane gets its NaN either way. If you
  build the sources with your own build system, pass the flag.
* **Full-width vectors under clang and icpx.** On x86 both legalize vector
  operations to their *preferred* width, which their tuning for AVX-512 CPUs
  sets to 256 bits, so a 512-bit pack (`V = 8` doubles, the AVX-512 compact
  format) would be split into two ymm halves: half the FMA rate and twice the
  register pressure. The kernel headers mark their functions
  `min_vector_width(512)` (`CBK_KERNEL_BEGIN` in `src/cbk_common.hpp`), which
  restores full-width codegen for them alone; the equivalent whole-build
  flags are `-mprefer-vector-width=512` (clang) and `-qopt-zmm-usage=high`
  (icpx). GCC lowers explicit vector types at their natural width and needs
  neither. `objdump -d libcbk.a | grep -c zmm` shows whether a build got it.

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
| `CBK_BUILD_TESTS` | `ON` | Build the tests and register them with CTest; needs a LAPACKE + CBLAS stack. |
| `CBK_TEST_LAPACK` | `auto` | The stack the tests validate against: `auto` (MKL's with the MKL extension, else the first of OpenBLAS, Netlib, any), `mkl`, `openblas`, `netlib`, `accelerate`. |
| `BUILD_SHARED_LIBS` | `OFF` | CMake's own switch: build `libcbk` shared instead of static. |
| `CBK_INSTALL` | `ON` when top-level | Generate the install rules; off by default under a parent project's `add_subdirectory`. |
| `MKLCompact_ROOT` | -- | A oneAPI MKL prefix, when `MKLROOT` is not set (MKL build only). |
| `MKLCompact_THREADING` | `sequential` | `threaded` links MKL's internally threaded layer (MKL build only). |
| `MKLCompact_INTERFACE` | `lp64` | `ilp64` selects MKL's 64-bit integer interface and defines `MKL_ILP64` (MKL build only). |

The threading gate's flop threshold is a preprocessor macro, not a CMake
option: `-DCMAKE_CXX_FLAGS="-DCBK_OMP_MIN_FLOPS=1e5"` overrides the default
`5e4` (see [threading.md](threading.md)).
