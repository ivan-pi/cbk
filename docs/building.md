# Building

## Prerequisites

* **CMake >= 3.18**, a **C++17 compiler** (GCC/Clang) and a build tool (Make/Ninja).
* **Intel MKL** - provides the Compact-format extension (`mkl_compact.h`,
  `mkl_*geqrf_compact`, ...) that this project builds on. Any MKL works:
  * oneAPI MKL - `source /opt/intel/oneapi/setvars.sh` (sets `MKLROOT`), or
  * Debian/Ubuntu - `sudo apt-get install libmkl-dev` (headers in
    `/usr/include/mkl`, LP64 libs in the default library path).

`-DCQR_WITH_MKL=OFF` builds the portable kernels alone: no MKL, no MKL-backed
tests, examples or benchmarks.

## Configure, build, test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Locating MKL

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
> `cqr_mkl_?gels_compact` also uses `work` (as the `tau` scratch of its
> factorization, `min(m,n) * V * ceil(nm/V)` scalars, independent of the thread
> count); size it from its own query. The other cqr routines need no scratch.
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

## Options

| Option | Default | Effect |
|--------|---------|--------|
| `CQR_WITH_MKL` | `ON` | Build the MKL-style API, the MKL-backed tests, the example and the benchmarks. `OFF` leaves the portable kernels and their BLAS-free tests. |
| `CQR_WITH_OPENMP` | `ON` | Thread the loop over groups with OpenMP; `OFF` gives single-threaded routines. See [threading.md](threading.md). |
| `CQR_BUILD_TESTS` | `ON` | Build the tests and register them with CTest. |
| `MKLCompact_ROOT` | -- | A oneAPI MKL prefix, when `MKLROOT` is not set. |
| `MKLCompact_THREADING` | `sequential` | `threaded` links MKL's internally threaded layer. |
| `MKLCompact_INTERFACE` | `lp64` | `ilp64` selects MKL's 64-bit integer interface (and defines `MKL_ILP64`). |

The threading gate's flop threshold is a preprocessor macro, not a CMake
option: `-DCMAKE_CXX_FLAGS="-DCQR_OMP_MIN_FLOPS=1e5"` overrides the default
`5e4` (see [threading.md](threading.md)).
