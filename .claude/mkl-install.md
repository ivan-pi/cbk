# Installing Intel MKL for this repo

Two sources work on Debian/Ubuntu. Both were exercised in September 2026; the
versions and paths below are what they installed then.

## 1. Distro package (what the repo and CI use)

```sh
sudo apt-get install libmkl-dev          # MKL 2020.0.4 (Debian 2020.4.304-4)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCBK_WITH_MKL=ON
```

Headers land in `/usr/include/mkl`, libraries in `/usr/lib/x86_64-linux-gnu`,
both on the default search paths, so nothing else is needed. This is what
`.claude/hooks/session-start.sh` installs on a remote session (Intel's apt
repository was blocked there when the hook was written; it is reachable now,
see below) and what `.github/workflows/*.yml` build against.

## 2. Intel oneAPI apt repository (current oneMKL)

Intel's instructions (`intel-oneapi-mkl` for runtime only,
`intel-oneapi-mkl-devel` for development):

```sh
curl -fsS https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update
sudo apt-get install intel-oneapi-mkl-devel   # 2026.1.0-236 in Sept 2026; ~4 GB on disk
```

`apt-cache madison intel-oneapi-mkl-devel` lists every version the configured
repositories offer, with its source (the subcommand imitates the output of
Debian's `madison` archive tool, hence the name); 2025.2 through 2026.1 were
listed at the time, and `intel-oneapi-mkl-devel-<ver>` pins one. The
install pulls in the compiler runtime (`libiomp5` under
`/opt/intel/oneapi/compiler/<ver>/lib`) but not the compilers themselves.

Layout: `/opt/intel/oneapi/mkl/latest` -> `<ver>`, with `include/`, `lib/`
(flat -- no `lib/intel64` since oneAPI 2024), `bin/`, `env/vars.sh`. The compact
header `include/mkl_compact.h` has the same prototypes as the 2020 one, and
still no `mkl_?ormqr_compact`.

### Building the repo against it

`cmake/FindMKLCompact.cmake` locates MKL directly and searches headers and
libraries in the same prefix order (package root, `$MKLROOT`, default paths,
`/opt/intel/oneapi/mkl/latest`), so with `libmkl-dev` also installed there is
no mixed build. Point it at oneAPI with either form:

```sh
MKLROOT=/opt/intel/oneapi/mkl/latest cmake -S . -B build-oneapi -DCBK_WITH_MKL=ON -DCMAKE_BUILD_TYPE=Release
cmake -S . -B build-oneapi -DCBK_WITH_MKL=ON -DMKLCompact_ROOT=/opt/intel/oneapi/mkl/latest -DCMAKE_BUILD_TYPE=Release
cmake --build build-oneapi -j
ctest --test-dir build-oneapi          # 17/17 on 2026.1
```

Check the cache: `MKLCompact_mkl_core_LIBRARY` must point under `/opt/intel`.
CMake embeds a RUNPATH, so the test binaries run without `LD_LIBRARY_PATH`; a
program linked by hand needs `LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib`
(or `source /opt/intel/oneapi/mkl/latest/env/vars.sh`). Verify what is loaded
with `ldd <binary> | grep mkl_core` or, at run time, `mkl_get_version_string()`.

Hand-built probes (as in `.claude/mkl-compact-behavior.md`):

```sh
gcc probe.c -I/opt/intel/oneapi/mkl/latest/include -L/opt/intel/oneapi/mkl/latest/lib \
    -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
```

That is the line Intel's link advisor gives for a GNU C compiler; the oneAPI
package ships the advisor as `/opt/intel/oneapi/mkl/latest/bin/mkl_link_tool`
(the web version at intel.com is unreachable from the remote session), and its
output for the cases that matter here:

```sh
mkl_link_tool -libs -c gnu_c -l dynamic -i lp64 -p no           # sequential
#  -m64 -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
mkl_link_tool -libs -c gnu_c -l dynamic -i lp64 -p yes -o gomp  # GNU-threaded
#  -m64 -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
mkl_link_tool -libs -c gnu_c -l static  -i lp64 -p no           # static
#  -m64 -Wl,--start-group ${MKLROOT}/lib/libmkl_intel_lp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
```

(`-p yes|no` selects threading, `-o gomp|iomp5` the OpenMP runtime.) The
sequential-layer probes here also linked and ran fine without
`-Wl,--no-as-needed`; `cmake/FindMKLCompact.cmake` links the same libraries by
full path, which needs neither the flag nor the group.

### Build options

- **`-DMKLCompact_THREADING=sequential`** (default): links `mkl_sequential`.
  MKL runs single-threaded inside each call; cbk's own group loops provide the
  parallelism. This is what CI and the benchmarks use.
- **`-DMKLCompact_THREADING=threaded`**: links MKL's OpenMP layer matched to
  the compiler's own OpenMP runtime, so a process never carries two runtimes:
  - g++: `mkl_gnu_thread` on libgomp.
  - clang++: `mkl_gnu_thread` on LLVM's libomp, which exports the GOMP entry
    points; no libgomp is loaded (Intel's `MKLConfig.cmake` would add `-lgomp`
    here).
  - Intel compilers: `mkl_intel_thread` on `libiomp5`.
  The full test suite passes threaded under g++ and clang++. MKL's workspace
  query then scales with its thread count, and `lwork` is never checked: size
  every buffer from a query made under the thread count of the call
  (`docs/building.md`; `.claude/mkl-compact-behavior.md`, section 2).
- **`-DMKLCompact_INTERFACE=lp64`** (default) or **`ilp64`**: the integer
  interface; `ilp64` also defines `MKL_ILP64` on the headers target.
- **`-DMKLCompact_ROOT=<prefix>`** or **`MKLROOT`** in the environment: the
  installation to use when more than one is present or none is on the default
  paths.
- **`BLA_VENDOR`** is not read. No other BLAS has the compact API, and a stale
  `-DBLA_VENDOR=...` only draws CMake's "manually-specified variables were not
  used" warning.
