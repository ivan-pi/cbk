# Installing Intel MKL for this repo

Two sources work on Debian/Ubuntu. Both were exercised in September 2026; the
versions and paths below are what they installed then.

## 1. Distro package (what the repo and CI use)

```sh
sudo apt-get install libmkl-dev          # MKL 2020.0.4 (Debian 2020.4.304-4)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
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

`cmake/FindMKLCompact.cmake` locates MKL directly and takes headers and
libraries from one installation (the library dir is derived from where
`mkl_compact.h` was found), so with `libmkl-dev` also installed there is no
mixed build. Point it at oneAPI with either form:

```sh
MKLROOT=/opt/intel/oneapi/mkl/latest cmake -S . -B build-oneapi -DCMAKE_BUILD_TYPE=Release
cmake -S . -B build-oneapi -DMKLCompact_ROOT=/opt/intel/oneapi/mkl/latest -DCMAKE_BUILD_TYPE=Release
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

### Threading layer

`-DMKLCompact_THREADING=sequential` (default) links `mkl_sequential`; `gnu`
links `mkl_gnu_thread` with the compiler's own OpenMP runtime (libgomp under
g++, libomp under clang, which exports the GOMP entry points; the same recipe
as Intel's `MKLConfig.cmake`) and the full test suite passes either way;
`intel` links `mkl_intel_thread` + `libiomp5`, which is only right with an
Intel compiler -- under g++/clang it puts two OpenMP runtimes in one process,
the configure step warns, and tests fail. Under a threaded layer remember
that MKL's workspace query scales with its thread count (see
`.claude/mkl-compact-behavior.md`).
`-DMKLCompact_INTERFACE=ilp64` selects the ILP64 interface (and defines
`MKL_ILP64`). The old `-DBLA_VENDOR=Intel10_64lp[_seq]` spellings still map
onto these; any other `BLA_VENDOR` is rejected, since no other BLAS has the
compact API.
