# Installing Intel MKL for this repo

Two sources work on Debian/Ubuntu. Both were exercised in September 2026; the
versions and paths below are what they installed then.

## 1. Distro package (what the repo and CI use)

```sh
sudo apt-get install libmkl-dev          # MKL 2020.0.4 (Debian 2020.4.304-4)
cmake -S . -B build -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_BUILD_TYPE=Release
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

`apt-cache madison intel-oneapi-mkl-devel` lists the available versions (2025.2
through 2026.1 at the time); `intel-oneapi-mkl-devel-<ver>` pins one. The
install pulls in the compiler runtime (`libiomp5` under
`/opt/intel/oneapi/compiler/<ver>/lib`) but not the compilers themselves.

Layout: `/opt/intel/oneapi/mkl/latest` -> `<ver>`, with `include/`, `lib/`
(flat -- no `lib/intel64` since oneAPI 2024), `bin/`, `env/vars.sh`. The compact
header `include/mkl_compact.h` has the same prototypes as the 2020 one, and
still no `mkl_?ormqr_compact`.

### Building the repo against it

Coexistence with `libmkl-dev` is the trap: pointing `MKLROOT` at oneAPI makes
`cmake/FindMKLCompact.cmake` pick up the oneAPI *headers* (it hints
`$ENV{MKLROOT}/include`), but CMake 3.28's FindBLAS still resolved the
*libraries* from `/usr/lib/x86_64-linux-gnu` -- a mixed build that happens to
run (the ABI is compatible) but tests the wrong library. Add a library-path
hint so both come from oneAPI:

```sh
MKLROOT=/opt/intel/oneapi/mkl/latest cmake -S . -B build-oneapi \
  -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib
cmake --build build-oneapi -j
ctest --test-dir build-oneapi          # 17/17 on 2026.1
```

Check the cache: `BLAS_mkl_core_LIBRARY` must point under `/opt/intel`. CMake
embeds a RUNPATH, so the test binaries run without `LD_LIBRARY_PATH`; a
program linked by hand needs `LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib`
(or `source /opt/intel/oneapi/mkl/latest/env/vars.sh`). Verify what is loaded
with `ldd <binary> | grep mkl_core` or, at run time, `mkl_get_version_string()`.

Hand-built probes (as in `.claude/mkl-compact-behavior.md`):

```sh
gcc probe.c -I/opt/intel/oneapi/mkl/latest/include -L/opt/intel/oneapi/mkl/latest/lib \
    -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
```

`-Wl,--no-as-needed` is what Intel's link advisor prescribes for the dynamic
libraries; the sequential-layer probes here also linked fine without it.

### Threading layer

`-DBLA_VENDOR=Intel10_64lp` asks for the threaded layer, but on this C/C++-only
project FindBLAS pairs `mkl_intel_thread` with `iomp5`, which is not on the
default library path: the link succeeds and the binary segfaults at startup.
Do not use it without settling the OpenMP runtime for the whole build (see
`.claude/mkl-compact-behavior.md`, section 3b). The sequential layer is the
repo's choice.
