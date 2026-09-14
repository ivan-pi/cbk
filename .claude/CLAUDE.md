# CLAUDE.md

Working notes for contributors (human or agent) to **cbk** (Compact Batch
Kernels, libcbk), a library of batched QR, Cholesky and LDL^T factorizations
for many small matrices in the
compact (interleaved) format: a portable C API over SIMD kernels, plus an
optional Intel MKL-style API that drops into MKL's compact ecosystem.

## Build and test

The default configure builds the portable library and its tests. The library
needs nothing but CMake and a C++17 compiler; the tests validate against a
real LAPACKE + CBLAS (`cmake/FindLAPACKE.cmake`: OpenBLAS or Netlib without
MKL, `libopenblas-dev liblapacke-dev` / `liblapacke-dev liblapack-dev` on
Debian, MKL's own with the extension; `-DCBK_TEST_LAPACK=openblas|netlib|mkl`
pins one -- do that, since Debian's alternatives symlinks let either stand in
for the other):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`-DCBK_WITH_MKL=ON` adds the MKL-style API, the MKL-backed suites (validated
against MKL's compact kernels; every suite's LAPACKE is then MKL's), the
example and the benchmarks; it needs Intel MKL (`sudo apt-get install libmkl-dev` on
Debian/Ubuntu). Run that build too before pushing changes to the kernels or
the wrappers: it is the one that cross-checks against MKL.

Both APIs are one library target, `cbk` (`cbk::cbk`); `-DBUILD_SHARED_LIBS=ON`
builds it shared. `cmake --install build --prefix <p>` installs it, the
headers of what was built (the two MKL headers only with the extension), and a
`find_package(cbk CONFIG)` package (`cmake/cbkConfig.cmake.in`).
CI verifies an install tree and builds the consumer project in
`tests/install/` against it for every combination (`.github/workflows/install.yml`
and its `.github/check_install.sh`).
`.claude/mkl-install.md` covers installing MKL from the distro package or from
Intel's oneAPI apt repository, and how to point the build at a oneAPI install
(`MKLROOT` or `-DMKLCompact_ROOT`; `-DMKLCompact_THREADING=threaded` for MKL's
threaded layer).

## Performance builds

The library sets no `-march` of its own; optimization flags are the caller's to
choose. For a fast build, pass them through `CMAKE_CXX_FLAGS` so the SIMD kernels
target the host's widest vectors:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"
```

Correctness is independent of these flags; only throughput changes.

## Tree

```
include/   public headers: cbk.h (portable C API), cbk_compat.h
           (MKL-style API), cbk_mkl_alloc.h (optional RAII mkl_malloc helpers)
src/       the templated kernels (cbk_*_compact.hpp, one per routine, on the
           shared cbk_common.hpp; posv's and sysvnp's are drivers over
           the potrf/potrs and sytrfnp/sytrsnp group kernels), the two adapter
           sources that implement
           the public headers (cbk.cpp, cbk_compat.cpp), and
           cbk_matrix_view.hpp and cbk_matrix_batch.hpp, the dense MatrixView
           and the owning MatrixBatch the tests, benchmarks and examples share
           (internal: src/ is on their include path, but the public API stays
           include/)
tests/     the portable C API's suites (vs LAPACKE/CBLAS on any stack) and
           the MKL-backed ones, templated on the scalar type;
           test_compact_util.hpp (compact<T>, the generators, pack/unpack),
           test_lapack_util.hpp (lapack<T> and the ref_* procedures over
           views, matmul, tri_apply, solve_errors) and test_mkl_util.hpp
           (compat<T>, mkl<T>) hold the helpers and dispatch structs
           (tests/install/: the install check's consumer project and script)
examples/  the worked solve and the benchmarks (BENCHMARKS.md), on bench_util.hpp
docs/      one design document per routine, the guides README.md indexes
           (building, threading, layout, examples), and PLANS.md, the internal
           status of each routine and what remains
```

## Formatting and linting

Layout follows C++ Core Guidelines **NL.17** (K&R-derived / "Stroustrup"), the
style of `.clang-format`; `.clang-tidy` lints (the bug finders, the performance
checks, and the parts of modernize and readability that are not a matter of
taste). Both run through [pre-commit](https://pre-commit.com/), whose
`.pre-commit-config.yaml` pins the clang-format version CI checks with
(`.github/workflows/style.yml`):

```sh
pip install pre-commit
pre-commit install                 # format on every commit from now on
pre-commit run --all-files         # or by hand, over the whole tree
```

clang-tidy wants the compile database of a tree configured with **clang** (it
parses with clang's front end, which cannot read g++'s `omp.h`; `libomp-dev`
supplies clang's) and is a manual stage:

```sh
CXX=clang++ cmake -S . -B build-tidy
pre-commit run --hook-stage manual clang-tidy --all-files
```

Use the clang-tidy of the same LLVM release as that clang++. Only the
translation units are listed; the headers are checked through them. CI runs
this only on manual dispatch (`.github/workflows/clang-tidy.yml`), since the
runner has to install MKL and an LLVM toolchain first; run it locally before
pushing changes to the kernels.

Hand-aligned tables and compact one-liners that clang-format would expand are
fenced with `// clang-format off` / `// clang-format on`; leave those fences in
place. The dispatch macros (`CBK_TEST_*_DISPATCH` in the test headers,
`CBK_DEFINE_*_ENTRY_POINTS` in the two adapter sources) take a type name as an
argument, which cannot be parenthesized, so they also sit between
`// NOLINTBEGIN(bugprone-macro-parentheses)` and the matching `NOLINTEND`.

## Claude Code hooks

`.claude/settings.json` wires up two hooks:

- `.claude/hooks/session-start.sh` provisions a fresh remote session: Intel MKL,
  clang's OpenMP runtime, and pre-commit with its hook environments. It does
  nothing on a developer's own machine.
- `.claude/hooks/format.sh` runs after every `Edit` or `Write`: the pre-commit
  hooks on that one file, so Claude's edits come out the way a commit would.
  clang-format fixes silently; a finding the hooks cannot fix is fed back to
  Claude to correct. Without pre-commit it falls back to the system
  clang-format.

## Intel MKL Compact: measured behavior

`.claude/mkl-compact-behavior.md` records what MKL's own compact routines were
measured to do, identically on MKL 2020.0.4 (the Debian `libmkl-dev` package the
repo builds against) and oneMKL 2026.1: every pointer argument including `info` and `work`
is dereferenced unconditionally; `?geqrf`/`?getrinp` use `work` as their
scratch, sized `n*V` per thread (`n*V*mkl_get_max_threads()` under a threaded
MKL layer) and never check `lwork`, so an undersized buffer is a silent overrun;
the compact kernels thread internally only under the threaded layer, which this
repo does not link. Read it before touching the MKL-style wrappers, the
workspace contract, or the benchmarks' threading.

## Conventions worth knowing

- **Compact layout.** Element `(i,j)` of the `V` interleaved matrices in a group
  is stored contiguously; `V` is the SIMD width (FP64: 2/4/8 for SSE/AVX/AVX-512).
  A partial final group is padded with identities, so kernels run it unmasked.
- **SIMD via GNU vector types.** The kernels use
  `__attribute__((vector_size))` vectors, which the compiler lowers to the target
  ISA -- one portable source for every width. The pack type carries a relaxed
  `aligned(alignof(T))`; always name it as `typename pack<T,V>::type` and never
  pass it as a template *argument* (clang strips typedef alignment there and
  emits aligned loads that fault on 16-byte-aligned buffers -- issue #34).
- **Never pass a pack across a call by reference.** The second face of the
  same clang behavior (issue #34 fixed the template-argument face in PR #35):
  a `const typename pack<T,V>::type &` parameter is loaded with the *natural*
  vector alignment once the call is not inlined -- the typedef's relaxed
  alignment does not survive on the referent -- and faults on a buffer or stack
  local that is only `T`-aligned. The `JB = 4` block helper of `sytrfnp` was
  first written that way (the pivot `d` passed by reference from the caller's
  local) and segfaulted on a `movapd` under clang in the plain Release
  configuration CI uses, while the same source passed every test under gcc.
  Passing by value is no escape (`-Wpsabi`, see `cbk_common.hpp`).
  Instead give the helper the view and the indices and let it load what it
  needs (`sytrfnp_update_block`, which `potrf`'s panel calls too, does exactly that), or
  pass the scalar the pack was broadcast from. The tiny lane-wise helpers
  (`vsqrt`, `broadcast`, `trsm_dot_block`'s `va`) get away with references only
  because they always inline.
- **clang splits 512-bit packs in half unless told otherwise.** On x86, clang
  (icpx included) legalizes vector operations to its *preferred* width, which
  its tuning for AVX-512 CPUs sets to 256 bits: a V=8 double pack becomes two
  ymm halves, half the FMA rate and twice the register pressure, so the
  register-tiled updates spill (measured: the blocked potrf ran at ~15 GFLOP/s
  under clang against ~28 under gcc from the same source; icpx defaults its
  zmm usage the same way, `-qopt-zmm-usage=low`, on pre-Sapphire-Rapids
  targets). Every kernel header therefore wraps its body in
  `CBK_KERNEL_BEGIN` / `CBK_KERNEL_END` (`cbk_common.hpp`), which under clang
  marks every function declared there `min_vector_width(512)`, the attribute
  clang's own intrinsics headers use; wrap new kernel headers the same way.
  GCC lowers explicit vector types at their natural width and needs nothing.
  Check codegen with `objdump -d libcbk.a | grep -c zmm` (zero means split).
- **Build and test with both gcc and clang before pushing.** CI runs both, and
  the packs' alignment is exactly the kind of contract only one of them
  enforces: both alignment faults so far (issue #34 and the one above) were
  invisible to gcc. The default (portable) tree needs only the compiler and a
  LAPACKE stack for the tests (CI runs OpenBLAS and Netlib under both
  compilers):

  ```sh
  CXX=clang++ cmake -S . -B build-clang -DCBK_TEST_LAPACK=openblas && cmake --build build-clang && ctest --test-dir build-clang
  ```

- **Threading over groups.** Every all-groups driver is a call to
  `for_each_group<V>(nm, flops_per_group, body)` (`cbk_common.hpp`):
  a static-schedule `omp parallel for` on at most one thread per group, gated
  by an if-clause. Keep new drivers on it; do not add threading inside a group
  kernel. The gate refuses when nesting is exhausted (that is what makes the
  library compose with a caller's outer parallel loop), when there is a single
  group, or below `parallel_min_flops`.
- **Two views, one idea.** Compact (packed) operands are addressed through
  `BatchView` (`src/cbk_common.hpp`), dense host-side ones through
  `MatrixView` (`src/cbk_matrix_view.hpp`). Both carry the layout as runtime
  strides `(si, sj)`, so one body serves column-major and row-major and a
  transpose is a stride swap. Do not hand-write `A[i + (size_t)j * lda]` in
  new tests, benchmarks or examples -- take a view. `MatrixView` asserts its
  bounds, so run the suites once in a `Debug` build when adding indexing code.
  The tests use both: the pack/unpack helpers in `test_compact_util.hpp` write
  the interleaved side through `BatchView` (`for_vlen` turns their runtime `V`
  into its compile-time one) and the dense side through `MatrixBatch`. The
  library and its tests are one internal codebase and share these views on
  purpose; what keeps the suites honest is that they compute the *answers*
  independently -- scalar LAPACK references, dense LAPACK/MKL cross-checks.
  The dense references are the library's: `test_lapack_util.hpp` hands a
  view's `data` and leading dimension to LAPACKE/CBLAS (`is_rowmajor`,
  `lapack_ld`, `lapack_layout`), reads a BLAS-3 operand stored the other way
  round as its transpose, and stages the few LAPACKE operands that have no
  transpose flag (`Staged`).
  A routine that takes views (or batches) `assert`s what its contract
  assumes: dimension compatibility across operands (`matmul`, `tri_apply`,
  `solve_errors`, the `ref_*` procedures), squareness where required
  (`gen_spd`, `gen_tri`, `ref_potf2`, ...), and index ranges
  (`MatrixView::operator()`, `MatrixBatch::operator[]`). Costs nothing in
  Release, and the Debug suite run is what exercises it. `BatchView` is the
  exception by construction -- it carries strides, not extents, so a kernel
  cannot self-check; the kernels' dimension contract is the portable C API's
  argument validation.
  Pass a `BatchView` **by value** to any function that may stay out of line
  (a recursive driver, a large block helper): the pack element type is
  `may_alias`, so after every store through a view held *by reference* the
  compiler must reload its `data`/`si`/`sj` and redo the `i*si + j*sj`
  multiplies -- potrf's recursion measured 1.3-1.7x slower at `n <= 32` that
  way. The view is 16 bytes, two registers. The small `inline` helpers that
  always inline may keep `const BatchView &`.
  Dense batches everywhere are `MatrixBatch` (`src/cbk_matrix_batch.hpp`):
  storage, the per-matrix `view(v)`, and the `base_ptrs()` array the MKL
  pack/unpack routines take. The benchmarks alias it as `MatrixPool`
  (`examples/bench_util.hpp`) with 64-byte-aligned storage and pair it with
  `PackedPool`, the pristine compact image their timed passes restore from;
  each benchmark keeps only its fill.
- **One kernel per routine.** Every kernel addresses its operands through
  `BatchView` (strides `si`, `sj`), so column-major, row-major, and ormqr's
  `side='R'` are the same code with different strides -- and gels's LQ case is
  geqrf's kernel over the transposed view. The fused solves (gels, posv, sysvnp)
  compose the factorization and solve *group* kernels inside one
  `for_each_group` body; add a fused driver the same way rather than
  duplicating arithmetic. Do not add parameters to an existing group kernel's
  signature for a new caller, even ones compiled out: GCC's interprocedural
  passes key on the signature, and geqrf's plain path lost 8-13% that way
  (measured). Share the step as an inline helper and give the new caller its
  own kernel. Register blocking is
  written as a `JB`-templated block helper with `for (c < JB)` loops the
  compiler unrolls, not as hand-expanded `w0..w3` copies.
- **Argument checking.** The MKL-style API (`cbk_*`) skips validation like
  MKL's own compact routines (`info` is a scalar, `0` on success). The portable C
  API (`cbk.h`) validates LAPACK-style, returning `-j` for a bad j-th
  argument.
- **Workspace (`lwork`).** Size each routine's `work` from *its own* `lwork = -1`
  query, and give each routine its own buffer. The compact kernels here need no
  scratch (their query returns `1`) except the MKL-style `gels`, whose `work`
  is its `tau` scratch, one slot per group (`min(m,n)*V*ceil(nm/V)`, holding
  `tau` on exit; the portable `?gels_compact` takes it as an explicit `taup`
  output instead); MKL's `mkl_?geqrf_compact` needs `~n*V`.
  Because compact routines skip argument checking, handing one routine a `work`
  sized for another -- or sharing a buffer across `geqrf`/`ormqr` -- is undefined
  behavior: harmless on some MKL builds, silent heap corruption on others (this
  is exactly the bug that aborted the MKL test with "unaligned tcache chunk").
- **Buffer alignment.** Compact buffers are correct at any `T` alignment, but
  align the base to the pack width (64 B covers every format) so the SIMD kernels
  avoid cache-line splits -- `mkl_malloc(bytes, 64)`, which is what
  `mkl_alloc_bytes` does by default, or `std::aligned_alloc(64, ...)`. Only
  performance, not correctness, rides on it (up to ~40% on small sizes).
- **Default benchmark sizes stop at 256.** `bench_sizes`
  (`examples/bench_util.hpp`) is the one square size list the benchmarks run --
  change the set there, not per program (`bench_qr_compact` keeps its own). It
  ends at `256`: the larger orders made a run take ten minutes, and the batched
  gains are not there to be had anyway -- they live below `128`, most of them
  below `64`. `--size-sweep` is deliberately uncapped, so a scan can still go as
  large as the caller wants.
- **Scope.** Real precisions (`s`/`d`) only; no column pivoting; no overflow/
  underflow-safe reflector rescaling (see the design documents).
