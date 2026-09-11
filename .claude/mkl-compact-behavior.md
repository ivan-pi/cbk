# Intel MKL Compact routines: measured behavior

Empirical notes on how MKL's own compact routines treat their arguments, gathered
while settling the `info`/`work` contract of `cqr_mkl_ext.cpp` (PR #38). The
MKL documentation leaves most of this unstated ("info is reserved", "no error
checking"), so these were measured directly. Everything below is from small C
probes against the installed library; nothing is inferred from documentation.

**Checked against two MKL versions, six years apart.** Everything here was
measured on both of:

    Intel(R) Math Kernel Library Version 2020.0.4 Product Build 20200917
    (Debian/Ubuntu package libmkl-dev 2020.4.304-4 -- what the repo builds against)

    Intel(R) oneAPI Math Kernel Library Version 2026.1-Product Build 20260612
    (Intel apt repository, package intel-oneapi-mkl-devel 2026.1.0-236,
     installed under /opt/intel/oneapi/mkl/latest)

LP64 interface, on a 4-core AVX-512 Xeon (1 MiB L2 per core, 33 MiB L3) with
gcc 13, in September 2026. Every finding below -- the null-pointer crashes, the
`info` values, the workspace formula and its thread scaling, the unchecked
`lwork`, the leftover `work[0]`, zero internal allocation, internal threading,
and the exit-time crash -- came out identical on the two versions, so this
behavior has been stable across the MKL-to-oneMKL transition. It is still
measured behavior, not a documented contract: re-run the probes of section 5
on any other version before relying on it, and `mkl_get_version_string()`
reports the version actually linked.

Routines probed: `mkl_?potrf_compact`, `mkl_?getrfnp_compact`,
`mkl_?geqrf_compact`, `mkl_?getrinp_compact`, plus `mkl_?trsm_compact` for
threading only. Which compact routines MKL ships at all is answered by
`mkl_compact.h`, not by this note.

## 1. Every pointer argument is mandatory

| Routine | `info` = NULL | `work` = NULL (query) | `work` = NULL (call) | `taup` = NULL |
|---|---|---|---|---|
| potrf | segfault | no `work` arg | no `work` arg | n/a |
| getrfnp | segfault | no `work` arg | no `work` arg | n/a |
| geqrf | segfault | segfault | segfault | segfault |
| getrinp | segfault | segfault | segfault | n/a |

- **`info` is written unconditionally, and the value written is always 0.** A
  sentinel (777, or -5) came back as 0 from every routine on every path:
  success, the `lwork = -1` query, `nm = 0`, `n = 0`, and inputs LAPACK would
  reject.
- **Numerical failure never reaches `info`.** A non-positive-definite lane for
  `potrf` (LAPACK: `info = j`), a zero pivot for `getrfnp`, and `getrinp` on
  that singular factor all left `info = 0` and poisoned the offending lane with
  NaN/Inf; the other lanes of the group came out correct. "Reserved" means
  exactly that: `info` carries no information, and per-lane failure is
  detectable only by inspecting the output.
- **`taup` always receives the reflector scalars**; a null `taup` is not "skip
  the copy".
- **Consequence for cqr**: the MKL-style wrappers write `*info` and `work[0]`
  without null checks, matching MKL; a null pointer fails loudly at the first
  write, which is the benign failure mode. cqr keeps `info = 0` on success and
  a poisoned lane on numerical failure, and adds `info = -1` for an
  unrecognized `format`, the one failure its dispatch can see.

## 2. Workspace: size, use, and the unchecked `lwork`

Only `geqrf` and `getrinp` take `work`/`lwork`. For both:

- **Query formula**: `lwork = n * V * mkl_get_max_threads()`. Independent of `m`
  and of the number of groups; one `n * V` slice per thread (section 3). Under
  the sequential layer the multiplier is always 1.
- **`work` is the scratch, not a hint**: a real call writes `(n-1)*V` entries per
  thread (`geqrf`) or `(n-1)*V + 1` (`getrinp`). `mkl_peak_mem_usage` reports 0
  bytes allocated internally, so nothing is allocated behind the caller's back.
- **`lwork` is examined only for the value -1** (the query). Any other value is
  ignored: with a one-element buffer and `lwork = 1`, both routines wrote their
  full `n * V` scratch and hit a guard page (section 5). An undersized buffer is
  therefore an **out-of-bounds write (CWE-787)** into whatever is adjacent: a
  neighbouring allocation, malloc metadata ("malloc(): unaligned tcache chunk"
  at a later `free`), or a stack frame. It is silent unless the adjacent memory
  happens to matter. A generously oversized buffer is safe by accident; only the
  query makes the size a contract.
- **`work[0]` after a real call** holds leftover scratch, not the optimal
  `lwork` the documentation promises. The query is the only way to obtain the
  size.

Rule that follows: **query each routine, size its buffer from that routine's
own answer, never reuse one routine's buffer for another.** Under a threaded MKL
layer add: **query and call under the same thread count** (section 3).

*cqr's own policy here is not settled and may change.* As of PR #38:
`?gels_compact`'s `work` is the tau buffer for the whole batch,
`min(m,n) * V * ceil(nm/V)` scalars, and like every compact routine it does not
check `lwork`, so it carries the same CWE-787 hazard and must be sized from its
own query. `?geqrf`/`?ormqr` need no scratch (their `larf` keeps four columns'
dot products in registers instead of forming a full row vector), so their query
returns 1 and a real call never touches `work`.

## 3. Threading

MKL's compact routines are threaded internally, but only when the threaded
layer is linked (`mkl_gnu_thread` or `mkl_intel_thread`). Whole-batch calls,
AVX-512 format, best of five:

| routine, n, matrices | 1 thread | 2 threads | 4 threads |
|---|---|---|---|
| geqrf, 32, 8192 | 22.8 ms | 13.6 ms | 7.3 ms |
| geqrf, 64, 4096 | 100 ms | 57 ms | 28.6 ms |
| geqrf, 120, 1024 | 163 ms | 82 ms | 46.7 ms |
| potrf, 32, 8192 | 9.9 ms | 3.6 ms | 2.0 ms |
| trsm, 32, 8192 | 13.3 ms | 6.3 ms | 3.3 ms |

- **The workspace query scales with the *effective* thread count.** After
  `mkl_set_num_threads(t)`, `geqrf`/`getrinp` report `n * V * mkl_get_max_threads()`;
  asking for 8 or 16 threads on a 4-core machine clamps both to 4. A buffer
  queried at one thread and used at four overruns by three `n * V` slices (this
  crashed the first threaded probe). Under `mkl_sequential`,
  `mkl_set_num_threads` is accepted and ignored, `mkl_get_max_threads()` stays
  1, and the query stays `n * V`.
- **The repo currently links the sequential layer** (`BLA_VENDOR=Intel10_64lp_seq`
  in CMake and CI) and the benchmarks thread over groups with their own OpenMP
  loop, so every path (cqr, MKL compact, per-matrix LAPACK) runs on the same
  caller-controlled thread count. Measured against MKL's internal threading
  that loop is a fair proxy at n >= 16 (whole-batch at 4 threads: 143k vs 147k
  matrices/s at n=64, 1.12M vs 1.06M at n=32). *This choice may change*: using
  MKL's internal parallelism is an option. Its downside is for fused pipelines
  (`bench_qr_compact`'s factor -> apply -> solve per group): whole-batch calls
  stream the batch through each step separately and lose the per-group
  temporal locality that the per-group loop keeps.
- **Exit-time crash with `mkl_gnu_thread`.** A main program compiled with
  `-fopenmp` (libgomp) and linked against `mkl_gnu_thread` segfaults in
  `_dl_fini` after `main` returns, on both 2020.0.4 and 2026.1, with results
  already correct (stdout must be unbuffered to see them). The same program
  without `-fopenmp` exits cleanly. Under ctest that crash is a failed test.
  (Loading two OpenMP runtimes in one process -- libgomp for cqr's launcher
  and libiomp5 for `mkl_intel_thread` -- is never valid; a probe that did so
  during this investigation is disregarded.)

### 3a. Calls from inside an OpenMP parallel region

Whether MKL nests its own team inside the caller's region follows the
documented MKL rule, and the compact routines obey it. Probe: two batches of
2048 64x64 matrices, one per thread of a 2-thread `omp parallel` region, on 4
cores; CPU time / wall time counts the threads actually working. References
outside any region: 69 ms per batch on 1 thread, 18 ms on 4. Identical on
2020.0.4 and 2026.1 (`mkl_gnu_thread`):

| `max_active_levels` | `MKL_DYNAMIC` | query inside region | `mkl_get_max_threads()` inside | wall for both batches | CPU/wall | what ran |
|---|---|---|---|---|---|---|
| 1 | true (default) | 1 x n*V | 1 | 69 ms | 1.9 | 2 outer threads, MKL serial inside each |
| 1 | false | 4 x n*V | 4 | 69 ms | 1.9 | same: nesting is off, MKL's inner region is inactive |
| 2 | true (default) | 1 x n*V | 1 | 69 ms | 1.9 | 2 outer threads, MKL serial inside each |
| 2 | false | 4 x n*V | 4 | 40 ms | 3.2-3.6 | nested: each outer thread's call got a 4-thread team |

- **Default behavior is serial inside a region.** With `MKL_DYNAMIC` true (the
  default), MKL detects `omp_in_parallel()` and runs the call on one thread
  regardless of `max_active_levels`; the query drops to `1 x n*V` accordingly.
  That is what the sequential layer gives, so for a per-group OpenMP loop
  (one MKL call per group) linking the threaded layer changes nothing.
- **Nesting needs both switches, and the environment alone suffices.** No
  runtime API calls are required: `OMP_NUM_THREADS=2,4` with
  `OMP_MAX_ACTIVE_LEVELS=2` (or `OMP_NESTED=true`) *and* `MKL_DYNAMIC=false`
  produced the nested result above (CPU/wall 3.5-3.8). The OpenMP variables
  alone did not: with `MKL_DYNAMIC` at its default MKL stayed serial inside
  even with `max_active_levels` 2 or 255. `mkl_set_dynamic(0)` and
  `omp_set_max_active_levels()` are the API equivalents.
- **The query follows `mkl_get_max_threads()`, not what will actually run.**
  With `MKL_DYNAMIC` false and nesting disabled MKL reports 4 threads and a
  `4 x n*V` query but runs serially: oversized, harmless. The dangerous
  direction is the default one: a buffer queried *inside* a region
  (`1 x n*V`) and reused *outside* it at 4 threads overruns by three slices.
  Query where you call.

### 3b. Selecting the threaded layer

With CMake, `-DBLA_VENDOR=Intel10_64lp` asks FindBLAS for the threaded layer.
On a C/C++-only project FindBLAS 3.28 pairs `mkl_intel_thread` with `iomp5`
(it picks `mkl_gnu_thread` + `gomp` only when a GNU Fortran compiler is
enabled); if `libiomp5` is not on the library path the link still succeeds
and the binary segfaults at startup. Any threaded build of this repo must
settle on one OpenMP runtime for cqr's launcher and MKL alike. With Intel's
compilers that is the `-qmkl=parallel` / `-qmkl=sequential` flag (per Intel's
compiler documentation; not verified here, no Intel compiler is installed).

## 4. `geqrf` compared with cqr's

What is observable: MKL's query is `n * V` per thread and a real call writes
`(n-1) * V` of it, which is the size of a `w = v^T C` row vector over the
trailing columns, one lane per slot -- consistent with a `dlarf`-style update
that forms `w` explicitly before the rank-one update. The algorithm inside MKL
is not visible; that is an inference from the scratch size. cqr's `larf`
fuses the dot products and the rank-one update over four columns at a time,
one pass, no scratch. Measured with `bench_geqrf_compact` (square, pre-packed,
sequential MKL driven one group per call), cqr/MKL throughput:

| n | 1 thread, 512 matrices | 4 threads, 128 matrices |
|---|---|---|
| 8 | 1.09x | 0.80x |
| 16 | 1.41x | 1.30x |
| 32 | 1.60x | 1.56x |
| 64 | 1.24x | 1.30x |
| 128 | 1.25x | 1.20x |
| 256 | 1.49x | 1.46x |
| 500 | 1.45x | 1.48x |

- **The n=8 row is biased against MKL by the harness, not by MKL.** Driving
  MKL one group per call costs 13% at n=8 single-threaded against one
  whole-batch call (57 us vs 51 us for 512 matrices), 2-4% at n=16-32, and
  nothing at n=64. At four threads the harness additionally forks a team over
  groups of ~2700 flops each, which cqr's launcher would refuse to do (its
  `parallel_min_flops` gate); MKL's loop in the benchmark has no such gate.
  Whole-batch MKL calls would remove both effects.
- Relative error against per-matrix LAPACK is identical for both.
- **Where both fall below blocked LAPACK depends on the machine and on V.** On
  this box (1 MiB L2 per core) with V=8 doubles the crossover is near
  n~120-170, where one group (`n*n*V*8` bytes: 0.25 MiB at n=64, 0.88 MiB at
  n=120, 1.85 MiB at n=170) outgrows L2 and the unblocked sweep becomes
  bandwidth-bound; a smaller L2 or a wider V moves the crossover down, a
  narrower V or a larger cache moves it up. Blocking (compact-WY) is the next
  lever for either kernel.

## 5. How these were measured (for re-checking on another MKL)

- **Null-pointer and overrun tests**: one process per test, since the outcome is
  a segfault. Compile with `-lmkl_intel_lp64 -lmkl_sequential -lmkl_core`
  (add `-Wl,--no-as-needed` and `-lmkl_gnu_thread -lgomp` for the threaded
  layer). For a oneMKL from Intel's apt repository (`intel-oneapi-mkl-devel`,
  key and source line at apt.repos.intel.com/oneapi), use
  `-I/opt/intel/oneapi/mkl/latest/include -L/opt/intel/oneapi/mkl/latest/lib`
  and put that `lib` on `LD_LIBRARY_PATH` at run time.
- **Overrun detection**: `mmap` two pages, `mprotect` the second `PROT_NONE`,
  and place the `work` buffer so it ends exactly at the page boundary. Any write
  past `lwork` elements faults immediately instead of corrupting the heap.
- **Scratch use**: fill `work` with a sentinel (e.g. 1234.5) before the call and
  count changed entries after.
- **Internal allocation**: `mkl_peak_mem_usage(MKL_PEAK_MEM_ENABLE)`, then
  `MKL_PEAK_MEM_RESET` before and `MKL_PEAK_MEM` after the call.
- **Thread dependence**: `mkl_set_num_threads(t)` immediately before each
  `lwork = -1` query; print `mkl_get_max_threads()` alongside.
- **Nesting**: call from inside a 2-thread region and compare
  `CLOCK_PROCESS_CPUTIME_ID` to wall time across the region; ~2 means the outer
  threads alone worked, ~4 means MKL spawned teams underneath.
- Use `setvbuf(stdout, NULL, _IONBF, 0)` in any threaded-layer probe, or the
  exit-time crash swallows the buffered output.
