# Intel MKL Compact routines: measured behavior

Empirical notes on how MKL's own compact routines treat their arguments, gathered
while settling the `info`/`work` contract of `cqr_mkl_ext.cpp` (PR #38). The
MKL documentation leaves most of this unstated ("info is reserved", "no error
checking"), so these were measured directly. Everything below is from small C
probes against the installed library; nothing is inferred from documentation.

Setup: Intel MKL 2020.4 (Debian `libmkl-dev` 2020.4.304), LP64 interface, on a
4-core AVX-512 Xeon, gcc 13. Routines probed: `mkl_?potrf_compact`,
`mkl_?getrfnp_compact`, `mkl_?geqrf_compact`, `mkl_?getrinp_compact`, plus
`mkl_?trsm_compact` for threading only.

## 1. Every pointer argument is mandatory

| Routine | `info` = NULL | `work` = NULL (query) | `work` = NULL (call) | `taup` = NULL |
|---|---|---|---|---|
| potrf | segfault | no `work` arg | no `work` arg | n/a |
| getrfnp | segfault | no `work` arg | no `work` arg | n/a |
| geqrf | segfault | segfault | segfault | segfault |
| getrinp | segfault | segfault | segfault | n/a |

`info` is written unconditionally (0 on success; the docs call it "reserved"),
exactly like LAPACK's INFO. `taup` always receives the reflector scalars; a
null `taup` is not "skip the copy". Consequence for cqr: the MKL-style wrappers
write `*info` and `work[0]` without null checks, matching MKL. A null pointer
fails loudly at the first write, which is the benign failure mode.

## 2. Workspace: size, use, and the missing `lwork` check

Only `geqrf` and `getrinp` take `work`/`lwork`. For both:

- **Query formula**: `lwork = n * V * mkl_get_max_threads()`. Independent of `m`
  and of the number of groups. `n * V` is the reference `dlarf`/`dgetri`
  per-reflector row vector, one lane per interleaved slot; one slice per thread
  (section 3). Under the sequential layer the multiplier is always 1.
- **`work` is the scratch, not a hint**: a real call writes `(n-1)*V` entries per
  thread (`geqrf`) or `(n-1)*V + 1` (`getrinp`). `mkl_peak_mem_usage` reports 0
  bytes allocated internally, so nothing is allocated behind the caller's back.
- **`lwork` is never checked**: with a one-element buffer and `lwork = 1`, both
  routines wrote their full `n*V` scratch and hit a guard page (see section 6).
  A too-small buffer is a silent overrun into whatever is adjacent: a
  neighbouring allocation, malloc metadata ("malloc(): unaligned tcache chunk"
  at a later `free`), or a stack frame. A generously oversized buffer is safe
  by accident; only the query makes the size a contract.
- **`work[0]` after a real call** holds leftover scratch, not the optimal
  `lwork` the documentation promises.

The same hazard exists in cqr's own `?gels_compact`: its `work` is the tau
buffer for the whole batch, `min(m,n) * V * ceil(nm/V)` scalars, and like every
compact routine it validates nothing. Size it from its own query. cqr's
`?geqrf`/`?ormqr` are the exception only because they need no scratch (their
`larf` keeps four columns' dot products in registers instead of forming the
full row vector), so their query honestly returns 1 and a real call never
touches `work`.

Rule that follows: **query each routine, size its buffer from that routine's
own answer, never reuse one routine's buffer for another.** Under a threaded MKL
layer add: **query and call under the same thread count** (section 3).

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

Consequences:

- **The workspace query scales with the *effective* thread count.** After
  `mkl_set_num_threads(t)`, `geqrf`/`getrinp` report `n * V * mkl_get_max_threads()`;
  asking for 8 or 16 threads on a 4-core machine clamps both to 4. A buffer
  queried at one thread and used at four overruns by three `n*V` slices (this
  crashed the first threaded probe). Under `mkl_sequential`,
  `mkl_set_num_threads` is accepted and ignored, `mkl_get_max_threads()` stays
  1, and the query stays `n * V`.
- **This repo links the sequential layer everywhere** (`BLA_VENDOR=Intel10_64lp_seq`
  in CMake and CI), deliberately: the benchmarks thread over groups with their
  own OpenMP loop so every path (cqr, MKL compact, per-matrix LAPACK) runs on
  the same caller-controlled thread count. That loop is a fair proxy: MKL's
  internally threaded whole-batch throughput at 4 threads matches the per-group
  loop's (143k vs 147k matrices/s at n=64, 1.12M vs 1.06M at n=32). It only
  costs MKL at n~8, where per-call overhead is comparable to the work.
- **`mkl_gnu_thread` + libgomp crashes at exit** on this MKL version: every
  probe linked that way segfaulted in `_dl_fini` after `main` returned, with
  results already correct (stdout must be unbuffered to see them). Under ctest
  that is a failed test. Switching the build to the threaded layer would need
  `libiomp5` (installed), which conflicts with the OpenMP runtime cqr's own
  launcher uses.

## 4. What MKL does not ship

There is no `mkl_?ormqr_compact`; cqr's is the only compact `ormqr`, so its
workspace contract (query returns 1) is cqr's alone to define.

## 5. Algorithm comparison, `geqrf`

MKL's compact `geqrf` is the unblocked `geqr2` shape with an explicit `w = v^T C`
row vector (hence the `n * V` scratch): two passes over the trailing block per
reflector. cqr's `larf` fuses the dot products and the rank-one update over four
columns at a time, one pass, no scratch. Measured with `bench_geqrf_compact`
(square, pre-packed, sequential MKL driven per group), cqr/MKL throughput:

| n | 1 thread, 512 matrices | 4 threads, 128 matrices |
|---|---|---|
| 8 | 1.09x | 0.80x (per-call overhead, see section 3) |
| 16 | 1.41x | 1.30x |
| 32 | 1.60x | 1.56x |
| 64 | 1.24x | 1.30x |
| 128 | 1.25x | 1.20x |
| 256 | 1.49x | 1.46x |
| 500 | 1.45x | 1.48x |

Relative error against per-matrix LAPACK is identical for both (the arithmetic
per element is the same; only loop order differs). Both are bandwidth-bound past
n~120 and fall below blocked LAPACK there; blocking (compact-WY) is the next
lever for either.

## 6. How these were measured (for re-checking on another MKL)

- **Null-pointer and overrun tests**: one process per test, since the outcome is
  a segfault. Compile with `-lmkl_intel_lp64 -lmkl_sequential -lmkl_core`
  (add `-Wl,--no-as-needed` and `-lmkl_gnu_thread -lgomp` for the threaded
  layer).
- **Overrun detection**: `mmap` two pages, `mprotect` the second `PROT_NONE`,
  and place the `work` buffer so it ends exactly at the page boundary. Any write
  past `lwork` elements faults immediately instead of corrupting the heap.
- **Scratch use**: fill `work` with a sentinel (e.g. 1234.5) before the call and
  count changed entries after.
- **Internal allocation**: `mkl_peak_mem_usage(MKL_PEAK_MEM_ENABLE)`, then
  `MKL_PEAK_MEM_RESET` before and `MKL_PEAK_MEM` after the call.
- **Thread dependence**: `mkl_set_num_threads(t)` immediately before each
  `lwork = -1` query; print `mkl_get_max_threads()` alongside.
- Use `setvbuf(stdout, NULL, _IONBF, 0)` in any threaded-layer probe, or the
  exit-time crash swallows the buffered output.
