# Batched QR for many small matrices: extending Intel MKL's Compact format -- portably

> **Status: draft.** Structure and prose are ready for review; every table
> marked *TBD* awaits numbers measured on the named machine, and the editor
> notes in blockquotes (**[note]**) are to be resolved and deleted before
> publication. Modeled on Arm's
> [interleave-batch announcement post](https://developer.arm.com/community/arm-community-blogs/b/tools-software-ides-blog/posts/new-interleave-batched-linear-algebra-functions-in-arm-pl),
> which introduced the same storage idea in Arm Performance Libraries.
>
> Assisted-by: Claude

Suppose your application needs to factor not one matrix but fifty thousand of
them -- each only 8x8 to 100x100. Block preconditioners, chemistry integrals,
per-element finite-element kernels, sliding-window least squares, Kalman
filters and factor graphs all look like this. Calling LAPACK fifty thousand
times is the natural first attempt, and it wastes most of the machine: at these
sizes the call overhead, the argument checking and the blocked algorithms'
bookkeeping dominate, and the SIMD units run far below capacity because a
single tiny matrix offers too little parallelism per instruction.

The fix, adopted independently by Intel MKL (the **Compact** functions), Arm
Performance Libraries (the **interleave-batch** functions) and the batched-BLAS
research community, is to change the data layout: interleave the matrices so
that one SIMD instruction advances *many* factorizations by one step each.
[**cqr**](https://github.com/ivan-pi/cqr) is a small open-source library that
extends MKL's Compact format where MKL stops -- it adds the missing apply-Q,
form-Q, symmetric-indefinite and one-call least-squares routines -- and, because
its kernels are written as portable GNU vector code rather than x86 intrinsics,
the same source runs on Arm. This post walks through the format, the library,
and measurements on three very different machines: an Intel Xeon (Sapphire
Rapids, AVX-512), an NVIDIA Grace (Neoverse V2), and an Apple M2 Pro.

## The interleaved (compact) format

Conventional batched interfaces (`?gemm_batch`, pointer arrays) still store
each matrix contiguously, so the vector units work *within* one small matrix
-- and run out of lanes. The compact format transposes the problem: pick an
interleave width `V`, and store element `(i,j)` of `V` consecutive matrices
contiguously. For `V = 2` and two 3x3 matrices `A1`, `A2` in column-major
order, memory holds

```
A1(1,1) A2(1,1)  A1(2,1) A2(2,1)  A1(3,1) A2(3,1)   <- column 1, both matrices
A1(1,2) A2(1,2)  A1(2,2) A2(2,2)  A1(3,2) A2(3,2)   <- column 2
A1(1,3) A2(1,3)  A1(2,3) A2(2,3)  A1(3,3) A2(3,3)   <- column 3
```

Each aligned block of `V` scalars is one SIMD register holding the *same*
element of `V` different matrices. A scalar algorithm -- Householder QR,
Cholesky, a triangular solve -- can then be transcribed line by line with
vectors in place of scalars, and every arithmetic instruction performs the
same step of `V` independent factorizations at once. There are no
data-dependent branches to diverge over, because the algorithms taken into
this format are chosen to be branch-free (no pivoting; more on that below).

MKL calls a set of `V` interleaved matrices a *pack*; `V` is chosen to match
the SIMD register: for FP64, 2 (SSE), 4 (AVX), 8 (AVX-512), and twice that for
FP32. A batch of `nm` matrices becomes `ceil(nm/V)` groups, the last one padded
with identity matrices so the kernels run unmasked. Arm PL's interleave-batch
format is the same idea with explicit strides (`ninter` plays the role of `V`);
MKL fixes the strides and provides pack/unpack routines
(`mkl_?gepack_compact` / `mkl_?geunpack_compact`) to convert from arrays of
conventional column- or row-major matrices.

The packing traffic is not free, so the format pays when the batch is packed
once and used across several operations -- factor, then apply Q, then solve --
or when it can live in compact form for the whole computation.

## What MKL ships, and what cqr adds

MKL's Compact extension provides the BLAS side (`gemm`, `trsm`) and a few
factorizations (`geqrf`, `potrf`, unpivoted LU `getrfnp`, inversion
`getrinp`). But it ships **no** compact `ormqr` -- so after
`mkl_?geqrf_compact` there is no supported way to apply `Q^T` to a right-hand
side, which is the very next step of every QR solve -- no `orgqr`, no
symmetric-indefinite factorization, and no one-call least-squares driver.

cqr fills those gaps behind two API surfaces: an MKL-style one
(`cqr_mkl_*_compact`, taking the same `MKL_COMPACT_PACK` format tags, mixing
freely with MKL's own compact routines) and a portable C API with an explicit
interleave width `V` and no MKL dependency at all.

| Routine | What it does | MKL equivalent |
|---------|--------------|----------------|
| `cqr_mkl_?geqrf_compact` | QR factorization | `mkl_?geqrf_compact` (cqr's is an open alternative) |
| `cqr_mkl_?ormqr_compact` | apply `Q` / `Q^T` from a compact QR | **none** |
| `cqr_mkl_?orgqr_compact` | form the explicit (thin) `Q` | **none** |
| `cqr_mkl_?potrf_compact` | Cholesky (`A = L L^T` / `U^T U`) | `mkl_?potrf_compact` (open alternative) |
| `cqr_mkl_?trsm_compact` | batched triangular solve | `mkl_?trsm_compact` (open drop-in) |
| `cqr_mkl_?sytrfnp/?sytrsnp/?sysvnp_compact` | unpivoted LDL^T factor / solve / fused, symmetric *indefinite* | **none** (naming follows MKL's `getrfnp`) |
| `cqr_mkl_?gels_compact` | one-call least-squares / minimum-norm solve | **none** |

All routines come in FP64 and FP32, and together they run the whole
`AX = B` pipeline -- `geqrf -> ormqr -> trsm`, or the fused `gels` -- with no
MKL compute kernel, which is what makes the Arm measurements below possible.

The trade the format demands is the same one MKL and Arm PL make: **no
pivoting**. A data-dependent row swap would diverge across the lanes of a
register, so the compact `getrfnp` and cqr's `sytrfnp` factor without it, and
the QR routines skip LAPACK's overflow-safe reflector rescaling. That is the
right contract for the intended inputs -- masses of small, reasonably
conditioned matrices from a discretization or a filter bank -- and the wrong
one for a single ill-conditioned system, which still belongs to LAPACK.

## One portable kernel per routine

Where MKL's compact kernels are x86 binaries dispatched at runtime, cqr's are
one C++ source per routine built on GNU vector types:

```c++
template <typename T, int V> struct pack {
    using type __attribute__((vector_size(V * sizeof(T)),
                              aligned(alignof(T)), may_alias)) = T;
};
```

`pack<double,8>::type` is a 512-bit vector of 8 doubles; `+`, `*`, broadcasts
and lane-wise `sqrt` on it compile to plain vector instructions for whatever
ISA the compiler targets. The same kernel source lowers to SSE, AVX or AVX-512
on x86 and to NEON or SVE on Arm -- selected only by the compiler flags
(`-march=native`, `-mcpu=neoverse-v2`, ...), with no intrinsics and no
per-ISA code paths.

A useful consequence: the interleave width `V` does not need to match the
hardware register. On a 128-bit NEON machine, `V = 8` doubles is a legal GNU
vector that the compiler lowers to a short unrolled burst of four independent
`fmla v*.2d` chains -- exactly the instruction-level parallelism a wide Arm
core with four NEON pipes wants to see. So the format constant and the
register width become independent tuning knobs, and the benchmarks below sweep
`V` on the Arm machines rather than assume `V = 2`.

Above the kernels, every routine threads its loop over groups with one OpenMP
`parallel for` (at most one thread per group, gated by a minimum-flops
heuristic), and stays serial when called from inside the application's own
parallel region -- the batch loop composes instead of oversubscribing.

## A worked example: solving 10,000 systems by QR

The complete program is
[`examples/solve_qr_compact.cpp`](solve_qr_compact.cpp); the skeleton is
short. Pack once:

```c
MKL_COMPACT_PACK fmt = mkl_get_format_compact();     /* widest host format */
MKL_INT bytes = mkl_dget_size_compact(n, n, fmt, nm);
double *ap = mkl_malloc(bytes, 64);                  /* interleaved A's    */
/* ... taup, bp likewise ... */
mkl_dgepack_compact(MKL_COL_MAJOR, n, n, Aptrs, n, ap, n, fmt, nm);
mkl_dgepack_compact(MKL_COL_MAJOR, n, nrhs, Bptrs, n, bp, n, fmt, nm);
```

then run the three-step solve on the interleaved buffers -- MKL's
factorization, cqr's apply-Q (the routine MKL omits), cqr's triangular solve:

```c
mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap, n, taup,
                   work, lwork, info, fmt, nm);          /* A = QR      */
cqr_mkl_dormqr_compact(MKL_COL_MAJOR, 'L', 'T', n, nrhs, n, ap, n, taup,
                       bp, n, work1, 1, info, fmt, nm);  /* B <- Q^T B  */
cqr_mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS,
                      MKL_NONUNIT, n, nrhs, 1.0, ap, n,
                      bp, n, fmt, nm);                   /* R X = Q^T B */
mkl_dgeunpack_compact(MKL_COL_MAJOR, n, nrhs, Xptrs, n, bp, n, fmt, nm);
```

or as a single call, `cqr_mkl_dgels_compact`, which fuses all three steps per
group of `V` matrices while they are cache-resident (and handles over- and
underdetermined shapes, like LAPACK's `?gels`). One packing amortizes over the
whole pipeline, and the example cross-checks all paths against per-matrix
`LAPACKE_dgels` and a known exact solution.

On a machine without MKL the same solve runs through the portable C API
(`dgeqrf_compact(..., V, ...)` and friends from `cqr_compact.h`), with `V`
passed explicitly and the packing done by the caller -- the layout is fully
documented, and packing is a reshape any array library can express.

## Measurements

The repository carries four throughput benchmarks
([`examples/BENCHMARKS.md`](BENCHMARKS.md)); the results below use
`bench_geqrf_compact` (QR factorization), `bench_potrf_compact` (Cholesky) and
`bench_qr_compact` (the end-to-end QR solve). Methodology, common to all
machines:

* Pools of square FP64 matrices, orders ~10-260, deliberately including sizes
  that are *not* multiples of any SIMD width, so remainder handling stays in
  the picture. The pool is packed once; only the compute is timed, with the
  destroyed input restored untimed between repetitions.
* Every path gets the same outer parallelism: the cqr paths hand the whole
  pool to one call and let the library thread over groups; the MKL-compact and
  per-matrix LAPACK baselines run from an OpenMP loop with the same thread
  count.
* Each benchmark carries a correctness gate against per-matrix LAPACK, so a
  fast wrong kernel cannot report a result.
* Built with host-tuned flags (`-O3 -march=native` or the Arm equivalents) so
  the portable kernels emit the machine's full vector width -- MKL's runtime
  dispatch does the same, and anything less is an unfair matchup.

> **[note]** Numbers in the three subsections below are placeholders. The
> Sapphire Rapids tables come straight from the existing harness; the two Arm
> machines need the portable-benchmark driver described in their subsections
> before their tables can be filled.

### Intel Xeon (Sapphire Rapids, AVX-512)

Sapphire Rapids is the home game: FP64 `V = 8` fills a 512-bit register
exactly, MKL's own compact kernels are available as the yardstick, and the
Golden Cove cores execute two 512-bit FMAs per cycle.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"
cmake --build build -j
./build/bench_geqrf_compact   # QR factorization: cqr vs mkl_dgeqrf_compact vs LAPACKE_dgeqrf
./build/bench_potrf_compact   # Cholesky:         cqr vs mkl_dpotrf_compact vs LAPACKE_dpotrf
./build/bench_qr_compact      # end-to-end solve, incl. cqr_mkl_dgels_compact vs LAPACKE_dgels
```

*QR factorization (`dgeqrf`), FP64, TBD matrices, TBD threads:*

| n | cqr (GF/s) | MKL compact (GF/s) | per-matrix LAPACK (GF/s) | cqr vs LAPACK |
|---|-----------|--------------------|--------------------------|---------------|
| *TBD* | | | | |

> **[note]** Fill from `bench_geqrf_compact` on the SPR box; repeat the table
> (or overlay a plot) for `bench_potrf_compact` and the `gels`-vs-`dgels`
> column of `bench_qr_compact`. Pin threads (`OMP_PLACES=cores
> OMP_PROC_BIND=close`), note the SKU, core count used, and MKL version.

For calibration, on a smaller 4-core AVX-512 machine the harness has already
measured: the one-call `cqr_mkl_dgels_compact` beat per-matrix `LAPACKE_dgels`
by **3.2x** in geometric mean over orders 10-100 (8.5x at `n = 10`, tapering
to 1.9x at `n = 100`), and the fused symmetric-indefinite solve
(`cqr_mkl_dsysvnp_compact` vs per-matrix `LAPACKE_dsysv`) by 7-9x at orders
8-32, falling behind only past order ~300 -- where LAPACK's blocked, pivoted
algorithms are the better tool and the compact format was never the right
choice anyway. That crossover is the honest boundary of the approach on every
machine, and the tables here should show it rather than trim it.

### NVIDIA Grace (Neoverse V2)

Grace changes two things at once: the ISA (NEON/SVE2 with 128-bit vectors --
`V = 2` doubles per register, but four SIMD pipes per core) and the software
stack (no MKL on Arm). The second point is what the portable design is for:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCQR_WITH_MKL=OFF \
      -DCMAKE_CXX_FLAGS="-O3 -mcpu=neoverse-v2"
cmake --build build -j && ctest --test-dir build   # portable suites, no BLAS needed
```

builds and validates the full kernel set against the library's scalar
references, with no vendor library in sight.

> **[note]** The current benchmark drivers link MKL for pack/unpack and the
> LAPACK baselines, so they do not build under `-DCQR_WITH_MKL=OFF`; the
> Grace and M2 Pro numbers need the planned portable benchmark driver (the
> pack/unpack helpers already exist MKL-free in `tests/test_compact_util.hpp`)
> with the per-matrix baseline taken from the platform LAPACK -- Arm
> Performance Libraries here, Accelerate on the Mac. Until that lands, this
> section's table stays empty. A comparison against Arm PL's own
> `armpl_dgeqrf_interleave_batch` (a different interleaved layout, same idea)
> would be the natural stretch goal.

Because a 128-bit register holds only two doubles, the interesting question on
Grace is the interleave width: `V = 2` maps one group to one register, while
`V = 4` or `V = 8` lower to unrolled bursts of independent `fmla v*.2d`
chains that can keep all four pipes busy at the cost of more live registers.

*QR factorization (`dgeqrf`), FP64, sweep over `V`:*

| n | V=2 (GF/s) | V=4 (GF/s) | V=8 (GF/s) | per-matrix Arm PL (GF/s) |
|---|-----------|-----------|-----------|--------------------------|
| *TBD* | | | | |

Grace's Neoverse V2 implements SVE2 at the same 128-bit vector length as NEON,
so fixed-width SVE codegen (`-msve-vector-bits=128`) is not expected to add
width here -- unlike, say, an A64FX, where `-msve-vector-bits=512` maps
`V = 8` doubles onto a single register. Worth one measurement to confirm, not
a headline.

### Apple M2 Pro

The M2 Pro is the same Arm story with a laptop twist: 128-bit NEON only (no
SVE), very wide performance cores (again four FP/NEON pipes), a big L1/L2 and
outstanding memory bandwidth for its class -- friendly territory for kernels
that stream small matrices. Toolchain notes: Apple clang needs `libomp`
(`brew install libomp`) for the threaded group loop, or build single-threaded
with `-DCQR_WITH_OPENMP=OFF` and parallelize outside.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCQR_WITH_MKL=OFF \
      -DCMAKE_CXX_FLAGS="-O3 -mcpu=native"
cmake --build build -j && ctest --test-dir build
```

*QR factorization (`dgeqrf`), FP64, sweep over `V`, per-matrix baseline
Accelerate (`LAPACKE_dgeqrf` via the new LAPACK interface):*

| n | V=2 (GF/s) | V=4 (GF/s) | V=8 (GF/s) | per-matrix Accelerate (GF/s) |
|---|-----------|-----------|-----------|-------------------------------|
| *TBD* | | | | |

> **[note]** Same portable-driver prerequisite as Grace. On macOS, pin to the
> performance cores as far as QoS allows and report P-core count; Accelerate's
> LAPACK is the honest per-matrix baseline (its AMX-backed GEMM does not help
> per-matrix `dgeqrf` at these sizes, but say which LAPACK the baseline
> linked).

### What to expect across all three

Two shapes recur on every machine, and are worth looking for in the tables:

* **The small-size regime is where batching wins.** Below order ~100 the
  compact kernels' advantage over per-matrix LAPACK is largest -- there is no
  call overhead, no blocking bookkeeping, and the vector units are full. The
  advantage shrinks as `n` grows and LAPACK's cache-blocked algorithms take
  over; the crossover (order 200-400 in the x86 measurements so far) is the
  boundary of the tool.
* **The unblocked factorizations are bandwidth-bound.** Each Householder
  reflector streams the trailing submatrix, so per-core GF/s goes flat once
  the working set leaves cache (~10 GF/s per AVX-512 core, flat from `n = 40`
  to 120, in the x86 measurements). That makes the Grace and M2 Pro results
  more interesting than a peak-flops comparison would suggest: both are
  strong-bandwidth designs, and the compact format's streaming access is
  exactly the pattern they reward.

## Closing thoughts

The interleaved-batch idea now exists in every major vendor library -- MKL's
Compact functions, Arm PL's interleave-batch functions -- and in portable form
in projects like Kokkos Kernels and this one. The format is the easy part; the
gaps are in coverage, and they bite mid-pipeline: MKL gives you a batched QR
but no way to apply `Q`, Arm PL (as of its announcement) gave the QR path but
in its own layout, and neither offers a compact symmetric-indefinite solve.
cqr's contribution is to complete one of these ecosystems -- MKL's -- with open,
portable kernels that keep working when the vendor library is not there at
all, which is how an "MKL extension" ends up with benchmark sections for two
Arm machines.

If your workload factors thousands of small matrices, the recipe is: pack
once, keep the whole pipeline in the interleaved format, and measure the
crossover size on *your* machine -- the benchmarks here are built to be rerun
(`examples/BENCHMARKS.md`). Code, design documents (one per routine) and the
test suites are at [github.com/ivan-pi/cqr](https://github.com/ivan-pi/cqr).

## References

* Intel oneMKL, [Compact BLAS and LAPACK functions](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/compact-blas-and-lapack-functions.html).
* C. Armstrong, [*Introducing interleave-batched linear algebra functions in Arm PL*](https://developer.arm.com/community/arm-community-blogs/b/tools-software-ides-blog/posts/new-interleave-batched-linear-algebra-functions-in-arm-pl), Arm Community Blogs, 2021.
* [Batched BLAS (BBLAS)](https://icl.utk.edu/bblas/) -- the proposed standard interface.
* [Kokkos Kernels batched API](https://kokkos.org/kokkos-kernels/docs/API/batched-index.html) -- portable batched kernels.
* [batmat](https://github.com/tttapa/batmat) -- batched small-matrix linear algebra in the same interleaved spirit.
