/* bench_portable_util.hpp
 *
 * The MKL-free core of the benchmark harness: abort-on-failure checks,
 * pack-aligned std::vector storage, MatrixPool (the batch of dense matrices
 * every benchmark measures on), best-of-N timing, and the OpenMP thread
 * count. bench_util.hpp layers the MKL-specific pieces on top (compact-format
 * lookups, PackedPool, the shared command line) for the MKL benchmarks;
 * bench_geqrf_armpl, which has no MKL to link, includes this header alone.
 *
 * Assisted-by: Claude
 */

#ifndef CQR_BENCH_PORTABLE_UTIL_HPP
#define CQR_BENCH_PORTABLE_UTIL_HPP

#include "cqr_matrix_batch.hpp"
#include "cqr_matrix_view.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cqr::bench {

using cqr::detail::mat_view; /* dense strided view (cqr_matrix_view.hpp) */
using cqr::detail::MatrixView;

/* Denominator floor for relative errors: the same divide-by-zero guard, value
 * and rationale as the test suites' norm_floor (tests/test_compact_util.hpp);
 * the two harness trees share no header, hence the twin definition. */
constexpr double norm_floor = 1e-300;

/* Report and abort on the spot if cond is false. */
inline void check(bool cond, const char *what)
{
    if (!cond) {
        std::printf("FAILED: %s\n", what);
        std::exit(1);
    }
}

/* std::vector storage aligned to the compact pack width (64 B covers every
 * format), so a dense pool and its LAPACK working copy start pack-aligned like
 * the compact buffers -- no cache-line splits in the packing reads or the
 * per-matrix LAPACK path. A stateless allocator: allocator_traits defaults the
 * rest; aligned_alloc needs the size rounded up to the alignment. */
template <typename T> struct aligned_allocator {
    using value_type = T;
    T *allocate(std::size_t n)
    {
        void *p = std::aligned_alloc(64, (n * sizeof(T) + 63) & ~std::size_t(63));
        if (!p) throw std::bad_alloc();
        return static_cast<T *>(p);
    }
    void deallocate(T *p, std::size_t) noexcept { std::free(p); }
    bool operator==(const aligned_allocator &) const noexcept { return true; }
    bool operator!=(const aligned_allocator &) const noexcept { return false; }
};
template <typename T> using aligned_vector = std::vector<T, aligned_allocator<T>>;

/* The batch every benchmark measures on: MatrixBatch (src/cqr_matrix_batch.hpp,
 * shared with the test suites), allocated pack-aligned so a dense pool and its
 * LAPACK working copies start aligned like the compact buffers. The benchmarks
 * differ only in what they put in their matrices (diagonally dominant, SPD,
 * symmetric indefinite, ...), so the fill stays with each of them; a
 * right-hand-side block is the same thing with cols = nrhs, so the benchmarks
 * that solve keep two of these. */
using MatrixPool = cqr::detail::MatrixBatch<double, aligned_allocator<double>>;

/* Best (minimum) wall time over `reps` timed passes, in seconds. `reset` runs
 * untimed before every pass (e.g. to restore input the timed work destroys);
 * only `timed` is clocked, after one untimed warm-up. */
template <typename Reset, typename Timed>
double best_time(int reps, Reset &&reset, Timed &&timed)
{
    using clk = std::chrono::steady_clock;
    reset();
    timed();
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < reps; ++r) {
        reset();
        auto t0 = clk::now();
        timed();
        best = std::min(best, std::chrono::duration<double>(clk::now() - t0).count());
    }
    return best;
}

/* Number of OpenMP threads the outer loops will run on (1 without OpenMP). */
inline int omp_threads()
{
    int nthreads = 1;
#ifdef _OPENMP
#pragma omp parallel
#pragma omp single
    nthreads = omp_get_num_threads();
#endif
    return nthreads;
}

} /* namespace cqr::bench */

#endif /* CQR_BENCH_PORTABLE_UTIL_HPP */
