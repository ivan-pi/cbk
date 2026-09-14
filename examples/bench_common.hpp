/* bench_common.hpp
 *
 * The MKL-free part of the benchmark harness: abort-on-failure checks,
 * pack-aligned std::vector storage, MatrixPool (the batch of dense matrices
 * every benchmark measures on), the symmetric pool fill and the known
 * solution / forward error the solve benchmarks share, the Cholesky flop
 * count, best-of-N timing, the OpenMP thread count, the default square size
 * list every benchmark but bench_qr_compact runs (bench_sizes), the host's
 * interleave width (host_simdlen, what MKL's mkl_get_format_compact reports,
 * found without MKL), PackedImage (a pool's pristine compact image, packed by
 * the library-side pack_compact, restored before every timed pass) and the
 * benchmarks' shared command line (BenchArgs: --size-sweep, --simdlen, --nrhs,
 * [nmat] [reps]).
 *
 * Needs only the library and its internal headers, so a benchmark whose
 * baseline is a plain BLAS/LAPACK builds on it without MKL
 * (bench_trsm_compact); bench_util.hpp adds the MKL-side pieces the other
 * benchmarks use (MKL's pack/unpack in mkl_malloc storage, the
 * MKL_COMPACT_PACK resolution of --simdlen, cblas right-hand sides).
 *
 * Assisted-by: Claude:claude-opus-4-8 Claude:claude-fable-5
 */

#ifndef CBK_BENCH_COMMON_HPP
#define CBK_BENCH_COMMON_HPP

#include "cbk_compact_pack.hpp"
#include "cbk_matrix_batch.hpp"
#include "cbk_matrix_view.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <new>
#include <random>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cbk::bench {

using cbk::detail::mat_view; /* dense strided view (cbk_matrix_view.hpp) */
using cbk::detail::MatrixView;
using cbk::detail::pack_compact; /* the library-side pack (cbk_compact_pack.hpp) */
using cbk::detail::unpack_compact;

/* Denominator floor for relative errors: the same divide-by-zero guard, value
 * and rationale as the test suites' norm_floor (tests/test_compact_util.hpp);
 * the two harness trees share no header, hence the twin definition. */
constexpr double norm_floor = 1e-300;

/* The square orders the factorization and solve benchmarks run by default: one
 * shared list, so the set moves in one place (bench_qr_compact keeps its own).
 * It mixes orders that are not multiples of the SIMD width V -- 30, 45, 60, 105,
 * 168, from 2-D/3-D RBF-FD stencils -- with the round powers, so the remainder
 * handling (the staircase SIMD effect) stays visible, and stops at 256: a longer
 * run buys little, the gains live below 128. Ascending, so back() is the largest.
 * --size-sweep, which is not capped, is the finer cbk-only scan. */
inline constexpr std::array bench_sizes = {8,  16, 24,  30,  32,  45,  48, 60,
                                           64, 96, 105, 128, 168, 170, 256};

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

/* The batch every benchmark measures on: MatrixBatch (src/cbk_matrix_batch.hpp,
 * shared with the test suites), allocated pack-aligned so a dense pool and its
 * LAPACK working copies start aligned like the compact buffers. The benchmarks
 * differ in what they put in their matrices, so a fill unique to one benchmark
 * stays with it (the QR pools, the triangular pool); the symmetric diagonally
 * dominant fill the Cholesky and LDL^T benchmarks share is fill_sym_dd below. A
 * right-hand-side block is the same thing with cols = nrhs, so the benchmarks
 * that solve keep two of these (fill_known_x pairs the RHS with forward_error). */
using MatrixPool = cbk::detail::MatrixBatch<double, aligned_allocator<double>>;

/* Fill a square pool with symmetric, strictly diagonally dominant matrices:
 * random off-diagonals in [-1,1] mirrored across the diagonal, and a diagonal
 * of magnitude 2n (row off-diagonal magnitudes sum to at most n-1 < 2n, so
 * every leading principal minor is nonsingular) -- O(n^2) to build, no O(n^3)
 * M^T M product. With `indefinite` false the diagonal is positive, so the
 * matrices are SPD and well conditioned; with it true the sign alternates,
 * making them genuinely indefinite (Cholesky would fail; LAPACK needs ?sysv,
 * not ?posv) while the unpivoted LDL^T stays safe with bounded element growth.
 * The full matrix is stored (both triangles) so a per-matrix LAPACK path and
 * the compact pack see identical symmetric input; each routine reads only the
 * triangle it is told to. */
inline void fill_sym_dd(MatrixPool &P, bool indefinite)
{
    const int n = P.rows();
    std::mt19937_64 rng(2025);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int v = 0; v < P.count(); ++v) {
        const auto A = P.view(v);
        for (int j = 0; j < n; ++j) {
            for (int i = j + 1; i < n; ++i) {
                double x = dist(rng);
                A(i, j) = x; /* lower */
                A(j, i) = x; /* mirror to upper (symmetric) */
            }
            A(j, j) = (indefinite && j % 2 ? -2.0 : 2.0) * n; /* dominant */
        }
    }
}

/* X(:,j) = j + 1, the known solution the solve benchmarks recover (their
 * right-hand sides are formed from it) and forward_error measures against. */
inline void fill_known_x(MatrixView<double> X)
{
    for (int j = 0; j < X.cols; ++j)
        for (int i = 0; i < X.rows; ++i)
            X(i, j) = j + 1;
}

/* Worst relative forward error max|X_v - X| / max|X| over the pool, X_v the
 * solution in a dense pool-layout buffer (nmat * rows*cols, column-major, ld =
 * rows), X the known solution of fill_known_x (so max|X| = cols). */
inline double forward_error(const double *x, int rows, int cols, int nmat)
{
    double worst = 0;
    for (int v = 0; v < nmat; ++v) {
        const auto X = mat_view(x + (size_t)v * rows * cols, rows, cols);
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                worst = std::max(worst, std::abs(X(i, j) - (j + 1)));
    }
    return worst / cols;
}

/* Standard LAPACK ?potrf flop count in GFLOP: n^3/3 + n^2/2 + n/6 (adds +
 * mults, the classic LAWN 41 count; the n square roots are not counted, as in
 * LAPACK's own timing). The solve benchmarks add their sweeps' flops on top. */
inline double chol_gflop(int n)
{
    const double dn = n;
    return (dn * dn * dn / 3.0 + dn * dn / 2.0 + dn / 6.0) * 1e-9;
}

/* The interleave width for double that maps onto one of the host's widest
 * vector registers -- the rule mkl_get_format_compact applies (AVX-512F: 8,
 * AVX: 4, else 2), asked of the CPU at run time through the compiler's cpuid
 * builtin, so the portable build needs no MKL to pick the same default as the
 * MKL benchmarks. Off x86 there is no such query; the width this build was
 * compiled for (compiled_simdlen) stands in. Any width runs on any host -- the
 * kernels lower a pack wider than the ISA to several instructions -- so this
 * is a default for --simdlen, not a limit; a narrower width is always valid. */
#if defined(__AVX512F__)
inline constexpr int compiled_simdlen = 8;
inline constexpr const char *compiled_isa = "AVX-512";
#elif defined(__AVX__)
inline constexpr int compiled_simdlen = 4;
inline constexpr const char *compiled_isa = "AVX";
#elif defined(__SSE2__) || defined(__x86_64__)
inline constexpr int compiled_simdlen = 2;
inline constexpr const char *compiled_isa = "SSE";
#else
inline constexpr int compiled_simdlen = 2;
inline constexpr const char *compiled_isa = "128-bit";
#endif

/* The ISA name of an interleave width for double, for the header lines. */
inline const char *simdlen_name(int V)
{
    return V == 8 ? "AVX-512" : V == 4 ? "AVX" : "SSE";
}

inline int host_simdlen()
{
#if (defined(__x86_64__) || defined(__i386__)) &&                                        \
    (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) return 8;
    if (__builtin_cpu_supports("avx")) return 4;
    return 2;
#else
    return compiled_simdlen;
#endif
}

/* A pool packed column-major into a pack-aligned compact buffer it owns, by
 * the library-side pack_compact (no MKL): the pristine bytes an in-place
 * compact routine's working copy is restored from before every timed pass
 * (the pack itself stays untimed). The MKL-packed twin in mkl_malloc storage
 * is bench_util.hpp's PackedPool. */
struct PackedImage {
    int V;
    std::size_t size; /* scalars */
    aligned_vector<double> p;

    PackedImage(const MatrixPool &P, int V_)
        : V(V_),
          size(cbk::detail::compact_size(P.count(), P.rows(), P.cols(), P.rows(), V_)),
          p(size)
    {
        pack_compact(P, p.data(), P.rows(), V);
    }

    /* A working buffer of the same size. */
    aligned_vector<double> work() const { return aligned_vector<double>(size); }
    void restore_into(double *dst) const
    {
        std::memcpy(dst, p.data(), size * sizeof(double));
    }
    /* Read a working buffer of this image's shape back into a dense pool. */
    void unpack_into(MatrixPool &P, const double *src) const
    {
        unpack_compact(P, src, P.rows(), V);
    }
};

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

/* Command line of the factorization and solve benchmarks: positional [nmat]
 * [reps], plus --size-sweep=nmin:nmax[:stride] (cbk-only scan, any nmax),
 * --simdlen=2|4|8 (force the interleave width instead of the host's) and
 * --nrhs=k (right-hand sides; the factorization benchmarks ignore it). The
 * constructor parses and validates; hold the object const. V is the resolved
 * interleave width, host_simdlen() unless --simdlen names one (the MKL
 * harness's CmdArgs re-resolves it against MKL's native format). A benchmark
 * with options of its own passes `extra`, called with each `--` argument the
 * shared set does not know and returning whether it took it, and the usage
 * text of those options (appended to the shared usage line). */
struct BenchArgs {
    int nmat = 512;
    int reps = 3;
    int nrhs = 1;
    bool sweep = false;
    int sweep_min = 0, sweep_max = 0, sweep_step = 1;
    int simdlen = 0; /* --simdlen as given, 0 when not */
    int V;           /* the interleave width the benchmark runs at */

    BenchArgs(int argc, char **argv, const char *prog,
              const std::function<bool(const char *)> &extra = {},
              const char *extra_usage = "")
    {
        std::vector<const char *> pos;
        bool ok = true;
        for (int i = 1; i < argc; ++i) {
            if (std::strncmp(argv[i], "--size-sweep=", 13) == 0) {
                int got = std::sscanf(argv[i] + 13, "%d:%d:%d", &sweep_min, &sweep_max,
                                      &sweep_step);
                check(got >= 2, "usage: --size-sweep=nmin:nmax[:stride]");
                if (got == 2) sweep_step = 1;
                sweep = true;
            }
            else if (std::strncmp(argv[i], "--simdlen=", 10) == 0)
                simdlen = std::atoi(argv[i] + 10);
            else if (std::strncmp(argv[i], "--nrhs=", 7) == 0)
                nrhs = std::atoi(argv[i] + 7);
            else if (std::strncmp(argv[i], "--", 2) == 0) {
                if (!(extra && extra(argv[i]))) {
                    std::printf("unknown option %s\n", argv[i]);
                    ok = false;
                }
            }
            else
                pos.push_back(argv[i]);
        }
        if (!pos.empty()) nmat = std::atoi(pos[0]);
        if (pos.size() > 1) reps = std::atoi(pos[1]);
        if (!(ok && nmat > 0 && reps > 0 && nrhs > 0)) {
            std::printf("usage: %s [--size-sweep=nmin:nmax[:stride]] [--simdlen=2|4|8] "
                        "[--nrhs=k>0]%s [nmat>0] [reps>0]\n",
                        prog, extra_usage);
            std::exit(1);
        }
        check(!sweep || (sweep_min > 0 && sweep_max >= sweep_min && sweep_step > 0),
              "usage: --size-sweep needs 0 < nmin <= nmax and stride > 0");
        /* Double compact widths are 2/4/8 (SSE/AVX/AVX512); 16 is float's AVX512
         * width and has no double format. */
        check(simdlen == 0 || simdlen == 2 || simdlen == 4 || simdlen == 8,
              "usage: --simdlen must be 2, 4, or 8 (16 is float-only; this is double)");
        V = simdlen ? simdlen : host_simdlen();
    }
};

} /* namespace cbk::bench */

#endif /* CBK_BENCH_COMMON_HPP */
