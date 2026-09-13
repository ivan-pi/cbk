/* bench_util.hpp
 *
 * The harness shared by the benchmark programs: abort-on-failure checks, the
 * MKL compact-format lookups, pack-aligned std::vector storage, MatrixPool (the
 * batch of dense matrices every benchmark measures on) with PackedPool (its
 * pristine compact image, restored before every timed pass) and PackedSystems
 * (a solve benchmark's matrix + RHS pair of those), the symmetric pool fill
 * and the known-solution right-hand sides / forward error the Cholesky and
 * LDL^T benchmarks share, the Cholesky flop count, best-of-N timing,
 * the OpenMP thread count, and the factorization / solve benchmarks' command
 * line (--size-sweep, --simdlen, --nrhs, [nmat] [reps]). Needs the MKL
 * headers, and PackedPool calls mkl_malloc / mkl_dgepack_compact, so programs
 * using it link MKL (all benchmarks do).
 *
 * Assisted-by: Claude:claude-opus-4-8 Claude Claude:claude-fable-5
 */

#ifndef CBK_BENCH_UTIL_HPP
#define CBK_BENCH_UTIL_HPP

#include "cbk_compat.h"
#include "cbk_mkl_alloc.h" /* mkl_alloc_bytes / mkl_buffer, for PackedPool */
#include "cbk_matrix_batch.hpp"
#include "cbk_matrix_view.hpp"

#include <mkl.h>         /* cblas_dgemm, for the known-solution RHS */
#include <mkl_compact.h> /* mkl_dget_size_compact / mkl_dgepack_compact */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <random>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cbk::bench {

using cbk::detail::compact_format_name; /* format -> "SSE"/"AVX"/"AVX512" */
using cbk::detail::format_for_vlen;     /* interleave width -> pack format */
using cbk::detail::mat_view;            /* dense strided view (cbk_matrix_view.hpp) */
using cbk::detail::MatrixView;
using cbk::detail::vlen_for_format; /* pack format -> interleave width */

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

/* The batch every benchmark measures on: MatrixBatch (src/cbk_matrix_batch.hpp,
 * shared with the test suites), allocated pack-aligned so a dense pool and its
 * LAPACK working copies start aligned like the compact buffers. The benchmarks
 * differ in what they put in their matrices, so a fill unique to one benchmark
 * stays with it (the QR pools); the symmetric diagonally dominant fill the
 * Cholesky and LDL^T benchmarks share is fill_sym_dd below. A right-hand-side
 * block is the same thing with cols = nrhs, so the benchmarks that solve keep
 * two of these (fill_known_rhs pairs the RHS with forward_error). */
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

/* B := A X for the known solution X(:,j) = j + 1 -- the right-hand sides the
 * solve benchmarks recover, measured against that X by forward_error below. */
inline void fill_known_rhs(const MatrixPool &A, MatrixPool &B)
{
    const int n = A.rows(), nrhs = B.cols();
    std::vector<double> xs((size_t)n * nrhs);
    const auto X = mat_view(xs.data(), n, nrhs);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i)
            X(i, j) = j + 1;
    for (int v = 0; v < A.count(); ++v) {
        const auto Av = A.view(v);
        const auto Bv = B.view(v);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nrhs, n, 1.0, Av.data,
                    Av.ld(), X.data, X.ld(), 0.0, Bv.data, Bv.ld());
    }
}

/* Worst relative forward error max|X_v - X| / max|X| over the pool, X_v the
 * solution in a dense pool-layout buffer (nmat * n*nrhs), X the known solution
 * of fill_known_rhs. */
inline double forward_error(const double *x, int n, int nrhs, int nmat)
{
    double worst = 0;
    for (int v = 0; v < nmat; ++v) {
        const auto X = mat_view(x + (size_t)v * n * nrhs, n, nrhs);
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i)
                worst = std::max(worst, std::abs(X(i, j) - (j + 1)));
    }
    return worst / nrhs; /* max|X| = nrhs */
}

/* Standard LAPACK ?potrf flop count in GFLOP: n^3/3 + n^2/2 + n/6 (adds +
 * mults, the classic LAWN 41 count; the n square roots are not counted, as in
 * LAPACK's own timing). The solve benchmarks add their sweeps' flops on top. */
inline double chol_gflop(int n)
{
    const double dn = n;
    return (dn * dn * dn / 3.0 + dn * dn / 2.0 + dn / 6.0) * 1e-9;
}

/* A pool packed column-major into a compact buffer it owns: the pristine bytes
 * an in-place compact routine's working copy is restored from before every
 * timed pass (the pack itself stays untimed). */
struct PackedPool {
    MKL_INT bytes; /* mkl_dget_size_compact reports bytes */
    cbk::detail::mkl_buffer<double> p;

    PackedPool(const MatrixPool &P, MKL_COMPACT_PACK fmt)
        : bytes(mkl_dget_size_compact(P.rows(), P.cols(), fmt, P.count())),
          p(cbk::detail::mkl_alloc_bytes<double>(bytes))
    {
        auto ptrs = P.base_ptrs();
        mkl_dgepack_compact(MKL_COL_MAJOR, P.rows(), P.cols(), ptrs.data(), P.rows(),
                            p.get(), P.rows(), fmt, P.count());
    }

    /* An uninitialized working buffer of the same size. */
    cbk::detail::mkl_buffer<double> work() const
    {
        return cbk::detail::mkl_alloc_bytes<double>(bytes);
    }
    void restore_into(double *dst) const { std::memcpy(dst, p.get(), bytes); }
};

/* A solve benchmark's pair of pristine compact images -- the matrices and
 * their right-hand sides -- restored together before every timed pass. */
struct PackedSystems {
    PackedPool a, b;

    PackedSystems(const MatrixPool &A, const MatrixPool &B, MKL_COMPACT_PACK fmt)
        : a(A, fmt), b(B, fmt)
    {
    }

    void restore_into(double *ap, double *bp) const
    {
        a.restore_into(ap);
        b.restore_into(bp);
    }
};

/* Unpack a compact solution batch (n x nrhs per matrix, column-major, ld = n)
 * and measure it against fill_known_rhs's X: the correctness gate of a compact
 * solve path, read from the buffer its last timed pass left behind. */
inline double unpacked_forward_error(const double *bp, int n, int nrhs, int nmat,
                                     MKL_COMPACT_PACK fmt)
{
    std::vector<double> X((size_t)nmat * n * nrhs);
    std::vector<double *> Xp(nmat);
    for (int v = 0; v < nmat; ++v)
        Xp[v] = X.data() + (size_t)v * n * nrhs;
    mkl_dgeunpack_compact(MKL_COL_MAJOR, n, nrhs, Xp.data(), n, bp, n, fmt, nmat);
    return forward_error(X.data(), n, nrhs, nmat);
}

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
 * [reps], plus --size-sweep=nmin:nmax[:stride] (cbk-only scan), --simdlen=2|4|8
 * (force the interleave width instead of the host default) and --nrhs=k (right-
 * hand sides; the factorization benchmarks ignore it). The constructor parses
 * and validates and resolves the pack format; hold the object const. */
struct CmdArgs {
    int nmat = 512;
    int reps = 3;
    int nrhs = 1;
    bool sweep = false;
    int sweep_min = 0, sweep_max = 0, sweep_step = 1;
    MKL_COMPACT_PACK fmt; /* the host's widest, or the --simdlen one */
    int V;                /* its interleave width for double */

    CmdArgs(int argc, char **argv, const char *prog)
    {
        int simdlen = 0;
        std::vector<const char *> pos;
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
            else
                pos.push_back(argv[i]);
        }
        if (!pos.empty()) nmat = std::atoi(pos[0]);
        if (pos.size() > 1) reps = std::atoi(pos[1]);
        if (!(nmat > 0 && reps > 0 && nrhs > 0)) {
            std::printf("usage: %s [--size-sweep=nmin:nmax[:stride]] [--simdlen=2|4|8] "
                        "[--nrhs=k>0] [nmat>0] [reps>0]\n",
                        prog);
            std::exit(1);
        }
        check(!sweep || (sweep_min > 0 && sweep_max >= sweep_min && sweep_step > 0),
              "usage: --size-sweep needs 0 < nmin <= nmax and stride > 0");
        /* Double compact widths are 2/4/8 (SSE/AVX/AVX512); 16 is float's AVX512
         * width and has no double format. */
        check(simdlen == 0 || simdlen == 2 || simdlen == 4 || simdlen == 8,
              "usage: --simdlen must be 2, 4, or 8 (16 is float-only; this is double)");

        /* A wider interleave than the host's native SIMD cannot execute
         * (mkl_get_format_compact reports the widest the architecture supports). */
        const MKL_COMPACT_PACK native = mkl_get_format_compact();
        fmt = simdlen ? format_for_vlen<double>(simdlen) : native;
        V = vlen_for_format<double>(fmt);
        check(V > 0 && V <= vlen_for_format<double>(native),
              "requested --simdlen exceeds the host's native SIMD width");
    }
};

} /* namespace cbk::bench */

#endif /* CBK_BENCH_UTIL_HPP */
