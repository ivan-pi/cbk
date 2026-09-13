/* bench_geqrf_armpl.cpp
 *
 * Throughput benchmark of the QR *factorization* over pools of many small
 * matrices, comparing two implementations of the same LAPACK ?geqrf math:
 *
 *   cqr-compact  dgeqrf_compact   (this project's portable batched SIMD kernel)
 *   per-matrix   LAPACKE_dgeqrf   (Arm Performance Libraries, one matrix at a time)
 *
 * The AArch64 counterpart of bench_geqrf_compact (examples/BENCHMARKS.md):
 * ArmPL ships no compact-format kernels, so there is no batched yardstick and
 * the MKL pack/format helpers of bench_util.hpp are unavailable -- the pool
 * is packed here through the kernels' own BatchView, as the portable test
 * suites pack, and the comparison is the open compact kernel against the
 * conventional per-matrix LAPACK path. To measure the kernels rather than
 * data movement, the pool is packed once, up front; only the factorization is
 * timed, and the destroyed input is restored (untimed) before each pass. The
 * cqr path is one call on the whole pool -- the routine threads its own loop
 * over groups -- and the per-matrix path is an OpenMP loop of the same thread
 * count (ArmPL is linked serial; see cmake/FindArmPL.cmake). The
 * factorization is checked (untimed) against per-matrix LAPACK, so the
 * benchmark doubles as an integration test.
 *
 * Usage:  bench_geqrf_armpl [--simdlen=2|4|8] [nmat] [reps]
 *         (defaults: 512 matrices, 3 reps, and the interleave width of the
 *         host's native double vector: 8/4 under AVX-512/AVX, else 2 -- the
 *         128-bit NEON pair. The kernels run any width on any host, a wider
 *         one as unrolled 128-bit bursts, and on NEON/SVE no width is the
 *         obvious winner a priori: sweep --simdlen to see.)
 *
 * Build: -DCQR_WITH_ARMPL=ON; cmake/FindArmPL.cmake locates the library.
 * Nothing here is ArmPL-specific beyond the <armpl.h> include -- any LAPACKE
 * provider would serve as the per-matrix baseline.
 *
 * Assisted-by: Claude
 */

/* ArmPL ships the standard LAPACKE C interface as its own lapacke.h, next to
 * armpl.h; the one routine used here needs nothing armpl.h adds. */
#include <lapacke.h>

#include "bench_portable_util.hpp"
#include "cqr_compact.h"
#include "cqr_compact_common.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

namespace {

using namespace cqr::bench;

using cqr::detail::for_vlen;
using cqr::detail::group_stride;
using cqr::detail::make_const_view;
using cqr::detail::make_view;

/* The default interleave width: the host's native double vector -- what
 * bench_util.hpp asks MKL for on x86. Only a default for --simdlen, not a
 * capability probe: unlike MKL's kernels, ours run any width on any host. */
#if defined(__AVX512F__)
constexpr int default_simdlen = 8;
#elif defined(__AVX__)
constexpr int default_simdlen = 4;
#else
constexpr int default_simdlen = 2; /* SSE, and NEON's 128-bit double pair */
#endif

/* Standard LAPACK ?geqrf flop count (m >= n), in GFLOP. */
double geqrf_gflop(int m, int n)
{
    return (2.0 * m * n * (double)n - (2.0 / 3.0) * n * (double)n * n) * 1e-9;
}

/* A pool of `nmat` random m x n matrices, well conditioned (diagonal-boosted)
 * -- the same fill as bench_geqrf_compact. */
MatrixPool make_pool(int m, int n, int nmat)
{
    MatrixPool P(nmat, m, n);
    std::mt19937_64 rng(2025);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int v = 0; v < nmat; ++v) {
        const auto A = P.view(v);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                A(i, j) = dist(rng);
        for (int d = 0; d < std::min(m, n); ++d)
            A(d, d) += 2.0 * n;
    }
    return P;
}

/* Compact pack/unpack, column-major at ld = rows: the interleaved side
 * through the kernels' BatchView (for_vlen turns the runtime V into its
 * compile-time width), the dense side through the batch's views -- the
 * benchmark-tree twin of pack_compact / unpack_compact in
 * tests/test_compact_util.hpp (the two harness trees share no header).
 * Padded slots (idx >= count) carry the identity so kernels run the last
 * group unmasked; a tau batch (k x 1, zero pad) unpacks the same way. */
template <class Batch> void pack_pool(const Batch &Mk, double *p, int V)
{
    const int m = Mk.rows(), n = Mk.cols(), nm = Mk.count();
    const int ng = (nm + V - 1) / V;
    const std::size_t gstride = group_stride(false, m, m, n, V);
    const bool width_ok = for_vlen(V, [&](auto vw) {
        constexpr int VV = decltype(vw)::value;
        for (int g = 0; g < ng; ++g) {
            const auto P = make_view<double, VV>(p + (std::size_t)g * gstride, false, m);
            for (int v = 0; v < VV; ++v) {
                const int idx = g * VV + v;
                if (idx < nm) {
                    const auto M = Mk.view(idx);
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < m; ++i)
                            P(i, j)[v] = M(i, j);
                }
                else { /* padded slot */
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < m; ++i)
                            P(i, j)[v] = (i == j) ? 1.0 : 0.0;
                }
            }
        }
    });
    check(width_ok, "interleave width must be 2, 4, 8 or 16");
}

template <class Batch> void unpack_pool(Batch &Mk, const double *p, int V)
{
    const int m = Mk.rows(), n = Mk.cols(), nm = Mk.count();
    const int ng = (nm + V - 1) / V;
    const std::size_t gstride = group_stride(false, m, m, n, V);
    const bool width_ok = for_vlen(V, [&](auto vw) {
        constexpr int VV = decltype(vw)::value;
        for (int g = 0; g < ng; ++g) {
            const auto P =
                make_const_view<double, VV>(p + (std::size_t)g * gstride, false, m);
            for (int v = 0; v < VV; ++v) {
                const int idx = g * VV + v;
                if (idx >= nm) continue;
                const auto M = Mk.view(idx);
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i)
                        M(i, j) = P(i, j)[v];
            }
        }
    });
    check(width_ok, "interleave width must be 2, 4, 8 or 16");
}

/* A pool packed column-major into an aligned compact buffer it owns: the
 * pristine bytes the in-place factorization's working copy is restored from
 * before every timed pass -- the portable twin of bench_util.hpp's
 * PackedPool, whose sizing and packing are MKL calls. */
struct PackedPool {
    aligned_vector<double> p;

    PackedPool(const MatrixPool &P, int V)
        : p((std::size_t)((P.count() + V - 1) / V) *
            group_stride(false, P.rows(), P.rows(), P.cols(), V))
    {
        pack_pool(P, p.data(), V);
    }

    /* An uninitialized working buffer of the same size. */
    aligned_vector<double> work() const { return aligned_vector<double>(p.size()); }
    void restore_into(double *dst) const
    {
        std::memcpy(dst, p.data(), p.size() * sizeof(double));
    }
};

/* A compact tau buffer for nmat factorizations of k reflectors each. */
aligned_vector<double> tau_buffer(int nmat, int k, int V)
{
    return aligned_vector<double>((std::size_t)((nmat + V - 1) / V) * k * V);
}

/* Per-matrix ArmPL LAPACK factorization of a standard-layout pool copy in
 * place, an OpenMP loop over the matrices. */
void factor_unbatched(double *a, int m, int n, int nmat)
{
    const int k = std::min(m, n);
#pragma omp parallel
    {
        std::vector<double> tau(k);
#pragma omp for schedule(static)
        for (int v = 0; v < nmat; ++v)
            LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, a + (std::size_t)v * m * n, m,
                           tau.data());
    }
}

/* Relative factor error of the compact path vs per-matrix LAPACK: unpack the
 * compact (H, tau) and compare elementwise to a fresh LAPACKE_dgeqrf, scaled by
 * the matrix L1 norm. Untimed correctness gate. */
double factor_error(const MatrixPool &P, const PackedPool &pristine, int V)
{
    const int m = P.rows(), n = P.cols(), nmat = P.count(), k = std::min(m, n);
    const size_t sA = P.stride();

    /* factor a fresh working copy of the pristine pack; the one checked call
     * also vouches for the arguments the (unchecked) timed passes reuse */
    auto ap = pristine.work();
    pristine.restore_into(ap.data());
    auto tp = tau_buffer(nmat, k, V);
    check(dgeqrf_compact('C', m, n, ap.data(), m, tp.data(), V, nmat) == 0,
          "dgeqrf_compact accepts the benchmark arguments");

    MatrixPool H(nmat, m, n), tau(nmat, k, 1);
    unpack_pool(H, ap.data(), V);
    unpack_pool(tau, tp.data(), V);

    double worst = 0;
    std::vector<double> Href(sA), tref(k);
    for (int v = 0; v < nmat; ++v) {
        std::copy(P[v], P[v] + sA, Href.begin());
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, Href.data(), m, tref.data());
        double num = 0, den = 0;
        for (size_t i = 0; i < sA; ++i) {
            num = std::max(num, std::abs(H[v][i] - Href[i]));
            den = std::max(den, std::abs(Href[i]));
        }
        for (int i = 0; i < k; ++i)
            num = std::max(num, std::abs(tau[v][i] - tref[i]));
        worst = std::max(worst, num / std::max(den, norm_floor));
    }
    return worst;
}

} /* anonymous namespace */

int main(int argc, char **argv)
{
    int nmat = 512, reps = 3, V = default_simdlen;
    std::vector<const char *> pos;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--simdlen=", 10) == 0)
            V = std::atoi(argv[i] + 10);
        else
            pos.push_back(argv[i]);
    }
    if (!pos.empty()) nmat = std::atoi(pos[0]);
    if (pos.size() > 1) reps = std::atoi(pos[1]);
    /* 16 is float's width; this benchmark is double, like the MKL ones. */
    if (!(nmat > 0 && reps > 0) || !(V == 2 || V == 4 || V == 8)) {
        std::printf("usage: bench_geqrf_armpl [--simdlen=2|4|8] [nmat>0] [reps>0]\n");
        return 1;
    }
    const int nthreads = omp_threads();

    /* The size list of bench_geqrf_compact: sizes that are not multiples of
     * any interleave width mixed with round powers, so the SIMD remainder
     * handling stays visible, then a few larger sizes for the crossover. */
    constexpr std::array sizes = {8,  16,  24,  30,  32,  45,  48,  60, 64,
                                  96, 105, 128, 168, 170, 256, 384, 500};

    std::printf("QR factorization throughput: dgeqrf_compact (portable C API) vs "
                "per-matrix LAPACKE_dgeqrf (ArmPL)\n");
    std::printf("matrices=%d  reps=%d  simdlen=%d  OpenMP threads=%d  (square, "
                "pre-packed)\n\n",
                nmat, reps, V, nthreads);
    std::printf("   n | cqr GFLOP/s |   cqr mat/s | armpl mat/s | cqr/armpl | "
                "relerr(vs LAPACK)\n");
    std::printf("-----+-------------+-------------+-------------+-----------+"
                "------------------\n");

    double log_speed_vs_lapack = 0.0;
    for (int n : sizes) {
        const int m = n, k = n;
        const MatrixPool P = make_pool(m, n, nmat);

        /* pristine packed pool + the working copy the timed passes destroy */
        const PackedPool pristine(P, V);
        auto work_ap = pristine.work();
        auto taup = tau_buffer(nmat, k, V);

        auto restore = [&] { pristine.restore_into(work_ap.data()); };
        const double t_cqr = best_time(reps, restore, [&] {
            dgeqrf_compact('C', m, n, work_ap.data(), m, taup.data(), V, nmat);
        });

        aligned_vector<double> pool_work; /* standard-layout copy (aligned like pool) */
        const double t_lap = best_time(
            reps, [&] { pool_work = P.storage(); },
            [&] { factor_unbatched(pool_work.data(), m, n, nmat); });

        const double rel = factor_error(P, pristine, V);
        check(rel <= 1e-9, "compact factorization matches LAPACK");

        const double sp = t_lap / t_cqr; /* cqr speedup over per-matrix ArmPL */
        log_speed_vs_lapack += std::log(sp);
        std::printf("%4d | %11.2f | %11.2e | %11.2e | %8.2fx | %.2e\n", n,
                    nmat * geqrf_gflop(m, n) / t_cqr, nmat / t_cqr, nmat / t_lap, sp,
                    rel);
    }

    std::printf("-----+-------------+-------------+-------------+-----------+"
                "------------------\n");
    std::printf("geometric-mean speedup (cqr compact vs per-matrix ArmPL LAPACK): "
                "%.2fx\n",
                std::exp(log_speed_vs_lapack / sizes.size()));
    return 0;
}
