/* bench_posv_compact.cpp
 *
 * Throughput benchmark of the end-to-end SPD *solve* A X = B over pools of many
 * small symmetric positive-definite matrices, four ways:
 *
 *   cbk fused    cbk_dposv_compact     (Cholesky factor + solve fused per
 *                                           group, one call on the pool)
 *   cbk 2-step   cbk_dpotrf_compact -> cbk_dpotrs_compact
 *                                          (the same group kernels, but two
 *                                           whole-pool calls streaming the
 *                                           pool twice)
 *   mkl-compact  mkl_dpotrf_compact -> mkl_dtrsm_compact x2
 *                                          (MKL's compact kernels; MKL ships
 *                                           no compact potrs/posv, so this is
 *                                           its native pipeline, kept per
 *                                           group so its factors are solved
 *                                           with cache-resident too)
 *   per-matrix   LAPACKE_dposv             (conventional one-matrix-at-a-time)
 *
 * fused vs 2-step is what the fusion buys: identical kernels and bits, only the
 * memory traffic differs (each group's factor solved with while cache-resident
 * vs the pool streamed once for the factorization and again for the solve), so
 * the ratio isolates the design decision of the fused driver -- grow the pool
 * past the cache to see it. cbk vs mkl-compact is the open-vs-MKL comparison of
 * the same batched pipeline; cbk vs LAPACK the batched-vs-conventional headline.
 *
 * To measure the solvers rather than data movement, the pool is packed into
 * compact form once, up front (A and B); only the solve is timed, and the
 * destroyed input is restored (untimed) before each pass. The cbk calls thread
 * their own loop over groups; MKL's compact kernels are not threaded here
 * (sequential MKL), so its pipeline and the per-matrix LAPACK path are driven
 * from an OpenMP loop of the same thread count. Every path is checked against
 * the known solution, so the benchmark doubles as an integration test.
 *
 * Usage:  bench_posv_compact [--nrhs=k] [--size-sweep=nmin:nmax[:stride]]
 *                            [--simdlen=2|4|8] [nmat] [reps]
 *         (defaults: 1 right-hand side, 512 matrices, 3 reps)
 *
 * With no --size-sweep it runs the 4-way comparison; with it, a cbk-only
 * (fused) throughput scan over the size range. --simdlen forces the interleave
 * width (2/4/8) instead of the host's widest.
 *
 * Build: needs Intel MKL plus this repo's MKL-style API; wired up by CMakeLists.txt
 * as the `bench_posv_compact` target. OpenMP is used when available. For a fair
 * cbk-vs-MKL comparison, build with host-tuned flags (e.g.
 * `-DCMAKE_CXX_FLAGS="-O3 -march=native"`) so the open compact kernels emit the
 * full vector width, matching MKL's AVX-512 runtime dispatch.
 *
 * Assisted-by: Claude:claude-fable-5
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "bench_util.hpp"

#include <array>
#include <cmath>
#include <cstdio>

namespace {

using namespace cbk::bench;

/* Flop count of the Cholesky solve in GFLOP: the factorization (chol_gflop)
 * plus the 2 n^2 nrhs of the two triangular sweeps. */
double posv_gflop(int n, int nrhs)
{
    const double dn = n;
    return chol_gflop(n) + 2.0 * dn * dn * nrhs * 1e-9;
}

/* The batch of systems: `nmat` n x n SPD matrices in `a` (bench_util's
 * symmetric diagonally dominant fill with a positive diagonal, the pool of
 * bench_potrf_compact), each with its right-hand sides B = A X (n x nrhs) in
 * `b` for the known X(:,j) = j + 1. */
struct Systems {
    MatrixPool a, b;

    Systems(int n, int nmat, int nrhs) : a(nmat, n, n), b(nmat, n, nrhs)
    {
        fill_sym_dd(a, /*indefinite=*/false);
        fill_known_rhs(a, b);
    }
};

/* The fused compact solve of a pre-packed pool in place, column-major lower:
 * one call, threaded inside the library. */
void solve_fused(double *ap, double *bp, int n, int nrhs, int nmat, MKL_COMPACT_PACK fmt)
{
    MKL_INT info;
    cbk_dposv_compact(MKL_COL_MAJOR, MKL_LOWER, n, nrhs, ap, n, bp, n, &info, fmt, nmat);
}

/* The same kernels as two whole-pool calls: the pool is streamed once for the
 * factorization and again for the solve. Bit-identical result to the fused
 * call; the throughput difference is what the fusion buys. */
void solve_twostep(double *ap, double *bp, int n, int nrhs, int nmat,
                   MKL_COMPACT_PACK fmt)
{
    MKL_INT info;
    cbk_dpotrf_compact(MKL_COL_MAJOR, MKL_LOWER, n, ap, n, &info, fmt, nmat);
    cbk_dpotrs_compact(MKL_COL_MAJOR, MKL_LOWER, n, nrhs, ap, n, bp, n, &info, fmt, nmat);
}

/* MKL's native compact pipeline: potrf, then the forward and back trsm sweeps
 * (MKL has no compact potrs/posv). Kept per group of V from an OpenMP loop --
 * the same cache-resident structure as the fused cbk path, and the layout
 * bench_qr_compact measured 15-55% faster than whole-pool passes -- since
 * MKL's compact kernels are not threaded in this build (sequential MKL). A
 * padded last group is processed at full V; its identity padding is safe in
 * all three kernels. */
void solve_mkl(double *ap, double *bp, int n, int nrhs, int nmat, int V,
               MKL_COMPACT_PACK fmt)
{
    const int ngroups = (nmat + V - 1) / V;
#pragma omp parallel for schedule(static)
    for (int g = 0; g < ngroups; ++g) {
        MKL_INT info;
        double *apg = ap + (size_t)g * n * n * V;    /* group stride n*n*V   */
        double *bpg = bp + (size_t)g * n * nrhs * V; /* group stride n*nrhs*V */
        mkl_dpotrf_compact(MKL_COL_MAJOR, MKL_LOWER, n, apg, n, &info, fmt, V);
        mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, MKL_NOTRANS, MKL_NONUNIT, n,
                          nrhs, 1.0, apg, n, bpg, n, fmt, V);
        mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, MKL_TRANS, MKL_NONUNIT, n,
                          nrhs, 1.0, apg, n, bpg, n, fmt, V);
    }
}

/* Per-matrix LAPACK ?posv of a standard-layout pool copy in place (lower):
 * A is overwritten with its Cholesky factor, B with X. */
void solve_unbatched(double *a, double *b, int n, int nrhs, int nmat)
{
#pragma omp parallel for schedule(static)
    for (int v = 0; v < nmat; ++v)
        LAPACKE_dposv(LAPACK_COL_MAJOR, 'L', n, nrhs, a + (size_t)v * n * n, n,
                      b + (size_t)v * n * nrhs, n);
}

/* Single-solver size sweep: the fused compact solve of a pre-packed pool at
 * each n in [nmin, nmax] (step stride), throughput only -- no comparison, so
 * it stays cheap and isolates the kernel. The raw best-pass time is printed
 * next to the derived rates (rate = nmat / time): a total near the timer
 * granularity flags a noisy row -- raise nmat until it is comfortably above
 * the clock resolution. */
void run_sweep(int nmat, int reps, int nrhs, int nmin, int nmax, int stride,
               MKL_COMPACT_PACK fmt, int V, int nthreads)
{
    std::printf("SPD solve size sweep: cbk_dposv_compact only (throughput, "
                "no cross-check)\n");
    std::printf("matrices=%d  nrhs=%d  reps=%d  simdlen=%d (%s)  OpenMP threads=%d  "
                "(SPD, col-major lower, pre-packed)\n\n",
                nmat, nrhs, reps, V, compact_format_name(fmt), nthreads);
    std::printf("   n |  total (s) | cbk GFLOP/s |   cbk mat/s\n");
    std::printf("-----+------------+-------------+-------------\n");

    for (int n = nmin; n <= nmax; n += stride) {
        const Systems P(n, nmat, nrhs);
        PackedSystems pk(P.a, P.b, fmt);
        auto ap = pk.a.work();
        auto bp = pk.b.work();
        auto restore = [&] { pk.restore_into(ap.get(), bp.get()); };
        const double t = best_time(
            reps, restore, [&] { solve_fused(ap.get(), bp.get(), n, nrhs, nmat, fmt); });
        std::printf("%4d | %10.3e | %11.2f | %11.2e\n", n, t,
                    nmat * posv_gflop(n, nrhs) / t, nmat / t);
    }
    std::printf("-----+------------+-------------+-------------\n");
}

} /* anonymous namespace */

int main(int argc, char **argv)
{
    const CmdArgs args(argc, argv, "bench_posv_compact");
    const int nmat = args.nmat, reps = args.reps, nrhs = args.nrhs, V = args.V;
    const MKL_COMPACT_PACK fmt = args.fmt;

    /* Pin MKL's internal threading: the OpenMP outer loop is the only
     * parallelism. LAPACKE NaN-checking off so the per-matrix path is timed clean. */
    mkl_set_num_threads(1);
    LAPACKE_set_nancheck(0);
    const int nthreads = omp_threads();

    if (args.sweep) {
        run_sweep(nmat, reps, nrhs, args.sweep_min, args.sweep_max, args.sweep_step, fmt,
                  V, nthreads);
        return 0;
    }

    /* Square sizes spanning the target range (order 3..500, emphasis below 170),
     * mixing sizes that are not multiples of the SIMD width V with the round
     * powers, as in the factorization benchmarks. */
    constexpr std::array sizes = {8,  16,  24,  30,  32,  45,  48,  60, 64,
                                  96, 105, 128, 168, 170, 256, 384, 500};

    std::printf("SPD solve throughput: cbk_dposv_compact (fused Cholesky factor + "
                "solve) vs cbk two-step vs MKL compact pipeline vs per-matrix "
                "LAPACKE_dposv\n");
    std::printf("matrices=%d  nrhs=%d  reps=%d  simdlen=%d (%s)  OpenMP threads=%d  "
                "(SPD, col-major lower, pre-packed)\n\n",
                nmat, nrhs, reps, V, compact_format_name(fmt), nthreads);
    /* Throughput as matrices/second (scientific) so it stays legible across the
     * whole size range; three speedup ratios show where the wins come from
     * (fused vs the same kernels unfused, vs MKL's pipeline, vs LAPACK). The
     * error column is the fused path's forward error against the known
     * solution; the MKL and LAPACK paths are gated at the same tolerance. */
    std::printf("   n | cbk GFLOP/s |   cbk mat/s | 2step mat/s |   mkl mat/s | "
                "lapack mat/s | fus/2st | cbk/mkl | cbk/lap | fwderr(cbk)\n");
    std::printf("-----+-------------+-------------+-------------+-------------+"
                "--------------+---------+---------+---------+------------\n");

    double log_speed_vs_lapack = 0.0, log_speed_vs_2step = 0.0, log_speed_vs_mkl = 0.0;
    for (int n : sizes) {
        const Systems P(n, nmat, nrhs);
        PackedSystems pk(P.a, P.b, fmt);

        /* working copies: compact (the three compact paths share one pair) and
         * standard layout (LAPACK) */
        auto ap = pk.a.work();
        auto bp = pk.b.work();
        aligned_vector<double> a_work, b_work;

        /* every path destroys A and B, so restore the input (untimed) before
         * each timed pass */
        auto restore_compact = [&] { pk.restore_into(ap.get(), bp.get()); };
        auto restore_dense = [&] {
            a_work = P.a.storage();
            b_work = P.b.storage();
        };

        /* Each path's correctness gate against the known solution reads the X
         * its last timed pass left behind, so no path is solved twice: the
         * fused error is taken before the next path's restore overwrites bp
         * (the 2-step path needs no gate of its own -- it is bit-identical to
         * the fused one by construction, which the test suites gate). */
        double t_cbk = best_time(reps, restore_compact, [&] {
            solve_fused(ap.get(), bp.get(), n, nrhs, nmat, fmt);
        });
        const double err_cbk = unpacked_forward_error(bp.get(), n, nrhs, nmat, fmt);
        double t_2st = best_time(reps, restore_compact, [&] {
            solve_twostep(ap.get(), bp.get(), n, nrhs, nmat, fmt);
        });
        double t_mkl = best_time(reps, restore_compact, [&] {
            solve_mkl(ap.get(), bp.get(), n, nrhs, nmat, V, fmt);
        });
        const double err_mkl = unpacked_forward_error(bp.get(), n, nrhs, nmat, fmt);
        double t_lap = best_time(reps, restore_dense, [&] {
            solve_unbatched(a_work.data(), b_work.data(), n, nrhs, nmat);
        });
        const double err_lap = forward_error(b_work.data(), n, nrhs, nmat);
        check(err_cbk <= 1e-9, "fused compact solve recovers the known solution");
        check(err_mkl <= 1e-9, "MKL compact pipeline recovers the known solution");
        check(err_lap <= 1e-9, "LAPACK solve recovers the known solution");

        const double sp_lap = t_lap / t_cbk;
        const double sp_2st = t_2st / t_cbk;
        const double sp_mkl = t_mkl / t_cbk;
        const double gflops_cbk = nmat * posv_gflop(n, nrhs) / t_cbk;
        log_speed_vs_lapack += std::log(sp_lap);
        log_speed_vs_2step += std::log(sp_2st);
        log_speed_vs_mkl += std::log(sp_mkl);
        std::printf("%4d | %11.2f | %11.2e | %11.2e | %11.2e | %12.2e | %6.2fx | "
                    "%6.2fx | %6.2fx | %10.2e\n",
                    n, gflops_cbk, nmat / t_cbk, nmat / t_2st, nmat / t_mkl, nmat / t_lap,
                    sp_2st, sp_mkl, sp_lap, err_cbk);
    }

    std::printf("-----+-------------+-------------+-------------+-------------+"
                "--------------+---------+---------+---------+------------\n");
    const double denom = sizes.size();
    std::printf("geometric-mean speedups of the fused compact solve: %.2fx vs its own "
                "two-step calls, %.2fx vs the MKL compact pipeline, %.2fx vs "
                "per-matrix LAPACKE_dposv\n",
                std::exp(log_speed_vs_2step / denom), std::exp(log_speed_vs_mkl / denom),
                std::exp(log_speed_vs_lapack / denom));
    return 0;
}
