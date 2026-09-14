/* bench_sysvnp_compact.cpp
 *
 * Throughput benchmark of the end-to-end symmetric *solve* A X = B over pools of
 * many small symmetric indefinite matrices, comparing the fused compact solver
 * with the conventional per-matrix LAPACK driver:
 *
 *   cbk-compact  cbk_dsysvnp_compact  (this project's fused unpivoted LDL^T
 *                                          factor + solve, one call on the pool)
 *   per-matrix   LAPACKE_dsysv            (Bunch-Kaufman LDL^T factor + solve,
 *                                          one matrix at a time)
 *
 * There is no MKL compact yardstick here: MKL ships no compact sytrf/sysv of any
 * kind (the reason these routines exist). The two paths do not run the same
 * arithmetic either -- LAPACK pivots, the compact solver does not -- so this is
 * a comparison of the two ways of solving the batch, not of two implementations
 * of one algorithm; the pool is built so that the unpivoted factorization is
 * safe (diagonally dominant), and both paths are checked against the known
 * solution.
 *
 * To measure the solvers rather than data movement, the pool is packed into
 * compact form once, up front (A and B); only the solve is timed, and the
 * destroyed input is restored (untimed) before each pass. The cbk path is one
 * call on the whole pool -- the routine threads its own loop over groups, and
 * factors and solves each group while its factor is cache-resident. The
 * per-matrix LAPACK path is driven from an OpenMP loop of the same thread
 * count, with a per-thread ipiv (LAPACKE_dsysv allocates its own workspace).
 *
 * Usage:  bench_sysvnp_compact [--nrhs=k] [--size-sweep=nmin:nmax[:stride]]
 *                              [--simdlen=2|4|8] [nmat] [reps]
 *         (defaults: 1 right-hand side, 512 matrices, 3 reps)
 *
 * With no --size-sweep it runs the cbk-vs-LAPACK comparison; with it, a
 * cbk-only throughput scan over the size range. --simdlen forces the interleave
 * width (2/4/8) instead of the host's widest.
 *
 * Build: needs Intel MKL plus this repo's MKL-style API; wired up by CMakeLists.txt
 * as the `bench_sysvnp_compact` target. OpenMP is used when available. Build with
 * host-tuned flags (e.g. `-DCMAKE_CXX_FLAGS="-O3 -march=native"`) so the open
 * compact kernel emits the full vector width.
 *
 * Assisted-by: Claude:claude-fable-5
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "bench_util.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

using namespace cbk::bench;

/* Flop count of the unpivoted LDL^T solve in GFLOP: the factorization
 * (chol_gflop -- the same count as Cholesky, the n reciprocals uncounted as
 * LAPACK leaves the square roots uncounted) plus the 2 n^2 nrhs of the two
 * triangular sweeps and the n nrhs diagonal scaling. */
double sysv_gflop(int n, int nrhs)
{
    const double dn = n, dr = nrhs;
    return chol_gflop(n) + (2.0 * dn * dn * dr + dn * dr) * 1e-9;
}

/* The batch of systems: `nmat` n x n symmetric *indefinite* matrices in `a`
 * (bench_util's symmetric diagonally dominant fill with alternating diagonal
 * sign: every leading principal minor nonsingular, so the unpivoted LDL^T
 * exists with bounded element growth -- the class of input the unpivoted
 * solver is for -- yet genuinely indefinite, so LAPACK needs ?sysv, not
 * ?posv), each with its right-hand sides B = A X (n x nrhs) in `b` for the
 * known X(:,j) = j + 1. */
struct Systems {
    MatrixPool a, b;

    Systems(int n, int nmat, int nrhs) : a(nmat, n, n), b(nmat, n, nrhs)
    {
        fill_sym_dd(a, /*indefinite=*/true);
        fill_known_rhs(a, b);
    }
};

/* The fused compact solve of a pre-packed pool in place, column-major lower:
 * one call, threaded inside the library. */
void solve_compact(double *ap, double *bp, int n, int nrhs, int nmat,
                   MKL_COMPACT_PACK fmt)
{
    MKL_INT info;
    cbk_dsysvnp_compact(MKL_COL_MAJOR, MKL_LOWER, n, nrhs, ap, n, bp, n, &info, fmt,
                        nmat);
}

/* Per-matrix LAPACK ?sysv (Bunch-Kaufman) of a standard-layout pool copy in
 * place (lower): A is overwritten with its pivoted factor, B with X. ipiv is
 * one n-vector per OpenMP thread. */
void solve_unbatched(double *a, double *b, int n, int nrhs, int nmat, MKL_INT *ipiv)
{
#pragma omp parallel for schedule(static)
    for (int v = 0; v < nmat; ++v) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        LAPACKE_dsysv(LAPACK_COL_MAJOR, 'L', n, nrhs, a + (size_t)v * n * n, n,
                      ipiv + (size_t)tid * n, b + (size_t)v * n * nrhs, n);
    }
}

/* Single-solver size sweep: the fused compact solve of a pre-packed pool at
 * each n in [nmin, nmax] (step stride), throughput only -- no LAPACK
 * comparison, so it stays cheap and isolates the kernel. The raw best-pass
 * time is printed next to the derived rates (rate = nmat / time): a total near
 * the timer granularity flags a noisy row -- raise nmat until it is
 * comfortably above the clock resolution. */
void run_sweep(int nmat, int reps, int nrhs, int nmin, int nmax, int stride,
               MKL_COMPACT_PACK fmt, int V, int nthreads)
{
    std::printf("Symmetric solve size sweep: cbk_dsysvnp_compact only (throughput, "
                "no cross-check)\n");
    std::printf("matrices=%d  nrhs=%d  reps=%d  simdlen=%d (%s)  OpenMP threads=%d  "
                "(indefinite, col-major lower, pre-packed)\n\n",
                nmat, nrhs, reps, V, compact_format_name(fmt), nthreads);
    std::printf("   n |  total (s) | cbk GFLOP/s |   cbk mat/s\n");
    std::printf("-----+------------+-------------+-------------\n");

    for (int n = nmin; n <= nmax; n += stride) {
        const Systems P(n, nmat, nrhs);
        PackedSystems pk(P.a, P.b, fmt);
        auto ap = pk.a.work();
        auto bp = pk.b.work();
        auto restore = [&] { pk.restore_into(ap.get(), bp.get()); };
        const double t = best_time(reps, restore, [&] {
            solve_compact(ap.get(), bp.get(), n, nrhs, nmat, fmt);
        });
        std::printf("%4d | %10.3e | %11.2f | %11.2e\n", n, t,
                    nmat * sysv_gflop(n, nrhs) / t, nmat / t);
    }
    std::printf("-----+------------+-------------+-------------\n");
}

} /* anonymous namespace */

int main(int argc, char **argv)
{
    const CmdArgs args(argc, argv, "bench_sysvnp_compact");
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

    std::printf("Symmetric solve throughput: cbk_dsysvnp_compact (fused unpivoted "
                "LDL^T) vs per-matrix LAPACKE_dsysv (Bunch-Kaufman)\n");
    std::printf("matrices=%d  nrhs=%d  reps=%d  simdlen=%d (%s)  OpenMP threads=%d  "
                "(indefinite, col-major lower, pre-packed)\n\n",
                nmat, nrhs, reps, V, compact_format_name(fmt), nthreads);
    /* Throughput as matrices/second (scientific) so it stays legible across the
     * whole size range. The error columns are forward errors against the known
     * solution, one per path. */
    std::printf("   n | cbk GFLOP/s |   cbk mat/s | lapack mat/s | cbk/lap | "
                "fwderr(cbk) | fwderr(lapack)\n");
    std::printf("-----+-------------+-------------+--------------+---------+"
                "-------------+---------------\n");

    /* one ipiv per thread, sized for the largest order in the list -- its back,
     * since bench_sizes is ascending (bench_util.hpp) */
    std::vector<MKL_INT> ipiv((size_t)nthreads * bench_sizes.back());
    double log_speed = 0.0;
    for (int n : bench_sizes) {
        const Systems P(n, nmat, nrhs);
        PackedSystems pk(P.a, P.b, fmt);

        /* working copies: compact (cbk) and standard layout (LAPACK) */
        auto ap = pk.a.work();
        auto bp = pk.b.work();
        aligned_vector<double> a_work, b_work;

        /* both paths destroy A and B, so restore the input (untimed) before
         * each timed pass */
        auto restore_compact = [&] { pk.restore_into(ap.get(), bp.get()); };
        auto restore_dense = [&] {
            a_work = P.a.storage();
            b_work = P.b.storage();
        };

        double t_cbk = best_time(reps, restore_compact, [&] {
            solve_compact(ap.get(), bp.get(), n, nrhs, nmat, fmt);
        });
        double t_lap = best_time(reps, restore_dense, [&] {
            solve_unbatched(a_work.data(), b_work.data(), n, nrhs, nmat, ipiv.data());
        });

        /* correctness gates: both paths vs the known solution, each read from
         * the X its last timed pass left behind (t_lap does not touch bp) */
        const double err_cbk = unpacked_forward_error(bp.get(), n, nrhs, nmat, fmt);
        const double err_lap = forward_error(b_work.data(), n, nrhs, nmat);
        check(err_cbk <= 1e-9, "compact solve recovers the known solution");
        check(err_lap <= 1e-9, "LAPACK solve recovers the known solution");

        const double sp = t_lap / t_cbk;
        const double gflops_cbk = nmat * sysv_gflop(n, nrhs) / t_cbk;
        log_speed += std::log(sp);
        std::printf("%4d | %11.2f | %11.2e | %12.2e | %6.2fx | %11.2e | %13.2e\n", n,
                    gflops_cbk, nmat / t_cbk, nmat / t_lap, sp, err_cbk, err_lap);
    }

    std::printf("-----+-------------+-------------+--------------+---------+"
                "-------------+---------------\n");
    std::printf("geometric-mean speedup (cbk fused compact solve vs per-matrix "
                "LAPACKE_dsysv): %.2fx\n",
                std::exp(log_speed / bench_sizes.size()));
    return 0;
}
