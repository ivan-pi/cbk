/* bench_trs_compact.cpp
 *
 * Throughput benchmark of the *substitution* stage alone -- the solve that
 * follows a factorization already in hand -- over pools of many small matrices:
 *
 *   cbk-compact  cbk_dpotrs_compact    (Cholesky solve: two non-unit sweeps)
 *                cbk_dsytrsnp_compact  (unpivoted LDL^T solve: two unit sweeps
 *                                          and the D^-1 row scaling between)
 *   mkl-compact  mkl_dtrsm_compact x2  (MKL's compact pipeline for the Cholesky
 *                                          solve; it ships no compact potrs)
 *   per-matrix   LAPACKE_dpotrs / LAPACKE_dsytrs  (one matrix at a time)
 *
 * The factorization benchmarks (bench_potrf_compact) and the end-to-end solve
 * benchmarks (bench_posv_compact, bench_sysvnp_compact) bracket this stage but
 * never time it on its own, so a per-routine rate for potrs or sytrsnp could
 * only be inferred from a whole-solve time -- and dividing the solve's 2 n^2
 * nrhs by a time that also contains the factorization's n^3/3 yields a rate
 * that falls off with n for reasons that have nothing to do with the kernel.
 * This program times the sweeps directly: the pool is packed and factored once,
 * up front and untimed, and only the substitution is measured, with the
 * destroyed right-hand sides restored (untimed) before each pass. The factor
 * itself is read-only here, so it stays put across passes.
 *
 * What the numbers say: at one right-hand side the substitution moves as much
 * of the factor as it does arithmetic on it (n^2 reads for n^2 flops), so it is
 * bandwidth-bound and its rate is set by which level of cache serves the
 * factor on the repeat passes best_time takes the best of. With a *fixed*
 * matrix count the pool grows as n^2 and crosses a cache boundary partway
 * along the size list, so the rows either side of it are not measured in the
 * same regime -- which is how a cache knee reads as a property of the kernel.
 * --pool=MiB derives the count from a fixed pool size instead, holding every
 * size in one regime; each row reports the count and the pool it came to
 * either way. Reuse of the factor across right-hand sides is what lifts the
 * stage off the bound within a pass: --nrhs=k feeds the tuned path's 4/2/1-
 * column blocking.
 *
 * Usage:  bench_trs_compact [--nrhs=k] [--pool=MiB]
 *                           [--size-sweep=nmin:nmax[:stride]]
 *                           [--simdlen=2|4|8] [nmat] [reps]
 *         (defaults: 1 right-hand side, 512 matrices, 3 reps)
 *
 * With no --size-sweep it runs the comparison at the default size list; with
 * it, a cbk-only scan of both routines over the size range. --simdlen forces
 * the interleave width (2/4/8) instead of the host's widest. --pool replaces
 * the fixed [nmat] with a per-size count.
 *
 * Build: needs Intel MKL plus this repo's MKL-style API; wired up by
 * CMakeLists.txt as the `bench_trs_compact` target. OpenMP is used when
 * available. Build with host-tuned flags (e.g. `-DCMAKE_CXX_FLAGS="-O3
 * -march=native"`) so the open compact kernels emit the full vector width.
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "bench_util.hpp"

#include <cstdio>
#include <vector>

namespace {

using namespace cbk::bench;

/* Flops of the substitution alone, in GFLOP: the two triangular sweeps, plus
 * the n nrhs row scaling of the LDL^T solve (counted the way bench_sysvnp
 * counts it, so the two programs' symmetric numbers are comparable). */
double trs_gflop(int n, int nrhs, bool ldlt)
{
    const double dn = n, dr = nrhs;
    return (2.0 * dn * dn * dr + (ldlt ? dn * dr : 0.0)) * 1e-9;
}

/* The batch of systems, as in bench_posv_compact / bench_sysvnp_compact:
 * `nmat` symmetric diagonally dominant n x n matrices (SPD, or indefinite for
 * the LDL^T path) with right-hand sides B = A X for the known X(:,j) = j + 1. */
struct Systems {
    MatrixPool a, b;

    Systems(int n, int nmat, int nrhs, bool indefinite) : a(nmat, n, n), b(nmat, n, nrhs)
    {
        fill_sym_dd(a, indefinite);
        fill_known_rhs(a, b);
    }
};

/* A pool's *factor* in compact form, held twice: the pristine image and the
 * buffer the timed passes read. Only the right-hand sides are overwritten by a
 * solve, so the factor is packed and factored once here, untimed, and the
 * working copy is never restored. */
struct FactoredPool {
    PackedPool packed;
    cbk::detail::mkl_buffer<double> f;

    FactoredPool(const MatrixPool &A, MKL_COMPACT_PACK fmt, bool ldlt)
        : packed(A, fmt), f(packed.work())
    {
        MKL_INT info;
        packed.restore_into(f.get());
        const int n = A.rows();
        if (ldlt)
            cbk_dsytrfnp_compact(MKL_COL_MAJOR, MKL_LOWER, n, f.get(), n, &info, fmt,
                                 A.count());
        else
            cbk_dpotrf_compact(MKL_COL_MAJOR, MKL_LOWER, n, f.get(), n, &info, fmt,
                               A.count());
        check(info == 0, "compact factorization reported info != 0");
    }

    double *get() const { return f.get(); }
};

/* The same for the per-matrix LAPACK baseline: a dense pool-layout copy of the
 * factors, plus the pivots of the (pivoted) LDL^T, factored once and untimed.
 * LAPACK's ?sytrf is Bunch-Kaufman, so its factor is not the compact one --
 * the two paths solve the same systems, not with the same arithmetic, exactly
 * as in bench_sysvnp_compact. */
struct DenseFactors {
    aligned_vector<double> a;
    std::vector<MKL_INT> ipiv;

    DenseFactors(const MatrixPool &A, bool ldlt)
        : a(A.storage()), ipiv((size_t)A.count() * A.rows())
    {
        const int n = A.rows();
#pragma omp parallel for schedule(static)
        for (int v = 0; v < A.count(); ++v) {
            double *av = a.data() + (size_t)v * n * n;
            if (ldlt)
                LAPACKE_dsytrf(LAPACK_COL_MAJOR, 'L', n, av, n,
                               ipiv.data() + (size_t)v * n);
            else
                LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, av, n);
        }
    }
};

/* MKL's compact pipeline for the Cholesky solve: the forward and back sweeps,
 * kept per group of V from an OpenMP loop -- the same cache-resident structure
 * as the cbk call, since MKL's compact kernels are not threaded in this build
 * (sequential MKL). A padded last group is processed at full V; its identity
 * padding is safe in both sweeps. */
void potrs_mkl(const double *ap, double *bp, int n, int nrhs, int nmat, int V,
               MKL_COMPACT_PACK fmt)
{
    const int ngroups = (nmat + V - 1) / V;
#pragma omp parallel for schedule(static)
    for (int g = 0; g < ngroups; ++g) {
        const double *apg = ap + (size_t)g * n * n * V;
        double *bpg = bp + (size_t)g * n * nrhs * V;
        mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, MKL_NOTRANS, MKL_NONUNIT, n,
                          nrhs, 1.0, const_cast<double *>(apg), n, bpg, n, fmt, V);
        mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, MKL_TRANS, MKL_NONUNIT, n,
                          nrhs, 1.0, const_cast<double *>(apg), n, bpg, n, fmt, V);
    }
}

/* Per-matrix LAPACK substitution over a standard-layout pool: B is overwritten
 * with X, the factors are read-only. */
void trs_unbatched(const double *a, double *b, const MKL_INT *ipiv, int n, int nrhs,
                   int nmat, bool ldlt)
{
#pragma omp parallel for schedule(static)
    for (int v = 0; v < nmat; ++v) {
        const double *av = a + (size_t)v * n * n;
        double *bv = b + (size_t)v * n * nrhs;
        if (ldlt)
            LAPACKE_dsytrs(LAPACK_COL_MAJOR, 'L', n, nrhs, av, n, ipiv + (size_t)v * n,
                           bv, n);
        else
            LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', n, nrhs, av, n, bv, n);
    }
}

/* One size, one of the two routines: the cbk compact solve, the per-matrix
 * LAPACK solve, and (Cholesky only) MKL's compact trsm pair. Every path is
 * checked against the known solution. */
struct Row {
    double t_cbk, t_mkl, t_lap, err_cbk, err_lap;
};

Row run_one(int n, int nmat, int nrhs, int reps, int V, MKL_COMPACT_PACK fmt, bool ldlt)
{
    const Systems P(n, nmat, nrhs, ldlt);
    const FactoredPool F(P.a, fmt, ldlt);
    const PackedPool rhs(P.b, fmt);
    const DenseFactors D(P.a, ldlt);

    auto bp = rhs.work();
    auto restore_compact = [&] { rhs.restore_into(bp.get()); };
    aligned_vector<double> b_dense;
    auto restore_dense = [&] { b_dense = P.b.storage(); };

    MKL_INT info;
    Row r{};
    r.t_cbk = best_time(reps, restore_compact, [&] {
        if (ldlt)
            cbk_dsytrsnp_compact(MKL_COL_MAJOR, MKL_LOWER, n, nrhs, F.get(), n, bp.get(),
                                 n, &info, fmt, nmat);
        else
            cbk_dpotrs_compact(MKL_COL_MAJOR, MKL_LOWER, n, nrhs, F.get(), n, bp.get(), n,
                               &info, fmt, nmat);
    });
    r.err_cbk = unpacked_forward_error(bp.get(), n, nrhs, nmat, fmt);

    if (!ldlt) {
        r.t_mkl = best_time(reps, restore_compact,
                            [&] { potrs_mkl(F.get(), bp.get(), n, nrhs, nmat, V, fmt); });
        check(unpacked_forward_error(bp.get(), n, nrhs, nmat, fmt) < 1e-6,
              "MKL compact trsm pipeline did not recover the known solution");
    }

    r.t_lap = best_time(reps, restore_dense, [&] {
        trs_unbatched(D.a.data(), b_dense.data(), D.ipiv.data(), n, nrhs, nmat, ldlt);
    });
    r.err_lap = forward_error(b_dense.data(), n, nrhs, nmat);
    return r;
}

/* The matrix count for one size: the fixed [nmat], or the --pool policy's
 * per-size count. Reported on every row, with the pool it comes to, so a table
 * is readable without knowing which policy produced it. */
int nmat_at(const CmdArgs &args, int n, int nthreads)
{
    return args.pool_mib > 0
               ? nmat_for_pool(args.pool_mib, (std::size_t)n * n * sizeof(double), args.V,
                               nthreads)
               : args.nmat;
}

double pool_mib_of(int n, int nmat)
{
    return (double)nmat * n * n * sizeof(double) / 1048576.0;
}

/* Cholesky solve table: cbk vs MKL's compact trsm pair vs per-matrix LAPACK. */
void run_potrs(const CmdArgs &args, const std::vector<int> &sizes, int nthreads)
{
    std::printf("\nCholesky substitution (cbk_dpotrs_compact): two non-unit sweeps\n");
    std::printf("   n |   nmat | pool (MiB) | cbk GFLOP/s |   cbk mat/s |   mkl mat/s | "
                "lapack mat/s | cbk/mkl | cbk/lap | fwderr(cbk) | fwderr(lapack)\n");
    std::printf("-----+--------+------------+-------------+-------------+-------------+"
                "--------------+---------+---------+-------------+---------------\n");
    for (int n : sizes) {
        const int nmat = nmat_at(args, n, nthreads);
        const Row r = run_one(n, nmat, args.nrhs, args.reps, args.V, args.fmt,
                              /*ldlt=*/false);
        check(r.err_cbk < 1e-6, "cbk_dpotrs_compact did not recover the known solution");
        check(r.err_lap < 1e-6, "LAPACKE_dpotrs did not recover the known solution");
        std::printf("%4d | %6d | %10.2f | %11.2f | %11.2e | %11.2e | %12.2e | %6.2fx | "
                    "%6.2fx | %11.2e | %13.2e\n",
                    n, nmat, pool_mib_of(n, nmat),
                    nmat * trs_gflop(n, args.nrhs, false) / r.t_cbk, nmat / r.t_cbk,
                    nmat / r.t_mkl, nmat / r.t_lap, r.t_mkl / r.t_cbk, r.t_lap / r.t_cbk,
                    r.err_cbk, r.err_lap);
        std::fflush(stdout);
    }
    std::printf("-----+--------+------------+-------------+-------------+-------------+"
                "--------------+---------+---------+-------------+---------------\n");
}

/* LDL^T solve table: cbk vs per-matrix LAPACK (MKL has no compact sytrs). */
void run_sytrsnp(const CmdArgs &args, const std::vector<int> &sizes, int nthreads)
{
    std::printf("\nUnpivoted LDL^T substitution (cbk_dsytrsnp_compact): two unit sweeps "
                "and the D^-1 scaling\n");
    std::printf("   n |   nmat | pool (MiB) | cbk GFLOP/s |   cbk mat/s | lapack mat/s "
                "| cbk/lap | fwderr(cbk) | fwderr(lapack)\n");
    std::printf("-----+--------+------------+-------------+-------------+--------------+"
                "---------+-------------+---------------\n");
    for (int n : sizes) {
        const int nmat = nmat_at(args, n, nthreads);
        const Row r = run_one(n, nmat, args.nrhs, args.reps, args.V, args.fmt,
                              /*ldlt=*/true);
        check(r.err_cbk < 1e-6,
              "cbk_dsytrsnp_compact did not recover the known solution");
        check(r.err_lap < 1e-6, "LAPACKE_dsytrs did not recover the known solution");
        std::printf("%4d | %6d | %10.2f | %11.2f | %11.2e | %12.2e | %6.2fx | %11.2e | "
                    "%13.2e\n",
                    n, nmat, pool_mib_of(n, nmat),
                    nmat * trs_gflop(n, args.nrhs, true) / r.t_cbk, nmat / r.t_cbk,
                    nmat / r.t_lap, r.t_lap / r.t_cbk, r.err_cbk, r.err_lap);
        std::fflush(stdout);
    }
    std::printf("-----+--------+------------+-------------+-------------+--------------+"
                "---------+-------------+---------------\n");
}

/* Both routines side by side, cbk only: the scan that shows where the
 * substitution's rate turns over as the pool outgrows a cache level. */
void run_sweep(const CmdArgs &args, const std::vector<int> &sizes, int nthreads)
{
    std::printf("\nSubstitution size sweep: cbk only (throughput, no cross-check)\n");
    std::printf("   n |   nmat | pool A (MiB) |  potrs GFLOP/s | sytrsnp GFLOP/s\n");
    std::printf("-----+--------+--------------+----------------+----------------\n");
    for (int n : sizes) {
        const int nmat = nmat_at(args, n, nthreads);
        const Row p = run_one(n, nmat, args.nrhs, args.reps, args.V, args.fmt,
                              /*ldlt=*/false);
        const Row s = run_one(n, nmat, args.nrhs, args.reps, args.V, args.fmt,
                              /*ldlt=*/true);
        std::printf("%4d | %6d | %12.2f | %14.2f | %15.2f\n", n, nmat,
                    pool_mib_of(n, nmat), nmat * trs_gflop(n, args.nrhs, false) / p.t_cbk,
                    nmat * trs_gflop(n, args.nrhs, true) / s.t_cbk);
        std::fflush(stdout);
    }
    std::printf("-----+--------+--------------+----------------+----------------\n");
}

} /* anonymous namespace */

int main(int argc, char **argv)
{
    const CmdArgs args(argc, argv, "bench_trs_compact");

    /* Pin MKL's internal threading: the OpenMP outer loops are the only
     * parallelism. LAPACKE NaN-checking off so the per-matrix path is timed
     * clean. */
    mkl_set_num_threads(1);
    LAPACKE_set_nancheck(0);
    const int nthreads = omp_threads();

    std::vector<int> sizes;
    if (args.sweep)
        for (int n = args.sweep_min; n <= args.sweep_max; n += args.sweep_step)
            sizes.push_back(n);
    else
        sizes.assign(bench_sizes.begin(), bench_sizes.end());

    std::printf("Triangular substitution throughput: the solve stage alone, on a "
                "factor already in hand\n");
    if (args.pool_mib > 0)
        std::printf("pool=%.1f MiB (matrices derived per size)", args.pool_mib);
    else
        std::printf("matrices=%d (fixed; the pool then grows as n^2)", args.nmat);
    std::printf("  nrhs=%d  reps=%d  simdlen=%d (%s)  OpenMP threads=%d  "
                "(col-major lower, pre-packed and pre-factored)\n",
                args.nrhs, args.reps, args.V, compact_format_name(args.fmt), nthreads);

    if (args.sweep)
        run_sweep(args, sizes, nthreads);
    else {
        run_potrs(args, sizes, nthreads);
        run_sytrsnp(args, sizes, nthreads);
    }
    return 0;
}
