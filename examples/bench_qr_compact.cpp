/* bench_qr_compact.cpp
 *
 * Throughput benchmark: solving many small square systems A_v X_v = B_v with
 * the QR pipeline (X = R^-1 Q^T B), comparing five ways to run the same math:
 *
 *   MKL batched  mkl_dgeqrf_compact     -> cbk_dormqr_compact -> mkl_dtrsm_compact
 *   cbk batched  cbk_dgeqrf_compact -> cbk_dormqr_compact -> cbk_dtrsm_compact
 *   cbk gels     cbk_dgels_compact  (the three steps as one call per group)
 *   unbatched    LAPACKE_dgeqrf         -> LAPACKE_dormqr          -> cblas_dtrsm
 *   LAPACK gels  LAPACKE_dgels          (LAPACK's own one-call driver, per matrix)
 *
 * The two three-step batched paths run the same compact pipeline from different
 * libraries: the MKL path uses MKL's own `mkl_?geqrf_compact` and
 * `mkl_?trsm_compact`, the cbk path this repo's open `cbk_?geqrf_compact`
 * and `cbk_?trsm_compact`, so the cbk path runs the whole solve with no MKL
 * compute kernel (MKL only packs and unpacks) and their ratio is the end-to-end
 * MKL-vs-open comparison. MKL has no compact `ormqr`, so `cbk_dormqr_compact`
 * is shared by both. The gels path is the same open math fused into one call
 * (the apply-Q^T folded into the factorization, no separate reflector sweep), so
 * gels vs cbk-batch is what the fusion buys. The unbatched path is the
 * conventional per-matrix LAPACK baseline, and LAPACKE_dgels the like-for-like
 * one-call baseline for cbk_dgels_compact (it runs the same three steps
 * inside, blocked, plus its norm scaling and rank test).
 *
 * For each size, a pool of `nmat` well-conditioned matrices with known solution
 * X == 1 is built once, and each path solves it -- the batched paths packing
 * each group of `V` (the compact SIMD width) on the fly, the per-matrix path
 * factoring in place. The outer loop over the pool runs under OpenMP (MKL's own
 * threading pinned to 1); each size is timed `reps` times keeping the best, and
 * geometric-mean speedups across sizes are printed at the end. (Details on the
 * timing harness and the in-place working copy are at best_time() and
 * run_unbatched().)
 *
 * The batched pipeline is deliberately driven group by group from the caller's
 * loop, not as three whole-pool calls (which the routines would thread
 * internally): per group, pack -> geqrf -> ormqr -> trsm -> unpack all touch one
 * group's buffers, a few tens of KB that stay in L1/L2 across the five steps,
 * whereas whole-pool calls stream the entire pool through five separate passes.
 * Measured on 4 cores, the whole-pool variant was 15-55% slower over n = 10..100.
 * (For a single factorization the two are equivalent; see bench_geqrf_compact.) A single right-hand side per system (nrhs = 1); every path
 * is checked against the known solution X == 1, so the reported error is a
 * forward error, not a comparison to LAPACK.
 *
 * Usage:  bench_qr_compact [nmat] [reps]      (defaults: 1000 matrices, 3 reps)
 *
 * Build: needs Intel MKL plus this repo's cbk_ormqr_compact and
 * cbk_trsm_compact; wired up by CMakeLists.txt as the `bench_qr_compact`
 * target. OpenMP is used when available. For a fair cbk-vs-MKL comparison, build
 * with host-tuned flags (e.g. `-DCMAKE_CXX_FLAGS="-O3 -march=native"`) so the
 * open compact kernels emit the full vector width, matching MKL's AVX-512 runtime
 * dispatch; without it the geqrf-dominated cbk path runs the V-wide packs on the
 * baseline ISA and is unfairly slow (cbk/MKL well below 1).
 *
 * Assisted-by: Claude:claude-opus-4-8 Claude:claude-fable-5
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "cbk_mkl_alloc.h" /* mkl_alloc_bytes (calls mkl_malloc; links MKL) */
#include "bench_util.hpp"  /* check, best_time, omp_threads, format helpers */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
#include <algorithm>

namespace {

using namespace cbk::bench;

/* The batch of systems: `nmat` square matrices of order n in `a`, and their
 * matching right-hand sides in `b` (one RHS per system). The exact solution is
 * X == 1, so each RHS is the row sum b_v = A_v 1, and a correct solve returns
 * all ones. */
struct Systems {
    MatrixPool a, b;

    Systems(int n, int nmat) : a(nmat, n, n), b(nmat, n, 1)
    {
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int v = 0; v < nmat; ++v) {
            const auto A = a.view(v), B = b.view(v);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    A(i, j) = dist(rng);
            for (int d = 0; d < n; ++d)
                A(d, d) += 2.0 * n;       /* diag dominant */
            for (int i = 0; i < n; ++i) { /* B = A * ones */
                double s = 0.0;
                for (int j = 0; j < n; ++j)
                    s += A(i, j);
                B(i, 0) = s;
            }
        }
    }
};

/* Largest deviation of a computed solution (expected all ones) from 1. */
double sol_error(const double *x, int n)
{
    double e = 0.0;
    for (int i = 0; i < n; ++i)
        e = std::max(e, std::fabs(x[i] - 1.0));
    return e;
}

/* What runs the batched compute between the shared pack and unpack: the
 * three-step pipeline with MKL's or cbk's geqrf and trsm (MKL has no compact
 * ormqr, so cbk_dormqr is shared by both), or cbk's one-call gels. */
enum class Backend { Mkl, Cbk, Gels };

/* ===== batched path: compact group-of-V pipeline ======================= *
 * Process the pool in groups of V, packing/factoring/solving/unpacking each
 * group inside the timed region. `impl` picks the backend for the geqrf and the
 * trsm (MKL's compact kernels, or this repo's open drop-ins) -- the shared
 * cbk_dormqr and the pack/unpack around them are identical for both -- or
 * the one-call cbk_dgels_compact between the same pack and unpack. Returns
 * the max solution error. */
double run_batched(const Systems &P, MKL_COMPACT_PACK fmt, int V, Backend impl)
{
    const int n = P.a.rows(), nmat = P.a.count(), nrhs = 1;
    const int ngroups = (nmat + V - 1) / V;
    double maxerr = 0.0;

#pragma omp parallel reduction(max : maxerr)
    {
        /* Per-thread compact buffers, sized for a full group of V. */
        const int align = 64;
        auto ap_buf =
            cbk::detail::mkl_alloc_bytes<double>(compact_bytes(n, n, fmt, V), align);
        auto taup_buf =
            cbk::detail::mkl_alloc_bytes<double>(compact_bytes(n, 1, fmt, V), align);
        auto bp_buf =
            cbk::detail::mkl_alloc_bytes<double>(compact_bytes(n, nrhs, fmt, V), align);
        double *ap = ap_buf.get(), *taup = taup_buf.get(), *bp = bp_buf.get();

        /* Select the pipeline's kernels once: MKL's own compact kernels, or this
         * repo's open drop-ins (byte-identical signatures). The ormqr below is
         * always cbk's -- MKL ships no compact ormqr. (Unused, like taup, on the
         * gels path, which keeps its tau in work.) */
        const auto geqrf_compact =
            (impl == Backend::Cbk) ? cbk_dgeqrf_compact : mkl_dgeqrf_compact;
        const auto trsm_compact =
            (impl == Backend::Cbk) ? cbk_dtrsm_compact : mkl_dtrsm_compact;

        /* Each routine's workspace from its own query (gels's is its tau
         * scratch, the size of a compact tau buffer for one group). */
        MKL_INT info[1]; /* compact status: a single scalar (MKL convention) */
        double wq;
        if (impl == Backend::Gels)
            cbk_dgels_compact(MKL_COL_MAJOR, 'N', n, n, nrhs, ap, n, bp, n, &wq, -1, info,
                              fmt, V);
        else
            geqrf_compact(MKL_COL_MAJOR, n, n, ap, n, taup, &wq, -1, info, fmt, V);
        const MKL_INT lwork = (MKL_INT)wq;
        std::vector<double> work((size_t)std::max<MKL_INT>(lwork, 1));

        std::vector<const double *> Aptr(V), Bptr(V); /* per-matrix base pointers */
        std::vector<double> xout((size_t)V * n);      /* unpacked solutions      */
        std::vector<double *> Xptr(V);

#pragma omp for schedule(static)
        for (int g = 0; g < ngroups; ++g) {
            const int base = g * V;
            const MKL_INT cnt = std::min(V, nmat - base); /* last group may be short */

            for (int s = 0; s < cnt; ++s) {
                Aptr[s] = P.a[base + s];
                Bptr[s] = P.b[base + s];
                Xptr[s] = xout.data() + (size_t)s * n;
            }

            mkl_dgepack_compact(MKL_COL_MAJOR, n, n, Aptr.data(), n, ap, n, fmt, cnt);
            mkl_dgepack_compact(MKL_COL_MAJOR, n, nrhs, Bptr.data(), n, bp, n, fmt, cnt);

            if (impl == Backend::Gels) {
                cbk_dgels_compact(MKL_COL_MAJOR, 'N', n, n, nrhs, ap, n, bp, n,
                                  work.data(), lwork, info, fmt, cnt);
            }
            else {
                geqrf_compact(MKL_COL_MAJOR, n, n, ap, n, taup, work.data(), lwork, info,
                              fmt, cnt);
                double dummy;
                cbk_dormqr_compact(MKL_COL_MAJOR, 'L', 'T', n, nrhs, n, ap, n, taup, bp,
                                   n, &dummy, 1, info, fmt, cnt);
                trsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS, MKL_NONUNIT,
                             n, nrhs, 1.0, ap, n, bp, n, fmt, cnt);
            }

            mkl_dgeunpack_compact(MKL_COL_MAJOR, n, nrhs, Xptr.data(), n, bp, n, fmt,
                                  cnt);
            for (int s = 0; s < cnt; ++s)
                maxerr = std::max(maxerr, sol_error(Xptr[s], n));
        }
        /* ap/taup/bp freed by their RAII owners at end of the parallel region. */
    }
    return maxerr;
}

/* ===== non-batched paths: conventional per-matrix LAPACK =============== *
 * One dense matrix at a time, factoring in place (no per-matrix copy in the
 * hot loop -- the application does not reuse the matrix afterwards): the
 * dgeqrf -> dormqr -> dtrsm chain, or LAPACK's own one-call driver dgels (the
 * same QR solve for a square full-rank system, so the two differ by dgels's
 * bookkeeping; dgels is the like-for-like baseline for cbk-gels). The caller
 * refreshes `a`/`b` from the pristine pool outside the timed region, since the
 * factorization destroys them. `a` holds the matrices (n*n each), `b` the
 * right-hand sides (n each, overwritten with the solutions). Returns the max
 * solution error. */
double run_unbatched(int n, int nmat, double *a, double *b, bool dgels)
{
    const int nrhs = 1;
    double maxerr = 0.0;

#pragma omp parallel reduction(max : maxerr)
    {
        std::vector<double> tau(n);

#pragma omp for schedule(static)
        for (int v = 0; v < nmat; ++v) {
            double *A = a + (size_t)v * n * n; /* overwritten with (H, R) */
            double *B = b + (size_t)v * n;     /* overwritten with X */

            if (dgels)
                LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', n, n, nrhs, A, n, B, n);
            else {
                LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, n, A, n, tau.data());
                LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', n, nrhs, n, A, n, tau.data(),
                               B, n);
                cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                            CblasNonUnit, n, nrhs, 1.0, A, n, B, n);
            }
            maxerr = std::max(maxerr, sol_error(B, n));
        }
    }
    return maxerr;
}

} /* anonymous namespace */

int main(int argc, char **argv)
{
    const int nmat = (argc > 1) ? std::atoi(argv[1]) : 1000;
    const int reps = (argc > 2) ? std::atoi(argv[2]) : 3;
    check(nmat > 0 && reps > 0, "usage: bench_qr_compact [nmat>0] [reps>0]");

    /* mkl_get_format_compact() returns the architecture's optimal packing
     * format -- always one of SSE/AVX/AVX512 -- so V is one of 2/4/8. */
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = vlen_for_format<double>(fmt);

    /* Pin MKL's internal threading: the OpenMP outer loop is the only
     * parallelism, so per-call MKL threads would just oversubscribe. */
    mkl_set_num_threads(1);
    /* Turn LAPACKE NaN-checking off so the per-matrix path is timed clean. */
    LAPACKE_set_nancheck(0);

    const int nthreads = omp_threads();

    const int sizes[] = {10, 20, 30, 40, 50, 60, 80, 100};
    const int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    const int nrhs = 1; /* single RHS per system (see Systems / run_batched) */
    const double eps = std::numeric_limits<double>::epsilon();

    std::printf("QR solve throughput (matrices/second), five paths:\n");
    std::printf("  MKL-batch  mkl_dgeqrf_compact     -> cbk_dormqr_compact -> "
                "mkl_dtrsm_compact\n");
    std::printf("  cbk-batch  cbk_dgeqrf_compact -> cbk_dormqr_compact -> "
                "cbk_dtrsm_compact\n");
    std::printf("  cbk-gels   cbk_dgels_compact  (the three steps as one call per "
                "group)\n");
    std::printf("  unbatched  LAPACKE_dgeqrf         -> LAPACKE_dormqr          -> "
                "cblas_dtrsm\n");
    std::printf("  dgels      LAPACKE_dgels          (LAPACK's one-call driver, per "
                "matrix)\n");
    std::printf("matrices=%d  reps=%d  rhs=%d  simdlen=%d (%s)  OpenMP threads=%d\n\n",
                nmat, reps, nrhs, V, compact_format_name(fmt), nthreads);
    /* Throughput as matrices/second for each path, then the four speedups
     * BENCHMARKS.md explains; the error is the forward error vs the known
     * solution X == 1, not vs LAPACK. */
    std::printf("   n | MKL-batch   cbk-batch   cbk-gels    unbatched   dgels      | "
                "gels/dgels | gels/unbat | gels/cbk | cbk/MKL | max fwd err (vs X=1)\n");
    std::printf("     |  (mat/s)     (mat/s)     (mat/s)     (mat/s)     (mat/s)    | "
                "           |            |          |         |\n");
    std::printf("-----+-------------------------------------------------------------+-"
                "-----------+------------+----------+---------+-------------------\n");

    double log_gels_vs_dgels = 0.0, log_gels_vs_unbat = 0.0, log_gels_vs_cbk = 0.0,
           log_cbk_vs_mkl = 0.0;
    for (const int n : sizes) {
        const Systems P(n, nmat);

        /* The batched paths read the pool read-only (pack copies into the
         * interleaved buffers), so they need no reset. The non-batched path
         * factors in place, so refresh a destroyable working copy of the pool
         * before each pass -- untimed, mirroring an application that consumes
         * the matrix rather than copying it inside the solve. */
        aligned_vector<double> wa, wb; /* filled by restore, untimed, before each pass */
        const auto restore = [&] {
            wa = P.a.storage();
            wb = P.b.storage();
        };

        double err_m = 0.0, err_c = 0.0, err_g = 0.0, err_u = 0.0, err_d = 0.0;
        const double tb_mkl =
            best_time(reps, [] {}, [&] { err_m = run_batched(P, fmt, V, Backend::Mkl); });
        const double tb_cbk =
            best_time(reps, [] {}, [&] { err_c = run_batched(P, fmt, V, Backend::Cbk); });
        const double tb_gels = best_time(
            reps, [] {}, [&] { err_g = run_batched(P, fmt, V, Backend::Gels); });
        const double tu = best_time(reps, restore, [&] {
            err_u = run_unbatched(n, nmat, wa.data(), wb.data(), false);
        });
        const double td = best_time(reps, restore, [&] {
            err_d = run_unbatched(n, nmat, wa.data(), wb.data(), true);
        });

        const double rtol = 100.0 * n * eps;
        const double maxerr = std::max({err_m, err_c, err_g, err_u, err_d});
        check(maxerr <= rtol, "solve accuracy within rtol");

        const double gels_vs_dgels = td / tb_gels;
        const double gels_vs_unbat = tu / tb_gels;
        const double gels_vs_cbk = tb_cbk / tb_gels;
        const double cbk_vs_mkl = tb_mkl / tb_cbk;
        log_gels_vs_dgels += std::log(gels_vs_dgels);
        log_gels_vs_unbat += std::log(gels_vs_unbat);
        log_gels_vs_cbk += std::log(gels_vs_cbk);
        log_cbk_vs_mkl += std::log(cbk_vs_mkl);
        std::printf("%4d | %10.2e  %10.2e  %10.2e  %10.2e  %10.2e | %9.2fx | %9.2fx | "
                    "%7.2fx | %6.2fx | %.2e (rtol %.1e)\n",
                    n, nmat / tb_mkl, nmat / tb_cbk, nmat / tb_gels, nmat / tu, nmat / td,
                    gels_vs_dgels, gels_vs_unbat, gels_vs_cbk, cbk_vs_mkl, maxerr, rtol);
    }

    std::printf("-----+-------------------------------------------------------------+-"
                "-----------+------------+----------+---------+-------------------\n");
    std::printf("geometric-mean speedup across sizes:  cbk-gels vs dgels %.2fx   |   "
                "cbk-gels vs unbatched %.2fx   |   cbk-gels vs cbk-batch %.2fx   |   "
                "cbk-batch vs MKL-batch %.2fx\n",
                std::exp(log_gels_vs_dgels / nsizes),
                std::exp(log_gels_vs_unbat / nsizes), std::exp(log_gels_vs_cbk / nsizes),
                std::exp(log_cbk_vs_mkl / nsizes));
    return 0;
}
