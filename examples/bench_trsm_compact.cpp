/* bench_trsm_compact.cpp
 *
 * Throughput benchmark of the triangular solve
 *
 *     op(A) X = alpha B  (side='L')   or   X op(A) = alpha B  (side='R')
 *
 * over pools of many small triangular systems, comparing implementations of
 * the same BLAS ?trsm math:
 *
 *   cbk-compact  dtrsm_compact      (the portable C API of cbk.h: this
 *                                    project's batched SIMD kernel)
 *   per-matrix   dtrsm              (the ?trsm of the BLAS/LAPACK library the
 *                                    build linked, one matrix at a time)
 *   mkl-compact  mkl_dtrsm_compact  (Intel MKL's batched compact kernel;
 *                                    MKL build only)
 *
 * A benchmark of the portable C API, so it needs no MKL: it packs with the
 * library-side pack (src/cbk_compact_pack.hpp) and takes its baseline from
 * the BLAS/LAPACK CMake's find_package(LAPACK) finds (BLA_VENDOR selects:
 * OpenBLAS, sequential MKL, ...), so `-DCBK_BUILD_BENCHMARKS=ON` builds it in
 * a tree without the MKL extension. In the MKL build (-DCBK_WITH_MKL=ON) MKL
 * is that BLAS, and MKL's own compact kernel is the third path. The baseline
 * is always a library's ?trsm, never the scalar reference kernels of the test
 * suites, which are correctness oracles, not competitors.
 *
 * Every system is solved on the same case, column-major with `alpha = 1`;
 * the default is the back-substitution that closes a batched QR solve
 * (side='L', upper, non-transposed, non-unit: R X = Q^T B), and --side,
 * --uplo, --transa, --diag select any other. The triangular pool is strictly
 * diagonally dominant (diagonal 2n, off-diagonals in [-1, 1]), the
 * right-hand sides are formed from the known solution X(:,j) = j + 1, and
 * every path is checked against it, so the reported error is a forward error
 * and the benchmark doubles as an integration test. To measure the kernels
 * rather than data movement the pool is packed once, up front; only the
 * solve is timed, and the overwritten right-hand sides are restored (untimed)
 * before each pass. The cbk path is one call on the whole pool -- the routine
 * threads its own loop over groups. The per-matrix path (and MKL's compact
 * kernel, which is not threaded in this build) is driven from an OpenMP loop
 * of the same thread count; a BLAS that threads internally must be pinned to
 * one thread through its own knob (OPENBLAS_NUM_THREADS=1, ...; the MKL build
 * pins MKL itself).
 *
 * Usage:  bench_trsm_compact [--size-sweep=nmin:nmax[:stride]] [--simdlen=2|4|8]
 *         [--nrhs=k] [--side=L|R] [--uplo=U|L] [--transa=N|T] [--diag=N|U]
 *         [nmat] [reps]      (defaults: 512 matrices, 3 reps, 1 right-hand side)
 *
 * With no --size-sweep it runs the comparison over bench_sizes; with it, a
 * cbk-only throughput scan over the size range. --simdlen forces the
 * interleave width (2/4/8) instead of the host's widest, which is detected at
 * run time (host_simdlen, bench_common.hpp).
 *
 * Build: `-DCBK_BUILD_BENCHMARKS=ON` (needs a BLAS/LAPACK; `-DBLA_VENDOR`
 * selects it) or `-DCBK_WITH_MKL=ON` (needs Intel MKL); wired up by
 * CMakeLists.txt as the `bench_trsm_compact` target. OpenMP is used when
 * available. Build with host-tuned flags (e.g.
 * `-DCMAKE_CXX_FLAGS="-O3 -march=native"`) so the open compact kernel emits
 * the full vector width -- the header line says what it was compiled for.
 *
 * Assisted-by: Claude:claude-fable-5
 */

#include "cbk.h"

#include "bench_common.hpp"

#ifdef CBK_BENCH_WITH_MKL
#include "cbk_compat.h" /* format_for_vlen / vlen_for_format / compact_format_name */

#include <mkl.h>
#include <mkl_compact.h>
#else
#include <cstdint>
#endif

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

namespace {

using namespace cbk::bench;
#ifdef CBK_BENCH_WITH_MKL
using cbk::detail::format_for_vlen; /* interleave width <-> MKL pack format */
using cbk::detail::vlen_for_format;
#endif

/* The case every system is solved on: A is the order-s triangular factor, B is
 * m x n -- s x nrhs for side='L' (op(A) X = B), nrhs x s for side='R'
 * (X op(A) = B). */
struct TrsmCase {
    char side = 'L', uplo = 'U', transa = 'N', diag = 'N';
    int s = 0, nrhs = 1;

    int m() const { return side == 'L' ? s : nrhs; }
    int n() const { return side == 'L' ? nrhs : s; }
    /* Standard ?trsm flop count in GFLOP: s^2 per right-hand side (adds +
     * mults over the triangle, LAWN 41), the same for every side/uplo/transa. */
    double gflop() const { return (double)s * s * nrhs * 1e-9; }
};

/* The per-matrix baseline: the ?trsm of the BLAS/LAPACK library this
 * benchmark was linked against. The MKL build calls MKL's CBLAS (MKL_INT, so
 * the ilp64 interface works too); the LAPACK build calls the Fortran symbol
 * directly, which is the one interface every BLAS ships (the CBLAS header is
 * not part of find_package(LAPACK)), with the integer width FindBLAS was
 * asked for (CBK_BENCH_BLAS_INT, int unless BLA_SIZEOF_INTEGER=8) and the
 * hidden character-length arguments gfortran expects (harmless to a BLAS
 * without). */
#ifdef CBK_BENCH_WITH_MKL
void blas_dtrsm(const TrsmCase &c, const double *a, double *b)
{
    cblas_dtrsm(CblasColMajor, c.side == 'L' ? CblasLeft : CblasRight,
                c.uplo == 'U' ? CblasUpper : CblasLower,
                c.transa == 'N' ? CblasNoTrans : CblasTrans,
                c.diag == 'U' ? CblasUnit : CblasNonUnit, c.m(), c.n(), 1.0, a, c.s, b,
                c.m());
}
#else
#ifndef CBK_BENCH_BLAS_INT
#define CBK_BENCH_BLAS_INT int
#endif
using blas_int = CBK_BENCH_BLAS_INT;
extern "C" void dtrsm_(const char *side, const char *uplo, const char *transa,
                       const char *diag, const blas_int *m, const blas_int *n,
                       const double *alpha, const double *a, const blas_int *lda,
                       double *b, const blas_int *ldb, std::size_t, std::size_t,
                       std::size_t, std::size_t);

void blas_dtrsm(const TrsmCase &c, const double *a, double *b)
{
    const blas_int m = c.m(), n = c.n(), lda = c.s, ldb = c.m();
    const double alpha = 1.0;
    dtrsm_(&c.side, &c.uplo, &c.transa, &c.diag, &m, &n, &alpha, a, &lda, b, &ldb, 1, 1,
           1, 1);
}
#endif

/* A pool of `nmat` order-s triangular matrices: random off-diagonals in
 * [-1, 1] in the referenced triangle, a diagonal of 2s (strictly diagonally
 * dominant in either orientation, so op(A) is well conditioned), the other
 * triangle stored as zeros and, for diag='U', the unit diagonal stored as 1 --
 * no solve reads either, but the dense product that forms the right-hand
 * sides does, so what is stored is what the solve assumes. */
MatrixPool make_pool(const TrsmCase &c, int nmat)
{
    const int s = c.s;
    MatrixPool P(nmat, s, s);
    std::mt19937_64 rng(2025);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    const bool upper = c.uplo == 'U';
    for (int v = 0; v < nmat; ++v) {
        const auto A = P.view(v);
        for (int j = 0; j < s; ++j)
            for (int i = 0; i < s; ++i) {
                const bool referenced = upper ? i < j : i > j;
                A(i, j) = i == j       ? (c.diag == 'U' ? 1.0 : 2.0 * s)
                          : referenced ? dist(rng)
                                       : 0.0;
            }
    }
    return P;
}

/* B := op(A) X (side='L') or X op(A) (side='R') for the known solution X of
 * fill_known_x, as a plain dense product over the stored A (whose unreferenced
 * triangle is zero and whose diagonal is what the solve assumes). */
void fill_rhs(const TrsmCase &c, const MatrixPool &A, MatrixPool &B)
{
    const int m = c.m(), n = c.n();
    std::vector<double> xs((size_t)m * n);
    const auto X = mat_view(xs.data(), m, n);
    fill_known_x(X);
    for (int v = 0; v < A.count(); ++v) {
        const auto Av = A.view(v);
        const auto Aop = c.transa == 'N' ? Av : Av.transposed();
        const auto Bv = B.view(v);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                double sum = 0;
                if (c.side == 'L')
                    for (int k = 0; k < m; ++k)
                        sum += Aop(i, k) * X(k, j);
                else
                    for (int k = 0; k < n; ++k)
                        sum += X(i, k) * Aop(k, j);
                Bv(i, j) = sum;
            }
    }
}

/* Solve a pre-packed compact pool in place with cbk: one call on the whole
 * pool, threaded inside the library. */
void solve_cbk(const TrsmCase &c, const double *ap, double *bp, int V, int nmat)
{
    const int info = dtrsm_compact('C', c.side, c.uplo, c.transa, c.diag, c.m(), c.n(),
                                   1.0, ap, c.s, bp, c.m(), V, nmat);
    check(info == 0, "dtrsm_compact accepted its arguments");
}

#ifdef CBK_BENCH_WITH_MKL
/* The same on MKL's compact kernel: an OpenMP loop over the groups of V (the
 * kernel is not threaded in this build), so both compact paths run on the same
 * thread count. */
void solve_mkl(const TrsmCase &c, const double *ap, double *bp, int V,
               MKL_COMPACT_PACK fmt, int nmat)
{
    const int s = c.s, m = c.m(), n = c.n();
    const int ngroups = (nmat + V - 1) / V;
#pragma omp parallel for schedule(static)
    for (int g = 0; g < ngroups; ++g) {
        const double *apg = ap + (size_t)g * s * s * V; /* group stride s*s*V */
        double *bpg = bp + (size_t)g * m * n * V;       /* group stride m*n*V */
        mkl_dtrsm_compact(MKL_COL_MAJOR, c.side == 'L' ? MKL_LEFT : MKL_RIGHT,
                          c.uplo == 'U' ? MKL_UPPER : MKL_LOWER,
                          c.transa == 'N' ? MKL_NOTRANS : MKL_TRANS,
                          c.diag == 'U' ? MKL_UNIT : MKL_NONUNIT, m, n, 1.0, apg, s, bpg,
                          m, fmt, V);
    }
}
#endif

/* Per-matrix solve of a standard-layout copy of the right-hand sides in place
 * with the linked library's dtrsm, from an OpenMP loop of the benchmark's
 * thread count. */
void solve_unbatched(const TrsmCase &c, const MatrixPool &A, double *b)
{
    const size_t stride = (size_t)c.m() * c.n();
#pragma omp parallel for schedule(static)
    for (int v = 0; v < A.count(); ++v)
        blas_dtrsm(c, A[v], b + (size_t)v * stride);
}

/* The forward error of a compact solve, read from the buffer its last pass
 * left behind: unpack and measure against the known solution. */
double compact_error(const TrsmCase &c, const PackedImage &b, const double *bw, int nmat)
{
    MatrixPool X(nmat, c.m(), c.n());
    b.unpack_into(X, bw);
    return forward_error(X.data(), c.m(), c.n(), nmat);
}

const char *case_line(const TrsmCase &c)
{
    static char buf[96];
    std::snprintf(buf, sizeof buf, "side=%c uplo=%c transa=%c diag=%c  nrhs=%d", c.side,
                  c.uplo, c.transa, c.diag, c.nrhs);
    return buf;
}

/* Single-kernel size sweep: solve a pre-packed pool with cbk at each order in
 * [nmin, nmax] (step stride) and print throughput only -- no cross-check, so it
 * stays cheap and isolates the kernel. The point is the staircase: orders that
 * are / are not a multiple of the interleave width V. The raw best-pass time
 * is printed next to the derived rates: it is the quantity they come from
 * (rate = nmat / time), so a total near the timer granularity flags a noisy
 * row -- raise nmat until it is comfortably above the clock resolution. */
void run_sweep(TrsmCase c, int nmat, int reps, int nmin, int nmax, int stride, int V,
               int nthreads)
{
    std::printf("Triangular solve size sweep: dtrsm_compact only (throughput, no "
                "cross-check)\n");
    std::printf("matrices=%d  reps=%d  %s  simdlen=%d (kernels compiled for %s)  "
                "OpenMP threads=%d  (col-major, pre-packed)\n\n",
                nmat, reps, case_line(c), V, compiled_isa, nthreads);
    std::printf("   n |  total (s) | cbk GFLOP/s |   cbk mat/s\n");
    std::printf("-----+------------+-------------+-------------\n");

    for (int s = nmin; s <= nmax; s += stride) {
        c.s = s;
        const MatrixPool A = make_pool(c, nmat);
        MatrixPool B(nmat, c.m(), c.n());
        fill_rhs(c, A, B);
        const PackedImage a(A, V), b(B, V);
        auto bw = b.work();

        auto restore = [&] { b.restore_into(bw.data()); };
        const double t = best_time(reps, restore,
                                   [&] { solve_cbk(c, a.p.data(), bw.data(), V, nmat); });
        std::printf("%4d | %10.3e | %11.2f | %11.2e\n", s, t, nmat * c.gflop() / t,
                    nmat / t);
    }
    std::printf("-----+------------+-------------+-------------\n");
}

} /* anonymous namespace */

int main(int argc, char **argv)
{
    TrsmCase c;
    /* --side=L|R --uplo=U|L --transa=N|T --diag=N|U, one letter each, either
     * case; 'C' is 'T' for the real types. */
    const auto take = [&](const char *arg, const char *key, char &dst,
                          const char *allowed) {
        const size_t k = std::strlen(key);
        if (std::strncmp(arg, key, k) != 0) return false;
        char v = (char)std::toupper((unsigned char)arg[k]);
        if (v == 'C') v = 'T';
        if (arg[k] == 0 || arg[k + 1] != 0 || !std::strchr(allowed, v)) {
            std::printf("usage: %s takes one of %s\n", key, allowed);
            std::exit(1);
        }
        dst = v;
        return true;
    };
    const auto extra = [&](const char *arg) {
        return take(arg, "--side=", c.side, "LR") || take(arg, "--uplo=", c.uplo, "UL") ||
               take(arg, "--transa=", c.transa, "NT") ||
               take(arg, "--diag=", c.diag, "NU");
    };
    const BenchArgs args(argc, argv, "bench_trsm_compact", extra,
                         " [--side=L|R] [--uplo=U|L] [--transa=N|T] [--diag=N|U]");
    const int nmat = args.nmat, reps = args.reps, V = args.V;
    c.nrhs = args.nrhs;

#ifdef CBK_BENCH_WITH_MKL
    /* MKL's kernel runs a format no wider than the host's native one (the
     * default is exactly that; only a --simdlen above it is refused). Pin
     * MKL's internal threading: the OpenMP outer loop is the only parallelism. */
    const MKL_COMPACT_PACK fmt = format_for_vlen<double>(V);
    check(V <= vlen_for_format<double>(mkl_get_format_compact()),
          "requested --simdlen exceeds the host's native SIMD width");
    mkl_set_num_threads(1);
#endif
    const int nthreads = omp_threads();

    if (args.sweep) {
        run_sweep(c, nmat, reps, args.sweep_min, args.sweep_max, args.sweep_step, V,
                  nthreads);
        return 0;
    }

#ifdef CBK_BENCH_WITH_MKL
    std::printf("Triangular solve throughput: dtrsm_compact vs mkl_dtrsm_compact vs "
                "per-matrix BLAS dtrsm (MKL)\n");
#else
    std::printf("Triangular solve throughput: dtrsm_compact vs per-matrix BLAS dtrsm\n");
#endif
    std::printf(
        "matrices=%d  reps=%d  %s  simdlen=%d (host %s; kernels compiled for %s)  "
        "OpenMP threads=%d  (col-major, alpha=1, pre-packed)\n\n",
        nmat, reps, case_line(c), V, simdlen_name(V), compiled_isa, nthreads);
    /* Throughput as matrices/second (scientific) so it stays legible across the
     * whole size range; the speedup ratios show where the wins come from. The
     * error columns are forward errors against the known solution. */
#ifdef CBK_BENCH_WITH_MKL
    std::printf(
        "   n | cbk GFLOP/s |   cbk mat/s |   mkl mat/s |  blas mat/s | cbk/blas | "
        "mkl/blas | cbk/mkl |  cbk err |  mkl err | blas err\n");
    std::printf(
        "-----+-------------+-------------+-------------+-------------+----------+"
        "----------+---------+----------+----------+---------\n");
#else
    std::printf("   n | cbk GFLOP/s |   cbk mat/s |  blas mat/s | cbk/blas |  cbk err | "
                "blas err\n");
    std::printf("-----+-------------+-------------+-------------+----------+----------+"
                "---------\n");
#endif

    double log_speed_vs_blas = 0.0, log_speed_vs_mkl = 0.0;
    for (int s : bench_sizes) {
        c.s = s;
        const MatrixPool A = make_pool(c, nmat);
        MatrixPool B(nmat, c.m(), c.n());
        fill_rhs(c, A, B);

        /* the pristine packed operands, and the one working copy of B the
         * compact paths overwrite (restored, untimed, before each pass) */
        const PackedImage a(A, V), b(B, V);
        auto bw = b.work();
        aligned_vector<double> pool_work; /* standard-layout copy (aligned like pool) */

        auto restore = [&] { b.restore_into(bw.data()); };
        const double t_cbk = best_time(
            reps, restore, [&] { solve_cbk(c, a.p.data(), bw.data(), V, nmat); });
        const double err_cbk = compact_error(c, b, bw.data(), nmat);
#ifdef CBK_BENCH_WITH_MKL
        const double t_mkl = best_time(
            reps, restore, [&] { solve_mkl(c, a.p.data(), bw.data(), V, fmt, nmat); });
        const double err_mkl = compact_error(c, b, bw.data(), nmat);
#endif
        const double t_blas = best_time(
            reps, [&] { pool_work = B.storage(); },
            [&] { solve_unbatched(c, A, pool_work.data()); });
        const double err_blas = forward_error(pool_work.data(), c.m(), c.n(), nmat);

        check(err_cbk <= 1e-9, "compact solve recovers the known solution");
        check(err_blas <= 1e-9, "per-matrix BLAS solve recovers the known solution");

        const double sp_blas = t_blas / t_cbk; /* cbk speedup over BLAS */
        const double gflops_cbk = nmat * c.gflop() / t_cbk;
        log_speed_vs_blas += std::log(sp_blas);
#ifdef CBK_BENCH_WITH_MKL
        check(err_mkl <= 1e-9, "MKL compact solve recovers the known solution");
        const double sp_mkl_blas = t_blas / t_mkl; /* MKL speedup over BLAS */
        const double sp_mkl = t_mkl / t_cbk;       /* cbk speedup over MKL  */
        log_speed_vs_mkl += std::log(sp_mkl);
        std::printf(
            "%4d | %11.2f | %11.2e | %11.2e | %11.2e | %7.2fx | %7.2fx | %6.2fx | "
            "%.2e | %.2e | %.2e\n",
            s, gflops_cbk, nmat / t_cbk, nmat / t_mkl, nmat / t_blas, sp_blas,
            sp_mkl_blas, sp_mkl, err_cbk, err_mkl, err_blas);
#else
        std::printf("%4d | %11.2f | %11.2e | %11.2e | %7.2fx | %.2e | %.2e\n", s,
                    gflops_cbk, nmat / t_cbk, nmat / t_blas, sp_blas, err_cbk, err_blas);
#endif
    }

#ifdef CBK_BENCH_WITH_MKL
    std::printf(
        "-----+-------------+-------------+-------------+-------------+----------+"
        "----------+---------+----------+----------+---------\n");
    std::printf("geometric-mean speedup (cbk compact vs mkl compact): %.2fx\n",
                std::exp(log_speed_vs_mkl / bench_sizes.size()));
#else
    std::printf("-----+-------------+-------------+-------------+----------+----------+"
                "---------\n");
#endif
    std::printf("geometric-mean speedup (cbk compact vs per-matrix BLAS): %.2fx\n",
                std::exp(log_speed_vs_blas / bench_sizes.size()));
    return 0;
}
