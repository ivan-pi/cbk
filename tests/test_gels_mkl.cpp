/* test_gels_mkl.cpp
 *
 * Validation of cbk_?gels_compact against real Intel MKL and dense LAPACK,
 * through the genuine MKL Compact pipeline (mkl_?gepack_compact /
 * mkl_?geunpack_compact) -- the "hard correctness gate against standard dense
 * LAPACK equivalents" of cbk_dgels_compact_design.md section 7.
 *
 * Suite 1 (section 7.1) -- Solution vs dense LAPACKE_?gels:
 *   For every (layout, trans) over square, tall and wide A -- so all four
 *   least-squares / minimum-norm cases in both layouts -- a random batch is
 *   solved by cbk_?gels_compact (packed in `layout`) and per matrix by
 *   LAPACKE_?gels on the same input, column-major. Gated: the forward error of X
 *   vs LAPACK's (relative, 100 * max(m,n) * eps); the defining property formed
 *   independently -- in the least-squares case the residual sums of squares in
 *   rows n..m-1 of B against ||B - op(A) X||^2, in the minimum-norm case the
 *   residual op(A) X - B; the workspace query against mkl_?get_size_compact;
 *   and the factorization left in ap and work, elementwise vs LAPACKE_?geqrf
 *   (m >= n) or LAPACKE_?gelqf (m < n), which pins the ?gelqf storage
 *   convention of the wide case.
 *
 * Suite 2 (section 7.2) -- Cross-check vs the three-step compact pipeline:
 *   On the same packed square batch with B = A X for a known X,
 *   cbk_?gels_compact and mkl_?geqrf_compact -> cbk_?ormqr_compact ->
 *   mkl_?trsm_compact must agree: the factorization in ap and the tau (gels's
 *   work vs geqrf's taup) elementwise at the cross-check tolerance, and the two
 *   solutions with each other and with X at the solve gate (100 * n * eps,
 *   relative), since the two triangular solves order their arithmetic
 *   differently.
 *
 * Build: needs Intel MKL (headers + libmkl_rt); wired up by CMakeLists.txt.
 *
 * Assisted-by: Claude:claude-fable-5
 */

#include "test_mkl_util.hpp" /* compat<T>, mkl<T>, lapack<T> + the MKL-free helpers */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

using namespace cbk::test;

namespace {

/* ---------------- Suite 1: solution vs dense LAPACKE_?gels ------------- */

template <class T>
int suite1(MKL_LAYOUT layout, char trans, int nm, int m, int n, int nrhs)
{
    const double eps = std::numeric_limits<T>::epsilon();
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt);
    const bool row = (layout == MKL_ROW_MAJOR);
    const bool tran = (trans == 'T');
    const bool tall = (m >= n);
    const int p = std::max(m, n), q = std::min(m, n);
    const bool overdet = (tall != tran);
    const int rows_op = tran ? n : m,
              cols_op = tran ? m : n;                      /* op(A): rows_op x cols_op */
    const MKL_INT lda = row ? n : m, ldb = row ? nrhs : p; /* dense == compact lds here */

    /* random A with a boosted diagonal (well-conditioned op(A)) and random B,
     * column-major: the dense side every check and reference reads */
    MatrixBatch<T> A(nm, m, n), B(nm, p, nrhs);
    for (int v = 0; v < nm; ++v) {
        gen_boosted(A.view(v));
        for (size_t e = 0; e < B.stride(); ++e)
            B[v][e] = frand<T>();
    }

    /* mkl_?ge(un)pack_compact read/write the dense side in `layout` too, so the
     * row-major runs pack from (and unpack to) row-major staging copies */
    MatrixBatch<T> Asrc(row ? nm : 0, m, n), Bsrc(row ? nm : 0, p, nrhs);
    for (int v = 0; v < Asrc.count(); ++v) { /* same matrices, the other layout */
        copy_matrix(A.view(v), Asrc.view(v, /*rowmajor=*/true));
        copy_matrix(B.view(v), Bsrc.view(v, /*rowmajor=*/true));
    }
    auto Ap = row ? Asrc.base_ptrs() : A.base_ptrs();
    auto Bp = row ? Bsrc.base_ptrs() : B.base_ptrs();

    /* pack, query the workspace, solve */
    auto ap_buf = cbk::detail::mkl_alloc_bytes<T>(mkl<T>::get_size(m, n, fmt, nm));
    auto bp_buf = cbk::detail::mkl_alloc_bytes<T>(mkl<T>::get_size(p, nrhs, fmt, nm));
    T *ap = ap_buf.get(), *bp = bp_buf.get();
    mkl<T>::gepack(layout, m, n, Ap.data(), lda, ap, lda, fmt, nm);
    mkl<T>::gepack(layout, p, nrhs, Bp.data(), ldb, bp, ldb, fmt, nm);

    MKL_INT info = 99;
    T wq = -1;
    compat<T>::gels(layout, trans, m, n, nrhs, ap, lda, bp, ldb, &wq, -1, &info, fmt, nm);
    /* the documented requirement: a compact tau buffer for the batch, in scalars */
    const MKL_INT lwork_want =
        std::max<MKL_INT>(1, mkl<T>::get_size(q, 1, fmt, nm) / (MKL_INT)sizeof(T));
    const bool ok_query = (info == 0) && ((MKL_INT)wq == lwork_want);
    std::vector<T> work((size_t)std::max<MKL_INT>((MKL_INT)wq, 1));
    compat<T>::gels(layout, trans, m, n, nrhs, ap, lda, bp, ldb, work.data(),
                    (MKL_INT)work.size(), &info, fmt, nm);

    /* unpack X and the factorization in `layout`, then stage back to
     * column-major for the checks; tau is a vector, layout-agnostic */
    MatrixBatch<T> Xout(nm, p, nrhs), Hout(nm, m, n), tau(nm, q, 1);
    auto Xp = Xout.base_ptrs();
    auto Hp = Hout.base_ptrs();
    auto Tp = tau.base_ptrs();
    mkl<T>::geunpack(layout, p, nrhs, Xp.data(), ldb, bp, ldb, fmt, nm);
    mkl<T>::geunpack(layout, m, n, Hp.data(), lda, ap, lda, fmt, nm);
    mkl<T>::geunpack(MKL_COL_MAJOR, q, 1, Tp.data(), q, work.data(), q, fmt, nm);
    MatrixBatch<T> Xhat(nm, p, nrhs), H(nm, m, n);
    if (row) {
        for (int v = 0; v < nm; ++v) {
            copy_matrix(Xout.view(v, /*rowmajor=*/true), Xhat.view(v));
            copy_matrix(Hout.view(v, /*rowmajor=*/true), H.view(v));
        }
    }
    else {
        Xhat = Xout;
        H = Hout;
    }

    int fails = 0;
    if (info != 0) {
        ++fails;
        std::printf("    info = %ld (expected 0)\n", (long)info);
    }

    double worst_fwd = 0, worst_prop = 0, worst_fac = 0;
    std::vector<T> Arefs(A.stride()), Xrefs(B.stride()), trefs(q),
        Rs((size_t)rows_op * nrhs);
    const auto Aref = mat_view(Arefs.data(), m, n);
    const auto Xref = mat_view(Xrefs.data(), p, nrhs);
    const auto R = mat_view(Rs.data(), rows_op, nrhs);
    for (int v = 0; v < nm; ++v) {
        const auto Av = A.view(v);
        const auto Aop = tran ? Av.transposed() : Av;
        const auto Bin = leading(B.view(v), rows_op, nrhs);
        const auto X = leading(Xhat.view(v), cols_op, nrhs);

        /* dense reference solve */
        std::copy(A[v], A[v] + A.stride(), Arefs.begin());
        std::copy(B[v], B[v] + B.stride(), Xrefs.begin());
        if (lapack<T>::gels(LAPACK_COL_MAJOR, trans, m, n, nrhs, Arefs.data(), m,
                            Xrefs.data(), p) != 0)
            ++fails;

        /* forward error over the solution rows, relative to ||X_lapack||_1 */
        const auto Xr = leading(Xref, cols_op, nrhs);
        double fwd = 0;
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < cols_op; ++i)
                fwd = std::max(fwd, (double)std::abs(X(i, j) - Xr(i, j)));
        worst_fwd = std::max(worst_fwd, fwd / std::max(norm1(Xr), norm_floor));

        /* R = op(A) X - B, formed independently */
        matmul(Aop, X, R);
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < rows_op; ++i)
                R(i, j) -= Bin(i, j);
        if (overdet) {
            /* rows cols_op..rows_op-1 of X carry the residual: their squared
             * column norms are the residual sums of squares (relative to
             * ||b_j||^2: a square system has no residual rows and none) */
            for (int j = 0; j < nrhs; ++j) {
                double rss = 0, rss_x = 0, bb = 0;
                for (int i = 0; i < rows_op; ++i) {
                    rss += (double)R(i, j) * R(i, j);
                    bb += (double)Bin(i, j) * Bin(i, j);
                }
                for (int i = cols_op; i < rows_op; ++i)
                    rss_x += (double)Xhat(v, i, j) * Xhat(v, i, j);
                worst_prop = std::max(worst_prop,
                                      std::abs(rss - rss_x) / std::max(bb, norm_floor));
            }
        }
        else {
            /* op(A) X = B */
            const double scale =
                std::max(norm1(Av), norm_floor) * std::max(norm1(X), norm_floor);
            for (size_t e = 0; e < Rs.size(); ++e)
                worst_prop = std::max(worst_prop, std::abs((double)Rs[e]) / scale);
        }

        /* the factorization left behind: QR (tall) or LQ (wide) of A, vs LAPACK */
        std::copy(A[v], A[v] + A.stride(), Arefs.begin());
        if (tall)
            lapack<T>::geqrf(LAPACK_COL_MAJOR, m, n, Arefs.data(), m, trefs.data());
        else
            lapack<T>::gelqf(LAPACK_COL_MAJOR, m, n, Arefs.data(), m, trefs.data());
        const double fac = std::max(max_abs_diff(H[v], Arefs.data(), A.stride()),
                                    max_abs_diff(tau[v], trefs.data(), (size_t)q));
        worst_fac = std::max(worst_fac, fac / std::max(norm1(Av), norm_floor));
    }

    /* one gate serves all three: the forward error is cond(op(A))-dependent but
     * op(A) is well conditioned here; the residual checks are backward-stable;
     * the factorization compares an unblocked kernel to LAPACK's blocked one
     * (same math, different order) */
    const double rtol = 100.0 * p * eps;
    const bool ok =
        (worst_fwd <= rtol) && (worst_prop <= rtol) && (worst_fac <= rtol) && ok_query;
    fails += !ok;
    std::printf("  [suite1] %s trans=%c V=%-2d nm=%-2d m=%-3d n=%-3d nrhs=%d %-6s | fwd "
                "%.2e (%.1e) %s %.2e fac %.2e %s%s\n",
                row ? "row" : "col", trans, V, nm, m, n, nrhs,
                overdet ? "lstsq" : "minnrm", worst_fwd, rtol, overdet ? "rss" : "res",
                worst_prop, worst_fac, ok ? "OK" : "FAIL",
                ok_query ? "" : " (workspace query)");
    return fails;
}

/* ---------------- Suite 2: cross-check vs the three-step pipeline ------ */

template <class T> int suite2(int nm, int n, int nrhs)
{
    const double eps = std::numeric_limits<T>::epsilon();
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt);
    const std::vector<T> Xs = known_solution<T>(n, nrhs);
    const auto X = mat_view(Xs.data(), n, nrhs);

    MatrixBatch<T> A(nm, n, n), B(nm, n, nrhs);
    for (int v = 0; v < nm; ++v) {
        gen_boosted(A.view(v));
        matmul(A.view(v), X, B.view(v)); /* B = A X */
    }
    auto Ap = A.base_ptrs();
    auto Bp = B.base_ptrs();

    const MKL_INT sz_a = mkl<T>::get_size(n, n, fmt, nm);
    const MKL_INT sz_t = mkl<T>::get_size(n, 1, fmt, nm);
    const MKL_INT sz_b = mkl<T>::get_size(n, nrhs, fmt, nm);
    auto ap1 = cbk::detail::mkl_alloc_bytes<T>(sz_a),
         ap2 = cbk::detail::mkl_alloc_bytes<T>(sz_a);
    auto bp1 = cbk::detail::mkl_alloc_bytes<T>(sz_b),
         bp2 = cbk::detail::mkl_alloc_bytes<T>(sz_b);
    auto tp2 = cbk::detail::mkl_alloc_bytes<T>(sz_t);
    mkl<T>::gepack(MKL_COL_MAJOR, n, n, Ap.data(), n, ap1.get(), n, fmt, nm);
    mkl<T>::gepack(MKL_COL_MAJOR, n, n, Ap.data(), n, ap2.get(), n, fmt, nm);
    mkl<T>::gepack(MKL_COL_MAJOR, n, nrhs, Bp.data(), n, bp1.get(), n, fmt, nm);
    mkl<T>::gepack(MKL_COL_MAJOR, n, nrhs, Bp.data(), n, bp2.get(), n, fmt, nm);

    MKL_INT info = 0;

    /* 1. the one-shot solve; its work is the tau of the factorization */
    T wq;
    compat<T>::gels(MKL_COL_MAJOR, 'N', n, n, nrhs, ap1.get(), n, bp1.get(), n, &wq, -1,
                    &info, fmt, nm);
    std::vector<T> work((size_t)std::max<MKL_INT>((MKL_INT)wq, 1));
    compat<T>::gels(MKL_COL_MAJOR, 'N', n, n, nrhs, ap1.get(), n, bp1.get(), n,
                    work.data(), (MKL_INT)work.size(), &info, fmt, nm);

    /* 2. the pipeline: MKL's geqrf (its own workspace), cbk's ormqr, MKL's trsm */
    T wq_mkl;
    mkl<T>::geqrf(MKL_COL_MAJOR, n, n, ap2.get(), n, tp2.get(), &wq_mkl, -1, &info, fmt,
                  nm);
    std::vector<T> work_mkl((size_t)std::max<MKL_INT>((MKL_INT)wq_mkl, 1));
    mkl<T>::geqrf(MKL_COL_MAJOR, n, n, ap2.get(), n, tp2.get(), work_mkl.data(),
                  (MKL_INT)work_mkl.size(), &info, fmt, nm);
    T dummy;
    compat<T>::ormqr(MKL_COL_MAJOR, 'L', 'T', n, nrhs, n, ap2.get(), n, tp2.get(),
                     bp2.get(), n, &dummy, 1, &info, fmt, nm);
    mkl<T>::trsm(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS, MKL_NONUNIT, n, nrhs,
                 T(1), ap2.get(), n, bp2.get(), n, fmt, nm);

    /* the factorizations elementwise; the solutions relative to ||X||_1, and
     * gels's against the known X */
    const size_t na = (size_t)sz_a / sizeof(T), nb = (size_t)sz_b / sizeof(T);
    const size_t nt = (size_t)sz_t / sizeof(T);
    const double nx = norm1(X);
    const double da = max_abs_diff(ap1.get(), ap2.get(), na);
    const double dt = max_abs_diff(work.data(), tp2.get(), std::min(nt, work.size()));
    const double dx = max_abs_diff(bp1.get(), bp2.get(), nb) / nx;
    MatrixBatch<T> Xhat(nm, n, nrhs);
    auto X1p = Xhat.base_ptrs();
    mkl<T>::geunpack(MKL_COL_MAJOR, n, nrhs, X1p.data(), n, bp1.get(), n, fmt, nm);
    const auto [fwd, res] = solve_errors(A, B, Xhat, X);
    const double tol = cross_tol<T>(), rtol = 100.0 * n * eps;
    const bool ok = (da <= tol && dt <= tol && work.size() >= nt) && (dx <= rtol) &&
                    (fwd <= rtol) && (res <= rtol);
    std::printf(
        "  [suite2] V=%-2d nm=%-2d n=%-3d nrhs=%d | ap-mkl %.2e tau-mkl %.2e (tol "
        "%.0e) | X-pipe %.2e fwd %.2e res %.2e (rtol %.1e) %s\n",
        V, nm, n, nrhs, da, dt, tol, dx, fwd, res, rtol, ok ? "OK" : "FAIL");
    return !ok;
}

} /* anonymous namespace */

template <class T> int run_suites()
{
    std::printf("\n== %s: MKL compact format = %d, V = %d ==\n", compact<T>::name,
                (int)mkl_get_format_compact(), mkl<T>::vlen(mkl_get_format_compact()));

    int fails = 0;

    /* Suite 1: every (layout, trans) over square, tall and wide shapes */
    for (MKL_LAYOUT lay : {MKL_COL_MAJOR, MKL_ROW_MAJOR})
        for (char tr : {'N', 'T'}) {
            fails += suite1<T>(lay, tr, 8, 32, 32, 5);  /* square */
            fails += suite1<T>(lay, tr, 8, 64, 20, 4);  /* tall */
            fails += suite1<T>(lay, tr, 8, 20, 64, 4);  /* wide */
            fails += suite1<T>(lay, tr, 11, 43, 17, 3); /* padded partial group */
            fails += suite1<T>(lay, tr, 7, 96, 128, 1); /* single RHS, padded group */
        }

    /* Suite 2: the square solve vs the three-step compact pipeline */
    fails += suite2<T>(8, 30, 5);
    fails += suite2<T>(16, 48, 4);
    fails += suite2<T>(7, 32, 6); /* padded partial group */

    return fails;
}

int main()
{
    const int fails = run_suites<double>() + run_suites<float>();
    return finish(fails);
}
