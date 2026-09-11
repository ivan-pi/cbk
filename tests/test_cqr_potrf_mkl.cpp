/* test_cqr_potrf_mkl.cpp
 *
 * Validation of cqr_mkl_?potrf_compact against real Intel MKL and dense LAPACK,
 * through the genuine MKL Compact pipeline (mkl_dgepack_compact /
 * mkl_dgeunpack_compact). This is the design document's correctness gate against
 * standard dense LAPACK (cqr_mkl_dpotrf_compact_design.md section 7).
 *
 * Suite 1 (section 7.1) -- Factorization invariants vs dense LAPACK, for each
 *   (uplo, layout): a random SPD batch is factored by cqr_mkl_dpotrf_compact,
 *   unpacked, and per matrix checked against the LAPACK Cholesky contract:
 *     - reconstruction residual || L L^T - A ||_1 / ||A||_1 <= 20 n eps
 *       (U^T U for upper),
 *     - the strictly-opposite triangle is bit-for-bit unchanged from the input,
 *     - since the SPD factor is unique, elementwise vs LAPACKE_dpotrf,
 *       || L_cqr - L_lapack ||_1 / ||L_lapack||_1 <= 20 n eps.
 *
 * Suite 2 (section 7.2) -- Cross-check vs mkl_dpotrf_compact: the same packed
 *   batch factored by both, compact buffers compared elementwise at a small
 *   fixed tolerance (1e-9). Run for both layouts and both uplo.
 *
 * Suite 3 (section 7.3) -- End-to-end SPD solve AX = B: B = A X for known X;
 *   cqr_mkl<T>::potrf('L') -> mkl<T>::trsm('L','L','N') ->
 *   mkl<T>::trsm('L','L','T') must recover X. Gates forward error and the
 *   system residual at 100 n eps.
 *
 * Build: needs Intel MKL (headers + libmkl_rt); wired up by CMakeLists.txt.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include "test_mkl_util.hpp" /* cqr_mkl<T>, mkl<T>, lapack<T> + the MKL-free helpers */

#include <cstdio>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

using namespace cqr::test;

namespace {

/* ---------------- Suite 1: invariants vs dense LAPACK ------------------ */

template <class T>
int suite1(MKL_LAYOUT layout, MKL_UPLO uplo, int nm, int n, double cond)
{
    const double eps = std::numeric_limits<T>::epsilon();
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt);
    const bool row = (layout == MKL_ROW_MAJOR);
    const bool up = (uplo == MKL_UPPER);
    const char ul = up ? 'U' : 'L';

    MatrixBatch<T> A(nm, n, n);
    for (int v = 0; v < nm; ++v)
        gen_spd(A.view(v), cond);

    /* pack the full symmetric A, factor with the routine under test, unpack */
    auto Ap = A.base_ptrs();
    MKL_INT sz_a = mkl<T>::get_size(n, n, fmt, nm);
    auto ap_buf = cqr::detail::mkl_alloc_bytes<T>(sz_a);
    T *ap = ap_buf.get();
    mkl<T>::gepack(layout, n, n, Ap.data(), n, ap, n, fmt, nm);

    MKL_INT info = 99;
    cqr_mkl<T>::potrf(layout, uplo, n, ap, n, &info, fmt, nm);

    MatrixBatch<T> H(nm, n, n);
    auto Hp = H.base_ptrs();
    mkl<T>::geunpack(layout, n, n, Hp.data(), n, ap, n, fmt, nm);

    int fails = 0;
    if (info != 0) {
        ++fails;
        std::printf("    info = %ld (expected 0)\n", (long)info);
    }

    double worst_res = 0, worst_el = 0, worst_untouched = 0;
    std::vector<T> Lref(A.stride());
    for (int v = 0; v < nm; ++v) {
        /* the factor is stored in `layout`; the input and the residual are the
         * column-major dense side */
        const auto Hm = H.view(v, row);
        const auto Am = A.view(v);

        /* reconstruction residual and untouched-triangle check */
        std::vector<T> R(A.stride(), 0.0);
        const auto Res = mat_view(R.data(), n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0;
                int lmax = std::min(i, j);
                if (!up) /* A = L L^T */
                    for (int l = 0; l <= lmax; ++l)
                        s += Hm(i, l) * Hm(j, l);
                else /* A = U^T U */
                    for (int l = 0; l <= lmax; ++l)
                        s += Hm(l, i) * Hm(l, j);
                Res(i, j) = (T)s - Am(i, j);
            }
        worst_res = std::max(worst_res, norm1(Res) / std::max(norm1(Am), norm_floor));

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                const bool named = up ? (i <= j) : (i >= j);
                if (!named)
                    worst_untouched =
                        std::max<double>(worst_untouched, std::abs(Hm(i, j) - Am(i, j)));
            }

        /* elementwise vs LAPACKE_dpotrf (unique SPD factor -> a sharp signal) */
        std::copy(A[v], A[v] + A.stride(), Lref.begin());
        lapack<T>::potrf(LAPACK_COL_MAJOR, ul, n, Lref.data(), n);
        const auto Lr = mat_view(Lref.data(), n, n);
        double el = 0, lref_norm = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                const bool named = up ? (i <= j) : (i >= j);
                if (named) el = std::max<double>(el, std::abs(Hm(i, j) - Lr(i, j)));
            }
        lref_norm = norm1(Lr); /* triangular factor L1 norm */
        worst_el = std::max(worst_el, el / std::max(lref_norm, norm_floor));
    }

    const double rtol = 20.0 * n * eps;
    bool ok = (worst_res <= rtol) && (worst_el <= rtol) && (worst_untouched == 0.0);
    fails += !ok;
    std::printf("  [suite1] %s uplo=%c V=%-2d nm=%-2d n=%-3d cond=%.0f | res %.2e "
                "el %.2e (%.1e) untouched %.0e %s\n",
                row ? "row" : "col", ul, V, nm, n, cond, worst_res, worst_el, rtol,
                worst_untouched, ok ? "OK" : "FAIL");
    return fails;
}

/* ---------------- Suite 2: cross-check vs mkl_dpotrf_compact ----------- */

template <class T> int suite2(MKL_LAYOUT layout, MKL_UPLO uplo, int nm, int n)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt);
    const bool row = (layout == MKL_ROW_MAJOR);
    const char ul = (uplo == MKL_UPPER) ? 'U' : 'L';

    MatrixBatch<T> A(nm, n, n);
    for (int v = 0; v < nm; ++v)
        gen_spd(A.view(v), 0.0);
    auto Ap = A.base_ptrs();

    MKL_INT sz_a = mkl<T>::get_size(n, n, fmt, nm);
    auto ap1 = cqr::detail::mkl_alloc_bytes<T>(sz_a);
    auto ap2 = cqr::detail::mkl_alloc_bytes<T>(sz_a);
    mkl<T>::gepack(layout, n, n, Ap.data(), n, ap1.get(), n, fmt, nm);
    mkl<T>::gepack(layout, n, n, Ap.data(), n, ap2.get(), n, fmt, nm);

    MKL_INT info = 0;
    cqr_mkl<T>::potrf(layout, uplo, n, ap1.get(), n, &info, fmt, nm);
    mkl<T>::potrf(layout, uplo, n, ap2.get(), n, &info, fmt, nm);

    /* compare the two compact buffers elementwise (same input, same convention).
     * The two implementations do not share an arithmetic order, so exact
     * agreement is not required; the tolerance confirms the same factor. */
    double da = max_abs_diff(ap1.get(), ap2.get(), (size_t)sz_a / sizeof(double));
    const double tol = cross_tol<T>();
    bool ok = (da <= tol);
    std::printf("  [suite2] %s uplo=%c V=%-2d nm=%-2d n=%-3d | max|ap-mkl| %.2e "
                "(tol %.0e) %s\n",
                row ? "row" : "col", ul, V, nm, n, da, tol, ok ? "OK" : "FAIL");
    return !ok;
}

/* ---------------- Suite 3: end-to-end AX = B (col-major lower) --------- */

template <class T> int suite3(int nm, int n, int nrhs)
{
    const double eps = std::numeric_limits<T>::epsilon();
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt);
    const std::vector<T> Xs = known_solution<T>(n, nrhs);
    const auto X = mat_view(Xs.data(), n, nrhs);

    MatrixBatch<T> A(nm, n, n), B(nm, n, nrhs);
    for (int v = 0; v < nm; ++v) {
        gen_spd(A.view(v), 0.0);
        matmul(A.view(v), X, B.view(v)); /* B = A X */
    }
    auto Ap = A.base_ptrs();
    auto Bp = B.base_ptrs();

    MKL_INT sz_a = mkl<T>::get_size(n, n, fmt, nm);
    MKL_INT sz_b = mkl<T>::get_size(n, nrhs, fmt, nm);
    auto ap_buf = cqr::detail::mkl_alloc_bytes<T>(sz_a);
    auto bp_buf = cqr::detail::mkl_alloc_bytes<T>(sz_b);
    T *ap = ap_buf.get(), *bp = bp_buf.get();
    mkl<T>::gepack(MKL_COL_MAJOR, n, n, Ap.data(), n, ap, n, fmt, nm);
    mkl<T>::gepack(MKL_COL_MAJOR, n, nrhs, Bp.data(), n, bp, n, fmt, nm);

    MKL_INT info = 99;

    /* 1. our compact Cholesky: A -> L (lower). No workspace. */
    cqr_mkl<T>::potrf(MKL_COL_MAJOR, MKL_LOWER, n, ap, n, &info, fmt, nm);

    /* 2. forward then back substitution with MKL's compact trsm:
     *    B := L^{-1} B, then B := L^{-T} B = A^{-1} B = X. */
    mkl<T>::trsm(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, MKL_NOTRANS, MKL_NONUNIT, n, nrhs,
                 T(1), ap, n, bp, n, fmt, nm);
    mkl<T>::trsm(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, MKL_TRANS, MKL_NONUNIT, n, nrhs,
                 T(1), ap, n, bp, n, fmt, nm);

    MatrixBatch<T> Xhat(nm, n, nrhs);
    auto Op = Xhat.base_ptrs();
    mkl<T>::geunpack(MKL_COL_MAJOR, n, nrhs, Op.data(), n, bp, n, fmt, nm);

    int fails = 0;
    if (info != 0) {
        ++fails;
        std::printf("    info = %ld (expected 0)\n", (long)info);
    }
    const auto [worst_fwd, worst_res] = solve_errors(A, B, Xhat, X);
    const double rtol = 100.0 * n * eps;
    bool ok = (worst_fwd <= rtol && worst_res <= rtol);
    fails += !ok;
    std::printf(
        "  [suite3] V=%-2d nm=%-2d n=%-3d nrhs=%d | fwd %.2e res %.2e (rtol %.2e) %s\n",
        V, nm, n, nrhs, worst_fwd, worst_res, rtol, ok ? "OK" : "FAIL");
    return fails;
}

} /* anonymous namespace */

template <class T> int run_suites()
{
    std::printf("\n== %s: MKL compact format = %d, V = %d ==\n", compact<T>::name,
                (int)mkl_get_format_compact(), mkl<T>::vlen(mkl_get_format_compact()));

    int fails = 0;

    /* Suite 1: invariants vs dense LAPACK, over (uplo, layout), sizes and cond */
    const MKL_LAYOUT lays[] = {MKL_COL_MAJOR, MKL_ROW_MAJOR};
    const MKL_UPLO ups[] = {MKL_LOWER, MKL_UPPER};
    for (MKL_LAYOUT L : lays)
        for (MKL_UPLO U : ups) {
            fails += suite1<T>(L, U, 8, 30, 0.0);
            fails += suite1<T>(L, U, 16, 60, 0.0);
            fails += suite1<T>(L, U, 11, 43, 0.0); /* padded partial group */
            fails += suite1<T>(L, U, 8, 40, 2.0);  /* dynamic range (cond knob) */
        }
    fails += suite1<T>(MKL_COL_MAJOR, MKL_LOWER, 8, 128, 0.0);
    fails += suite1<T>(MKL_COL_MAJOR, MKL_LOWER, 4, 3, 0.0); /* smallest, padded */

    /* Suite 2: cross-check vs mkl_dpotrf_compact, both layouts and uplo */
    for (MKL_LAYOUT L : lays)
        for (MKL_UPLO U : ups) {
            fails += suite2<T>(L, U, 8, 30);
            fails += suite2<T>(L, U, 11, 40); /* padded partial group */
        }

    /* Suite 3: end-to-end SPD solver */
    fails += suite3<T>(8, 30, 5);
    fails += suite3<T>(16, 60, 4);
    fails += suite3<T>(7, 32, 6); /* padded partial group */

    return fails;
}

int main()
{
    const int fails = run_suites<double>() + run_suites<float>();
    return finish(fails);
}
