/* test_geqrf_mkl.cpp
 *
 * Validation of cbk_?geqrf_compact against real Intel MKL, through the
 * genuine MKL Compact pipeline (mkl_dgepack_compact / mkl_dgeunpack_compact):
 * the MKL-side half of cbk_dgeqrf_compact_design.md section 7. The
 * dense-LAPACK invariants of section 7.1 (Q from ?orgqr, residual and
 * orthogonality gates over the structured inputs) run in the portable suite,
 * test_geqrf_compact.cpp, on any LAPACKE stack.
 *
 * Suite 2 (section 7.2) -- Cross-check vs mkl_dgeqrf_compact:
 *   The same packed batch is factored by both cbk_dgeqrf_compact and the
 *   native mkl_dgeqrf_compact; the two compact buffers are compared elementwise.
 *   Run for both column- and row-major (exercises the strided kernel).
 *
 * Suite 3 (section 7.3) -- End-to-end AX = B:
 *   B = A X for known X; cbk_dgeqrf_compact -> compat<T>::ormqr('L','T')
 *   -> mkl_dtrsm_compact must recover X. Gates forward error and residual.
 *
 * Build: needs Intel MKL (headers + libmkl_rt); wired up by CMakeLists.txt.
 *
 * Assisted-by: Claude:claude-opus-4-8 Claude:claude-fable-5
 */

#include "test_mkl_util.hpp" /* compat<T>, mkl<T>, lapack<T> + the shared helpers */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

using namespace cbk::test;

namespace {

/* ---------------- Suite 2: cross-check vs mkl_dgeqrf_compact ----------- */

template <class T> int suite2(MKL_LAYOUT layout, int nm, int m, int n)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt);
    const int k = std::min(m, n);
    const bool row = (layout == MKL_ROW_MAJOR);
    const MKL_INT ldA = row ? n : m, ldc = row ? n : m;

    MatrixBatch<T> A(nm, m, n);
    for (int v = 0; v < nm; ++v)
        gen_boosted(A.view(v));
    /* A was generated column-major; for a row-major run reinterpret the same
     * numbers as a row-major m x n (a genuinely different matrix, still fine). */
    auto Ap = A.base_ptrs();

    MKL_INT sz_a = mkl<T>::get_size(m, n, fmt, nm);
    MKL_INT sz_t = mkl<T>::get_size(k, 1, fmt, nm);
    auto ap1 = cbk::detail::mkl_alloc_bytes<T>(sz_a);
    auto tp1 = cbk::detail::mkl_alloc_bytes<T>(sz_t);
    auto ap2 = cbk::detail::mkl_alloc_bytes<T>(sz_a);
    auto tp2 = cbk::detail::mkl_alloc_bytes<T>(sz_t);
    mkl<T>::gepack(layout, m, n, Ap.data(), ldA, ap1.get(), ldc, fmt, nm);
    mkl<T>::gepack(layout, m, n, Ap.data(), ldA, ap2.get(), ldc, fmt, nm);

    MKL_INT info = 0;
    /* cbk's unblocked kernel needs no scratch (its query returns 1). */
    T wq_cbk;
    compat<T>::geqrf(layout, m, n, ap1.get(), ldc, tp1.get(), &wq_cbk, -1, &info, fmt,
                     nm);
    compat<T>::geqrf(layout, m, n, ap1.get(), ldc, tp1.get(), &wq_cbk, (MKL_INT)wq_cbk,
                     &info, fmt, nm);

    /* MKL's compact geqrf DOES need workspace (~n*V doubles), so query its own
     * lwork -- reusing cbk's lwork=1 would under-size work. Compact routines skip
     * argument checking, so an under-sized work array is undefined behavior: some
     * MKL builds tolerate it, others overrun the heap ("malloc unaligned tcache"). */
    T wq_mkl;
    mkl<T>::geqrf(layout, m, n, ap2.get(), ldc, tp2.get(), &wq_mkl, -1, &info, fmt, nm);
    std::vector<T> work((size_t)std::max<MKL_INT>((MKL_INT)wq_mkl, 1));
    mkl<T>::geqrf(layout, m, n, ap2.get(), ldc, tp2.get(), work.data(),
                  (MKL_INT)work.size(), &info, fmt, nm);

    /* compare the two compact buffers elementwise (same input, same convention) */
    double da = max_abs_diff(ap1.get(), ap2.get(), (size_t)sz_a / sizeof(double));
    double dt = max_abs_diff(tp1.get(), tp2.get(), (size_t)sz_t / sizeof(double));
    /* Same algorithm and sign convention, different arithmetic order: the two
     * agree to a few eps on these O(1) entries (observed ~n*eps/10), so the
     * gate scales with n*eps (design doc 7.2), tighter than the flat cross_tol
     * the other suites use. */
    const double eps = std::numeric_limits<T>::epsilon();
    const double tol = 100.0 * n * eps;
    bool ok = (da <= tol && dt <= tol);
    std::printf("  [suite2] %s V=%-2d nm=%-2d m=%-3d n=%-3d | max|ap-mkl| %.2e "
                "max|tau-mkl| %.2e (tol %.0e) %s\n",
                row ? "row" : "col", V, nm, m, n, da, dt, tol, ok ? "OK" : "FAIL");
    return !ok;
}

/* ---------------- Suite 3: end-to-end AX = B (col-major) --------------- */

template <class T> int suite3(int nm, int n, int nrhs)
{
    const double eps = std::numeric_limits<T>::epsilon();
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt), m = n, k = n;
    const std::vector<T> Xs = known_solution<T>(n, nrhs);
    const auto X = mat_view(Xs.data(), n, nrhs);

    MatrixBatch<T> A(nm, n, n), B(nm, n, nrhs);
    for (int v = 0; v < nm; ++v) {
        gen_boosted(A.view(v));
        matmul(A.view(v), X, B.view(v)); /* B = A X */
    }
    auto Ap = A.base_ptrs();
    auto Bp = B.base_ptrs();

    MKL_INT sz_a = mkl<T>::get_size(m, n, fmt, nm);
    MKL_INT sz_t = mkl<T>::get_size(k, 1, fmt, nm);
    MKL_INT sz_c = mkl<T>::get_size(m, nrhs, fmt, nm);
    auto ap_buf = cbk::detail::mkl_alloc_bytes<T>(sz_a);
    auto taup_buf = cbk::detail::mkl_alloc_bytes<T>(sz_t);
    auto cp_buf = cbk::detail::mkl_alloc_bytes<T>(sz_c);
    T *ap = ap_buf.get(), *taup = taup_buf.get(), *cp = cp_buf.get();
    mkl<T>::gepack(MKL_COL_MAJOR, m, n, Ap.data(), m, ap, m, fmt, nm);
    mkl<T>::gepack(MKL_COL_MAJOR, m, nrhs, Bp.data(), m, cp, m, fmt, nm);

    MKL_INT info = 99;

    /* 1. our compact QR. Each routine owns a work array sized from its OWN lwork
     *    query -- never carry one routine's lwork over to another (see the
     *    workspace note in cbk_compat.h): geqrf and ormqr can need different
     *    amounts, and an undersized work array is undefined behavior. */
    T wq_geqrf;
    compat<T>::geqrf(MKL_COL_MAJOR, m, n, ap, m, taup, &wq_geqrf, -1, &info, fmt, nm);
    std::vector<T> work_geqrf((size_t)std::max<MKL_INT>((MKL_INT)wq_geqrf, 1));
    compat<T>::geqrf(MKL_COL_MAJOR, m, n, ap, m, taup, work_geqrf.data(),
                     (MKL_INT)work_geqrf.size(), &info, fmt, nm);

    /* 2. our compact apply Q^T, with its own separately queried workspace. */
    T wq_ormqr;
    compat<T>::ormqr(MKL_COL_MAJOR, 'L', 'T', m, nrhs, k, ap, m, taup, cp, m, &wq_ormqr,
                     -1, &info, fmt, nm);
    std::vector<T> work_ormqr((size_t)std::max<MKL_INT>((MKL_INT)wq_ormqr, 1));
    compat<T>::ormqr(MKL_COL_MAJOR, 'L', 'T', m, nrhs, k, ap, m, taup, cp, m,
                     work_ormqr.data(), (MKL_INT)work_ormqr.size(), &info, fmt, nm);

    /* 3. MKL compact triangular solve R X = Q^T B */
    mkl<T>::trsm(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS, MKL_NONUNIT, n, nrhs,
                 T(1), ap, m, cp, m, fmt, nm);

    MatrixBatch<T> Xhat(nm, n, nrhs);
    auto Op = Xhat.base_ptrs();
    mkl<T>::geunpack(MKL_COL_MAJOR, n, nrhs, Op.data(), n, cp, m, fmt, nm);

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

    /* Suite 2: cross-check vs mkl_dgeqrf_compact, both layouts */
    fails += suite2<T>(MKL_COL_MAJOR, 8, 30, 30);
    fails += suite2<T>(MKL_COL_MAJOR, 8, 64, 20);
    fails += suite2<T>(MKL_ROW_MAJOR, 8, 30, 30);
    fails += suite2<T>(MKL_ROW_MAJOR, 11, 40, 24);

    /* Suite 3: end-to-end solver */
    fails += suite3<T>(8, 30, 5);
    fails += suite3<T>(8, 60, 4);
    fails += suite3<T>(16, 128, 3);
    fails += suite3<T>(7, 32, 6); /* padded partial group */

    return fails;
}

int main()
{
    const int fails = run_suites<double>() + run_suites<float>();
    return finish(fails);
}
