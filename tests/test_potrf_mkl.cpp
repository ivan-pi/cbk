/* test_potrf_mkl.cpp
 *
 * Validation of cbk_?potrf_compact against real Intel MKL, through the
 * genuine MKL Compact pipeline (mkl_dgepack_compact / mkl_dgeunpack_compact):
 * the MKL-side half of cbk_dpotrf_compact_design.md section 7. The
 * dense-LAPACK invariants of section 7.1 (reconstruction, the untouched
 * triangle, the factor vs LAPACKE_?potrf, over uplo / layout / cond) run in
 * the portable suite, test_potrf_compact.cpp, on any LAPACKE stack.
 *
 * Suite 2 (section 7.2) -- Cross-check vs mkl_dpotrf_compact: the same packed
 *   batch factored by both, compact buffers compared elementwise at a small
 *   fixed tolerance (1e-9). Run for both layouts and both uplo.
 *
 * Suite 3 (section 7.3) -- End-to-end SPD solve AX = B: B = A X for known X;
 *   compat<T>::potrf('L') -> mkl<T>::trsm('L','L','N') ->
 *   mkl<T>::trsm('L','L','T') must recover X. Gates forward error and the
 *   system residual at 100 n eps.
 *
 * Suite 4 (section 7.5) -- The packaged solve: compat<T>::potrf ->
 *   compat<T>::potrs must recover X, over both uplo and both layouts, with
 *   the same 100 n eps gates as suite 3; and compat<T>::posv on the same
 *   packed input must reproduce the two-step factor and X bit-for-bit.
 *
 * Suite 5 -- nrhs = 0: compat<T>::posv must still factor (LAPACK ?posv calls
 *   ?potrf unconditionally; the nrhs quick return is ?potrs's), bit-identical
 *   to compat<T>::potrf, with a never-referenced dummy bp.
 *
 * Build: needs Intel MKL (headers + libmkl_rt); wired up by CMakeLists.txt.
 *
 * Assisted-by: Claude:claude-opus-4-8 Claude:claude-fable-5
 */

#include "test_mkl_util.hpp" /* compat<T>, mkl<T> + the shared helpers */

#include <cstdio>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

using namespace cbk::test;

namespace {

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
    auto ap1 = cbk::detail::mkl_alloc_bytes<T>(sz_a);
    auto ap2 = cbk::detail::mkl_alloc_bytes<T>(sz_a);
    mkl<T>::gepack(layout, n, n, Ap.data(), n, ap1.get(), n, fmt, nm);
    mkl<T>::gepack(layout, n, n, Ap.data(), n, ap2.get(), n, fmt, nm);

    MKL_INT info = 0;
    compat<T>::potrf(layout, uplo, n, ap1.get(), n, &info, fmt, nm);
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
    auto ap_buf = cbk::detail::mkl_alloc_bytes<T>(sz_a);
    auto bp_buf = cbk::detail::mkl_alloc_bytes<T>(sz_b);
    T *ap = ap_buf.get(), *bp = bp_buf.get();
    mkl<T>::gepack(MKL_COL_MAJOR, n, n, Ap.data(), n, ap, n, fmt, nm);
    mkl<T>::gepack(MKL_COL_MAJOR, n, nrhs, Bp.data(), n, bp, n, fmt, nm);

    MKL_INT info = 99;

    /* 1. our compact Cholesky: A -> L (lower). No workspace. */
    compat<T>::potrf(MKL_COL_MAJOR, MKL_LOWER, n, ap, n, &info, fmt, nm);

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

/* ------- Suite 4: potrs / posv, both uplo and layouts, fused identity ---- */

template <class T> int suite4(MKL_LAYOUT layout, MKL_UPLO uplo, int nm, int n, int nrhs)
{
    const double eps = std::numeric_limits<T>::epsilon();
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt);
    const bool row = (layout == MKL_ROW_MAJOR);
    const char ul = (uplo == MKL_UPPER) ? 'U' : 'L';

    const std::vector<T> Xs = known_solution<T>(n, nrhs);
    const auto X = mat_view(Xs.data(), n, nrhs);

    MatrixBatch<T> A(nm, n, n), B(nm, n, nrhs);
    for (int v = 0; v < nm; ++v) {
        gen_spd(A.view(v), 0.0);
        matmul(A.view(v), X, B.view(v)); /* B = A X */
    }
    auto Ap = A.base_ptrs();

    /* mkl_?ge(un)pack_compact read/write the dense side in `layout` too, so the
     * row-major runs pack from (and unpack to) a row-major staging copy of the
     * non-square B with ld = nrhs. The symmetric square A needs no staging: its
     * row-major image is itself. */
    MatrixBatch<T> Bsrc(row ? nm : 0, n, nrhs);
    for (int v = 0; v < Bsrc.count(); ++v) /* same matrices, the other layout */
        copy_matrix(B.view(v), Bsrc.view(v, /*rowmajor=*/true));
    auto Bp = row ? Bsrc.base_ptrs() : B.base_ptrs();

    MKL_INT sz_a = mkl<T>::get_size(n, n, fmt, nm);
    MKL_INT sz_b = mkl<T>::get_size(n, nrhs, fmt, nm);
    auto ap_buf = cbk::detail::mkl_alloc_bytes<T>(sz_a);
    auto bp_buf = cbk::detail::mkl_alloc_bytes<T>(sz_b);
    auto ap2_buf = cbk::detail::mkl_alloc_bytes<T>(sz_a); /* the fused call's copies */
    auto bp2_buf = cbk::detail::mkl_alloc_bytes<T>(sz_b);
    T *ap = ap_buf.get(), *bp = bp_buf.get(), *ap2 = ap2_buf.get(), *bp2 = bp2_buf.get();
    const MKL_INT ldb = row ? nrhs : n;
    mkl<T>::gepack(layout, n, n, Ap.data(), n, ap, n, fmt, nm);
    mkl<T>::gepack(layout, n, nrhs, Bp.data(), ldb, bp, ldb, fmt, nm);
    mkl<T>::gepack(layout, n, n, Ap.data(), n, ap2, n, fmt, nm);
    mkl<T>::gepack(layout, n, nrhs, Bp.data(), ldb, bp2, ldb, fmt, nm);

    MKL_INT info_f = 99, info_s = 99, info_v = 99;
    compat<T>::potrf(layout, uplo, n, ap, n, &info_f, fmt, nm);
    compat<T>::potrs(layout, uplo, n, nrhs, ap, n, bp, ldb, &info_s, fmt, nm);
    compat<T>::posv(layout, uplo, n, nrhs, ap2, n, bp2, ldb, &info_v, fmt, nm);

    /* fused vs two-step: the same kernels in the same order -> the same bits,
     * over the whole compact buffers (padded lanes included) */
    const size_t na = (size_t)sz_a / sizeof(T), nb = (size_t)sz_b / sizeof(T);
    const bool fused_same = std::equal(ap, ap + na, ap2) && std::equal(bp, bp + nb, bp2);

    MatrixBatch<T> Xout(nm, n, nrhs), Xhat(nm, n, nrhs);
    auto Op = Xout.base_ptrs();
    mkl<T>::geunpack(layout, n, nrhs, Op.data(), ldb, bp, ldb, fmt, nm);
    if (row) { /* stage back to column-major for the checks */
        for (int v = 0; v < nm; ++v)
            copy_matrix(Xout.view(v, /*rowmajor=*/true), Xhat.view(v));
    }
    else {
        Xhat = Xout;
    }

    int fails = 0;
    if (info_f != 0 || info_s != 0 || info_v != 0) {
        ++fails;
        std::printf("    info = %ld/%ld/%ld (expected 0/0/0)\n", (long)info_f,
                    (long)info_s, (long)info_v);
    }
    const auto [worst_fwd, worst_res] = solve_errors(A, B, Xhat, X);
    const double rtol = 100.0 * n * eps;
    bool ok = (worst_fwd <= rtol && worst_res <= rtol && fused_same);
    fails += !ok;
    std::printf("  [suite4] %s uplo=%c V=%-2d nm=%-2d n=%-3d nrhs=%d | fwd %.2e "
                "res %.2e (rtol %.2e) posv==trf+trs:%s %s\n",
                row ? "row" : "col", ul, V, nm, n, nrhs, worst_fwd, worst_res, rtol,
                fused_same ? "yes" : "NO", ok ? "OK" : "FAIL");
    return fails;
}

/* ------ Suite 5: nrhs = 0 still factors (LAPACK ?posv contract) --------- */

template <class T> int suite5(int nm, int n)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt);

    MatrixBatch<T> A(nm, n, n);
    for (int v = 0; v < nm; ++v)
        gen_spd(A.view(v), 0.0);
    auto Ap = A.base_ptrs();

    MKL_INT sz_a = mkl<T>::get_size(n, n, fmt, nm);
    auto ap1 = cbk::detail::mkl_alloc_bytes<T>(sz_a);
    auto ap2 = cbk::detail::mkl_alloc_bytes<T>(sz_a);
    mkl<T>::gepack(MKL_COL_MAJOR, n, n, Ap.data(), n, ap1.get(), n, fmt, nm);
    mkl<T>::gepack(MKL_COL_MAJOR, n, n, Ap.data(), n, ap2.get(), n, fmt, nm);

    MKL_INT info_f = 99, info_v = 99;
    compat<T>::potrf(MKL_COL_MAJOR, MKL_LOWER, n, ap1.get(), n, &info_f, fmt, nm);
    T b_dummy = 0; /* never referenced at nrhs = 0, present per Fortran semantics */
    compat<T>::posv(MKL_COL_MAJOR, MKL_LOWER, n, /*nrhs=*/0, ap2.get(), n, &b_dummy, n,
                    &info_v, fmt, nm);

    const size_t na = (size_t)sz_a / sizeof(T);
    const bool same = std::equal(ap1.get(), ap1.get() + na, ap2.get());
    bool ok = (info_f == 0) && (info_v == 0) && same;
    std::printf("  [suite5] nrhs=0 V=%-2d nm=%-2d n=%-3d | posv==potrf:%s info=%ld/%ld "
                "%s\n",
                V, nm, n, same ? "yes" : "NO", (long)info_f, (long)info_v,
                ok ? "OK" : "FAIL");
    return !ok;
}

} /* anonymous namespace */

template <class T> int run_suites()
{
    std::printf("\n== %s: MKL compact format = %d, V = %d ==\n", compact<T>::name,
                (int)mkl_get_format_compact(), mkl<T>::vlen(mkl_get_format_compact()));

    int fails = 0;

    const MKL_LAYOUT lays[] = {MKL_COL_MAJOR, MKL_ROW_MAJOR};
    const MKL_UPLO ups[] = {MKL_LOWER, MKL_UPPER};

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

    /* Suite 4: the packaged potrs / posv solve, both uplo and layouts,
     * two-step and fused */
    for (MKL_LAYOUT L : lays)
        for (MKL_UPLO U : ups)
            fails += suite4<T>(L, U, 8, 30, 5);
    fails += suite4<T>(MKL_COL_MAJOR, MKL_LOWER, 16, 60, 4);
    fails += suite4<T>(MKL_COL_MAJOR, MKL_LOWER, 7, 32, 6); /* padded partial group */
    fails += suite4<T>(MKL_COL_MAJOR, MKL_UPPER, 6, 25, 1); /* single RHS */

    /* Suite 5: nrhs = 0 must factor anyway (LAPACK ?posv), bit-identical to
     * potrf, bp a never-referenced dummy */
    fails += suite5<T>(8, 30);

    return fails;
}

int main()
{
    const int fails = run_suites<double>() + run_suites<float>();
    return finish(fails);
}
