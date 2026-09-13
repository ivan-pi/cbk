/* test_sytrfnp_mkl.cpp
 *
 * Validation of cbk_?sytrfnp_compact, cbk_?sytrsnp_compact and
 * cbk_?sysvnp_compact against real Intel MKL, through the genuine MKL
 * Compact pipeline (mkl_?gepack_compact / mkl_?geunpack_compact). This is the
 * design document's correctness gate (docs/cbk_dsytrfnp_compact_design.md
 * section 7). Standard LAPACK has no unpivoted dense LDL^T (?sytrf is
 * Bunch-Kaufman-pivoted, so its factors differ elementwise), so the dense
 * reference here is the algebraic contract itself plus MKL's own unpivoted
 * compact LU:
 *
 * Suite 1 (section 7.1) -- Factorization invariants, for each (uplo, layout):
 *   a random symmetric *indefinite* batch (known-good unpivoted LDL^T by
 *   construction) is factored by compat<T>::sytrfnp, unpacked, and per matrix
 *   checked:
 *     - reconstruction residual || L D L^T - A ||_1 / ||A||_1 <= 20 n eps
 *       (U^T D U for upper),
 *     - the strictly-opposite triangle is bit-for-bit unchanged from the input.
 *
 * Suite 2 (section 7.2) -- Cross-check vs mkl_?getrfnp_compact: unpivoted LU of
 *   a symmetric matrix satisfies A = L (D L^T), i.e. it shares the unit-lower L
 *   and its U carries D on the diagonal. The same batch is factored by both
 *   routines (col-major lower vs full LU) and the strict lower triangles and
 *   diagonals are compared at the cross-check tolerance -- two independent
 *   implementations of the same pivots.
 *
 * Suite 3 (section 7.3) -- End-to-end indefinite solve A X = B: B = A X for
 *   known X; compat<T>::sytrfnp -> compat<T>::sytrsnp must recover X, over
 *   both uplo and both layouts, gating the system residual at 100 n eps and the
 *   forward error at 500 n eps; and compat<T>::sysvnp on the same packed input
 *   must reproduce the two-step factor and X bit-for-bit.
 *
 * Suite 4 -- nrhs = 0: compat<T>::sysvnp must still factor (LAPACK ?sysv
 *   calls ?sytrf unconditionally; the nrhs quick return is ?sytrs's),
 *   bit-identical to compat<T>::sytrfnp, with a never-referenced dummy bp.
 *
 * Build: needs Intel MKL (headers + libmkl_rt); wired up by CMakeLists.txt.
 *
 * Assisted-by: Claude Claude:claude-fable-5
 */

#include "test_mkl_util.hpp" /* compat<T>, mkl<T> + the MKL-free helpers */

#include <cstdio>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

using namespace cbk::test;

namespace {

/* ---------------- Suite 1: factorization invariants ------------------- */

template <class T> int suite1(MKL_LAYOUT layout, MKL_UPLO uplo, int nm, int n)
{
    const double eps = std::numeric_limits<T>::epsilon();
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt);
    const bool row = (layout == MKL_ROW_MAJOR);
    const bool up = (uplo == MKL_UPPER);
    const char ul = up ? 'U' : 'L';

    MatrixBatch<T> A(nm, n, n);
    for (int v = 0; v < nm; ++v)
        gen_sym_ldlt(A.view(v));

    /* pack the full symmetric A, factor with the routine under test, unpack */
    auto Ap = A.base_ptrs();
    MKL_INT sz_a = mkl<T>::get_size(n, n, fmt, nm);
    auto ap_buf = cbk::detail::mkl_alloc_bytes<T>(sz_a);
    T *ap = ap_buf.get();
    mkl<T>::gepack(layout, n, n, Ap.data(), n, ap, n, fmt, nm);

    MKL_INT info = 99;
    compat<T>::sytrfnp(layout, uplo, n, ap, n, &info, fmt, nm);

    MatrixBatch<T> H(nm, n, n);
    auto Hp = H.base_ptrs();
    mkl<T>::geunpack(layout, n, n, Hp.data(), n, ap, n, fmt, nm);

    int fails = 0;
    if (info != 0) {
        ++fails;
        std::printf("    info = %ld (expected 0)\n", (long)info);
    }

    double worst_res = 0, worst_untouched = 0;
    std::vector<T> R(A.stride());
    for (int v = 0; v < nm; ++v) {
        /* the factor is stored in `layout`; the input and the residual are the
         * column-major dense side */
        const auto Hm = H.view(v, row);
        const auto Am = A.view(v);
        const auto Res = mat_view(R.data(), n, n);

        /* reconstruction residual: unit factor off the diagonal, D on it */
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Res(i, j) = (T)(ldlt_reconstruct(Hm, i, j, up) - (double)Am(i, j));
        worst_res = std::max(worst_res, norm1(Res) / std::max(norm1(Am), norm_floor));

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                const bool named = up ? (i <= j) : (i >= j);
                if (!named)
                    worst_untouched =
                        std::max<double>(worst_untouched, std::abs(Hm(i, j) - Am(i, j)));
            }
    }

    const double rtol = 20.0 * n * eps;
    bool ok = (worst_res <= rtol) && (worst_untouched == 0.0);
    fails += !ok;
    std::printf("  [suite1] %s uplo=%c V=%-2d nm=%-2d n=%-3d | res %.2e (%.1e) "
                "untouched %.0e %s\n",
                row ? "row" : "col", ul, V, nm, n, worst_res, rtol, worst_untouched,
                ok ? "OK" : "FAIL");
    return fails;
}

/* -------- Suite 2: cross-check vs mkl_?getrfnp_compact (A = L * DL^T) -- */

template <class T> int suite2(int nm, int n)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt);

    MatrixBatch<T> A(nm, n, n);
    for (int v = 0; v < nm; ++v)
        gen_sym_ldlt(A.view(v));
    auto Ap = A.base_ptrs();

    MKL_INT sz_a = mkl<T>::get_size(n, n, fmt, nm);
    auto ap1 = cbk::detail::mkl_alloc_bytes<T>(sz_a);
    auto ap2 = cbk::detail::mkl_alloc_bytes<T>(sz_a);
    mkl<T>::gepack(MKL_COL_MAJOR, n, n, Ap.data(), n, ap1.get(), n, fmt, nm);
    mkl<T>::gepack(MKL_COL_MAJOR, n, n, Ap.data(), n, ap2.get(), n, fmt, nm);

    MKL_INT info_cbk = 99, info_mkl = 99;
    compat<T>::sytrfnp(MKL_COL_MAJOR, MKL_LOWER, n, ap1.get(), n, &info_cbk, fmt, nm);
    mkl<T>::getrfnp(MKL_COL_MAJOR, n, n, ap2.get(), n, &info_mkl, fmt, nm);

    /* unpack both; the unpivoted LU of symmetric A is A = L * (D L^T), so its
     * unit-lower L must match ours and its U diagonal must be our D. The two
     * implementations do not share an arithmetic order, so a small fixed
     * tolerance confirms the same pivots, not bit equality. */
    MatrixBatch<T> F(nm, n, n), G(nm, n, n);
    auto Fp = F.base_ptrs();
    auto Gp = G.base_ptrs();
    mkl<T>::geunpack(MKL_COL_MAJOR, n, n, Fp.data(), n, ap1.get(), n, fmt, nm);
    mkl<T>::geunpack(MKL_COL_MAJOR, n, n, Gp.data(), n, ap2.get(), n, fmt, nm);

    double dl = 0, dd = 0;
    for (int v = 0; v < nm; ++v) {
        const auto Fv = F.view(v); /* our (D, L)       */
        const auto Gv = G.view(v); /* MKL's LU (D L^T) */
        for (int j = 0; j < n; ++j) {
            dd = std::max<double>(dd, std::abs(Fv(j, j) - Gv(j, j)));
            for (int i = j + 1; i < n; ++i)
                dl = std::max<double>(dl, std::abs(Fv(i, j) - Gv(i, j)));
        }
    }
    const double tol = cross_tol<T>();
    bool ok = (dl <= tol) && (dd <= tol) && info_cbk == 0 && info_mkl == 0;
    std::printf("  [suite2] vs getrfnp V=%-2d nm=%-2d n=%-3d | max|L-L_lu| %.2e "
                "max|D-diag(U)| %.2e (tol %.0e) info=%ld/%ld %s\n",
                V, nm, n, dl, dd, tol, (long)info_cbk, (long)info_mkl,
                ok ? "OK" : "FAIL");
    return !ok;
}

/* ---------------- Suite 3: end-to-end indefinite solve ----------------- */

template <class T> int suite3(MKL_LAYOUT layout, MKL_UPLO uplo, int nm, int n, int nrhs)
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
        gen_sym_ldlt(A.view(v));
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
    compat<T>::sytrfnp(layout, uplo, n, ap, n, &info_f, fmt, nm);
    compat<T>::sytrsnp(layout, uplo, n, nrhs, ap, n, bp, ldb, &info_s, fmt, nm);
    compat<T>::sysvnp(layout, uplo, n, nrhs, ap2, n, bp2, ldb, &info_v, fmt, nm);

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
    /* the residual is what the backward-stable sweeps control; the forward
     * error additionally carries cond(A), so its gate gets headroom */
    const double rtol_res = 100.0 * n * eps;
    const double rtol_fwd = 500.0 * n * eps;
    bool ok = (worst_fwd <= rtol_fwd && worst_res <= rtol_res && fused_same);
    fails += !ok;
    std::printf("  [suite3] %s uplo=%c V=%-2d nm=%-2d n=%-3d nrhs=%d | fwd %.2e "
                "(%.1e) res %.2e (%.1e) sysv==trf+trs:%s %s\n",
                row ? "row" : "col", ul, V, nm, n, nrhs, worst_fwd, rtol_fwd, worst_res,
                rtol_res, fused_same ? "yes" : "NO", ok ? "OK" : "FAIL");
    return fails;
}

/* ------ Suite 4: nrhs = 0 still factors (LAPACK ?sysv contract) --------- */

template <class T> int suite4(int nm, int n)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt);

    MatrixBatch<T> A(nm, n, n);
    for (int v = 0; v < nm; ++v)
        gen_sym_ldlt(A.view(v));
    auto Ap = A.base_ptrs();

    MKL_INT sz_a = mkl<T>::get_size(n, n, fmt, nm);
    auto ap1 = cbk::detail::mkl_alloc_bytes<T>(sz_a);
    auto ap2 = cbk::detail::mkl_alloc_bytes<T>(sz_a);
    mkl<T>::gepack(MKL_COL_MAJOR, n, n, Ap.data(), n, ap1.get(), n, fmt, nm);
    mkl<T>::gepack(MKL_COL_MAJOR, n, n, Ap.data(), n, ap2.get(), n, fmt, nm);

    MKL_INT info_f = 99, info_v = 99;
    compat<T>::sytrfnp(MKL_COL_MAJOR, MKL_LOWER, n, ap1.get(), n, &info_f, fmt, nm);
    T b_dummy = 0; /* never referenced at nrhs = 0, present per Fortran semantics */
    compat<T>::sysvnp(MKL_COL_MAJOR, MKL_LOWER, n, /*nrhs=*/0, ap2.get(), n, &b_dummy, n,
                      &info_v, fmt, nm);

    const size_t na = (size_t)sz_a / sizeof(T);
    const bool same = std::equal(ap1.get(), ap1.get() + na, ap2.get());
    bool ok = (info_f == 0) && (info_v == 0) && same;
    std::printf("  [suite4] nrhs=0 V=%-2d nm=%-2d n=%-3d | sysv==trf:%s info=%ld/%ld "
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

    /* Suite 1: factorization invariants over (uplo, layout), sizes incl. padding */
    const MKL_LAYOUT lays[] = {MKL_COL_MAJOR, MKL_ROW_MAJOR};
    const MKL_UPLO ups[] = {MKL_LOWER, MKL_UPPER};
    for (MKL_LAYOUT L : lays)
        for (MKL_UPLO U : ups) {
            fails += suite1<T>(L, U, 8, 30);
            fails += suite1<T>(L, U, 16, 60);
            fails += suite1<T>(L, U, 11, 43); /* padded partial group */
        }
    fails += suite1<T>(MKL_COL_MAJOR, MKL_LOWER, 8, 128);
    fails += suite1<T>(MKL_COL_MAJOR, MKL_LOWER, 4, 3); /* smallest, padded */

    /* Suite 2: (L, D) cross-check vs MKL's unpivoted compact LU */
    fails += suite2<T>(8, 30);
    fails += suite2<T>(11, 40); /* padded partial group */

    /* Suite 3: end-to-end indefinite solver, both uplo and layouts, two-step
     * and fused */
    for (MKL_LAYOUT L : lays)
        for (MKL_UPLO U : ups)
            fails += suite3<T>(L, U, 8, 30, 5);
    fails += suite3<T>(MKL_COL_MAJOR, MKL_LOWER, 16, 60, 4);
    fails += suite3<T>(MKL_COL_MAJOR, MKL_LOWER, 7, 32, 6); /* padded partial group */
    fails += suite3<T>(MKL_COL_MAJOR, MKL_UPPER, 6, 25, 1); /* single RHS */

    /* Suite 4: nrhs = 0 must factor anyway (LAPACK ?sysv), bit-identical to
     * sytrfnp, bp a never-referenced dummy */
    fails += suite4<T>(8, 30);

    return fails;
}

int main()
{
    const int fails = run_suites<double>() + run_suites<float>();
    return finish(fails);
}
