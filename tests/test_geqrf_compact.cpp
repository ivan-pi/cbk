// test_geqrf_compact.cpp
//
// Validation of the templated compact QR factorization (dgeqrf_compact /
// sgeqrf_compact) against LAPACKE (test_lapack_util.hpp). The reference is
// LAPACK's unblocked Householder QR, ?geqr2 / ?larfg -- the same algorithm the
// vectorized kernel executes V lanes at a time, so a correct kernel matches it
// to working precision.
//
// Checks per (T, V):
//   1. compact (H, tau)  ==  LAPACKE_?geqr2 (H, tau)  (elementwise, ~eps*scale)
//   2. reconstruction  Q * triu(H) == A  by ?ormqr    (valid factorization)
//   3. reflectors are usable: ormqr_compact('T') then ?trsm recovers a known
//      X from B = A X                                 (in-situ with ormqr)
// and, per (T, structure, cond) on the dense-LAPACK contract (design 7.1):
//   4. Q = ?orgqr(H, tau), R = triu(H): the factorization residual
//      ||R - Q^T A||_1 / ||A||_1 <= 20 n eps and the orthogonality
//      ||Q^T Q - I||_1 <= 100 n eps, on dense, column-scaled, rank-deficient
//      and near-collinear inputs; (H, tau) elementwise vs the blocked
//      LAPACKE_?geqrf on the dense ones (relative, 100 n eps)
// plus LAPACK-style argument validation of the C API.
//
// Assisted-by: Claude:claude-opus-4-8 Claude:claude-fable-5

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>
#include <type_traits>

#include "test_lapack_util.hpp" // compact<T>, lapack<T>, ref_*, MatrixBatch, pack/unpack

using namespace cbk::test;

// --------------------------- one test case --------------------------

template <class T, int V> static int run_case(int nm, int m, int n)
{
    const int k = std::min(m, n);
    const T eps = std::numeric_limits<T>::epsilon();

    // random A batch, diagonal-boosted so the columns stay well conditioned
    MatrixBatch<T> A(nm, m, n), Aref(nm, m, n), tau_ref(nm, k, 1);
    for (int idx = 0; idx < nm; ++idx) {
        gen_boosted(A.view(idx));
        std::copy(A[idx], A[idx] + (size_t)m * n, Aref[idx]); // Aref <- A
        ref_geqr2(Aref.view(idx), tau_ref[idx]);              // reference (H, tau)
    }

    // pack A, factor with the routine under test, unpack (H, tau)
    std::vector<T> ap = pack_compact(A, m, V);
    std::vector<T> tp = compact_buffer<T>(nm, k, 1, k, V); /* the kernel fills it */

    int info = compact<T>::geqrf('C', m, n, ap.data(), m, tp.data(), V, nm);

    MatrixBatch<T> Aout(nm, m, n), tau_out(nm, k, 1);
    unpack_compact(Aout, ap.data(), m, V);
    unpack_tau(tau_out, tp.data(), V);

    // check 1: (H, tau) match the scalar reference elementwise
    double e_h = 0, e_t = 0;
    for (int idx = 0; idx < nm; ++idx) {
        e_h = std::max(e_h, max_abs_diff(Aout[idx], Aref[idx], (size_t)m * n));
        e_t = std::max(e_t, max_abs_diff(tau_out[idx], tau_ref[idx], (size_t)k));
    }

    // check 2: reconstruction Q * triu(H) == A (valid factorization)
    double e_rec = 0;
    for (int idx = 0; idx < nm; ++idx) {
        std::vector<T> Recs((size_t)m * n, T(0)); // start from R
        const auto Rec = mat_view(Recs.data(), m, n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i <= std::min(j, k - 1); ++i)
                Rec(i, j) = Aout(idx, i, j);
        ref_ormqr('N', k, Aout.view(idx), tau_out[idx], Rec);
        e_rec = std::max(e_rec, max_abs_diff(Recs.data(), A[idx], (size_t)m * n));
    }

    // check 3: solve A X = B with the produced reflectors (square only), using
    // the project's ormqr kernel + a triangular solve. X(:,j) = j+1.
    double e_solve = -1;
    if (m == n) {
        const int nrhs = 3;
        const std::vector<T> Xs = known_solution<T>(n, nrhs);
        const auto X = mat_view(Xs.data(), n, nrhs);
        MatrixBatch<T> B(nm, n, nrhs);
        for (int idx = 0; idx < nm; ++idx)
            matmul(A.view(idx), X, B.view(idx)); /* B = A X */
        std::vector<T> bp = pack_compact(B, n, V);
        compact<T>::ormqr('T', n, nrhs, k, ap.data(), n, tp.data(), bp.data(), n, V, nm);
        MatrixBatch<T> Bo(nm, n, nrhs);
        unpack_compact(Bo, bp.data(), n, V);
        e_solve = 0;
        for (int idx = 0; idx < nm; ++idx) {
            ref_trsm_upper(Aout.view(idx), Bo.view(idx));
            e_solve =
                std::max(e_solve, max_abs_diff(Bo[idx], Xs.data(), (size_t)n * nrhs));
        }
    }

    const double scale = std::max(1, m);
    const double tol_fac = 200.0 * eps * scale; // same op sequence as ?geqr2, ~eps
    const double tol_rec = 200.0 * eps * scale; // ?ormqr's rounding differs, same bound
    const double tol_sol = 1e5 * eps * m;       // cond(A)-dependent
    bool ok_h = e_h <= tol_fac, ok_t = e_t <= tol_fac, ok_r = e_rec <= tol_rec;
    bool ok_s = (e_solve < 0) || (e_solve <= tol_sol);

    std::printf("T=%-6s V=%-2d nm=%-2d m=%-3d n=%-3d | H:%.1e %s tau:%.1e %s rec:%.1e %s",
                compact<T>::name, V, nm, m, n, e_h, ok_h ? "OK" : "FAIL", e_t,
                ok_t ? "OK" : "FAIL", e_rec, ok_r ? "OK" : "FAIL");
    if (e_solve >= 0) std::printf(" solve:%.1e %s", e_solve, ok_s ? "OK" : "FAIL");
    std::printf(" | info=%d\n", info);
    return (info != 0) + !ok_h + !ok_t + !ok_r + !ok_s;
}

// ------------------ invariants vs dense LAPACK (design 7.1) ----------

// Input structures (a subset of the GPU competition's stress set). The QR
// residual and orthogonality are backward-stable quantities, so they must hold
// to working precision for every structure -- rank deficiency and near-collinear
// columns included -- exactly as they do for dense LAPACK.
enum Structure { DENSE, RANK_DEFICIENT, NEAR_COLLINEAR };

// A random m x n matrix: gen_boosted's, with the diagonal boost taming the
// conditioning for DENSE only, where cond applies the competition's column
// scaling (columns *= logspace(0,-cond,n)). The structured variants make the
// last column a (near-)copy of the first, so the trailing reflector sees a
// (near-)zero sub-diagonal norm -- the branch the masked larfg must get right.
template <class T> static void gen_matrix(MatrixView<T> A, double cond, Structure s)
{
    const int m = A.rows, n = A.cols;
    gen_boosted(A, s == DENSE ? T(2) : T(0));
    if (s == DENSE) {
        if (cond > 0.0)
            for (int j = 0; j < n; ++j) {
                double sc = std::pow(10.0, -cond * (n > 1 ? (double)j / (n - 1) : 0.0));
                for (int i = 0; i < m; ++i)
                    A(i, j) *= sc;
            }
    }
    else if (n >= 2) {
        const double noise = (s == NEAR_COLLINEAR) ? 1e-9 : 0.0; /* exact dup if 0 */
        for (int i = 0; i < m; ++i)
            A(i, n - 1) = A(i, 0) * (1.0 + noise * frand<T>());
    }
}

template <class T, int V>
static int run_invariants(int nm, int m, int n, double cond, Structure structure = DENSE)
{
    const double eps = std::numeric_limits<T>::epsilon();
    const int k = std::min(m, n);

    MatrixBatch<T> A(nm, m, n);
    for (int idx = 0; idx < nm; ++idx)
        gen_matrix(A.view(idx), cond, structure);

    std::vector<T> ap = pack_compact(A, m, V);
    std::vector<T> tp = compact_buffer<T>(nm, k, 1, k, V);
    const int info = compact<T>::geqrf('C', m, n, ap.data(), m, tp.data(), V, nm);
    MatrixBatch<T> H(nm, m, n), tau(nm, k, 1);
    unpack_compact(H, ap.data(), m, V);
    unpack_tau(tau, tp.data(), V);

    double worst_res = 0, worst_orth = 0, worst_el = 0;
    std::vector<T> Qs((size_t)m * k), QtAs((size_t)k * n), QtQs((size_t)k * k),
        Hrefs(A.stride()), tauref(k);
    const auto Q = mat_view(Qs.data(), m, k), QtA = mat_view(QtAs.data(), k, n),
               QtQ = mat_view(QtQs.data(), k, k), Href = mat_view(Hrefs.data(), m, n);
    for (int idx = 0; idx < nm; ++idx) {
        const auto Am = A.view(idx), Hm = H.view(idx);

        // Q = the first k columns of H(0)..H(k-1), from the reflectors in the
        // first k columns of H
        copy_matrix(leading(Hm, m, k), Q);
        ref_orgqr(k, Q, tau[idx]);

        // residual R - Q^T A, R = triu(H) (k x n)
        matmul(Q.transposed(), Am, QtA);
        double resid = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < k; ++i)
                resid = std::max(
                    resid, (double)std::abs(((i <= j) ? Hm(i, j) : T(0)) - QtA(i, j)));
        worst_res = std::max(worst_res, resid / std::max(norm1(Am), norm_floor));

        // orthogonality Q^T Q - I
        matmul(Q.transposed(), Q, QtQ);
        for (int d = 0; d < k; ++d)
            QtQ(d, d) -= T(1);
        worst_orth = std::max(worst_orth, norm1(QtQ));

        // elementwise vs the blocked LAPACKE_?geqrf, on the dense inputs only,
        // where the reflectors are essentially unique: it catches a
        // sign-convention regression the residual gates cannot see (observed
        // ~n*eps). Rank-deficient and near-collinear inputs have no unique
        // reflectors (el ~ 1e-2 there), so the comparison is skipped (design
        // doc 7.1).
        if (structure == DENSE) {
            copy_matrix(Am, Href);
            ref_geqrf(Href, tauref.data());
            const double el = std::max(max_abs_diff(H[idx], Hrefs.data(), A.stride()),
                                       max_abs_diff(tau[idx], tauref.data(), (size_t)k));
            worst_el = std::max(worst_el, el / std::max(norm1(Am), norm_floor));
        }
    }

    const double rtol_res = 20.0 * n * eps, rtol_orth = 100.0 * n * eps;
    const double rtol_el = 100.0 * n * eps;
    const bool ok = (info == 0) && (worst_res <= rtol_res) && (worst_orth <= rtol_orth) &&
                    (worst_el <= rtol_el);
    const char *sname = structure == DENSE            ? "dense"
                        : structure == RANK_DEFICIENT ? "rankdef"
                                                      : "collin";
    std::printf("T=%-6s V=%-2d nm=%-2d m=%-3d n=%-3d cond=%.0f %-8s| res %.2e (%.1e) "
                "orth %.2e (%.1e) el %.1e | info=%d %s\n",
                compact<T>::name, V, nm, m, n, cond, sname, worst_res, rtol_res,
                worst_orth, rtol_orth, worst_el, info, ok ? "OK" : "FAIL");
    return !ok;
}

// The invariants cases, in one precision and width.
template <class T, int V> static int run_invariant_set()
{
    int fails = 0;
    fails += run_invariants<T, V>(8, 30, 30, 0.0);
    fails += run_invariants<T, V>(16, 60, 60, 0.0);
    fails += run_invariants<T, V>(11, 43, 43, 0.0); // padded partial group
    fails += run_invariants<T, V>(8, 64, 20, 0.0);  // tall
    fails += run_invariants<T, V>(8, 20, 64, 0.0);  // wide
    fails += run_invariants<T, V>(8, 30, 30, 4.0);  // column-scaled (dynamic range)
    fails += run_invariants<T, V>(8, 128, 128, 0.0);
    // conditioning-robustness stress: backward-stable gates must still hold
    fails += run_invariants<T, V>(8, 40, 40, 0.0, RANK_DEFICIENT);
    fails += run_invariants<T, V>(8, 40, 40, 0.0, NEAR_COLLINEAR);
    fails +=
        run_invariants<T, V>(11, 60, 24, 0.0, RANK_DEFICIENT); // wide-ish, padded group
    return fails;
}

// ------------------------ underflow scope (design 6.6) ---------------

// A column whose sub-diagonal entries are so small that their squares
// underflow to zero (below the square root of the smallest subnormal, ~1.5e-162
// in FP64 and ~3.7e-23 in FP32) reads as already triangular: tau = 0, the
// diagonal stays, and the reflector body below it is scaled by 0 (so the
// storage holds zeros, which ormqr/orgqr ignore under tau = 0 anyway), rather
// than taking dlarfg's rescaled path. That is the documented scope, and the
// factorization it returns is still exact to working precision (the dropped
// tail is below eps * ||A|| by many orders). This case checks that contract
// directly -- the scalar reference (std::hypot) does not underflow, so it is
// not the yardstick here -- and that no lane turns NaN.
template <class T, int V> static int test_underflow()
{
    const int nm = V, m = 6, n = 4, tiny_col = 0; /* no earlier reflector touches it */
    const T tiny = std::is_same_v<T, double> ? T(1e-170) : T(1e-25);
    const T eps = std::numeric_limits<T>::epsilon();

    MatrixBatch<T> A(nm, m, n);
    for (int idx = 0; idx < nm; ++idx) {
        gen_boosted(A.view(idx));
        for (int i = tiny_col + 1; i < m; ++i)
            A(idx, i, tiny_col) = tiny * T(i); /* tail of column 0: underflows squared */
    }
    std::vector<T> ap = pack_compact(A, m, V);
    std::vector<T> tp = compact_buffer<T>(nm, n, 1, n, V);
    int info = compact<T>::geqrf('C', m, n, ap.data(), m, tp.data(), V, nm);

    MatrixBatch<T> Aout(nm, m, n), tau(nm, n, 1);
    unpack_compact(Aout, ap.data(), m, V);
    unpack_tau(tau, tp.data(), V);

    int fails = (info != 0);
    double e_rec = 0;
    for (int idx = 0; idx < nm; ++idx) {
        /* column 0 is the first reflector's, so nothing reduces it first: its
         * tail underflows squared, tau = 0, the diagonal is kept, the body
         * below it is zeroed */
        if (tau(idx, tiny_col, 0) != T(0)) ++fails;
        if (Aout(idx, tiny_col, tiny_col) != A(idx, tiny_col, tiny_col)) ++fails;
        for (int i = tiny_col + 1; i < m; ++i)
            if (Aout(idx, i, tiny_col) != T(0)) ++fails;
        bool finite = true;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                finite = finite && std::isfinite(Aout(idx, i, j));
        if (!finite) ++fails;

        std::vector<T> Recs((size_t)m * n, T(0));
        const auto Rec = mat_view(Recs.data(), m, n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i <= std::min(j, n - 1); ++i)
                Rec(i, j) = Aout(idx, i, j);
        ref_ormqr('N', n, Aout.view(idx), tau[idx], Rec);
        e_rec = std::max(e_rec, max_abs_diff(Recs.data(), A[idx], (size_t)m * n));
    }
    const double tol_rec = 200.0 * eps * m;
    fails += !(e_rec <= tol_rec);
    std::printf("T=%-6s V=%-2d underflow column: tau=0, diag kept, body 0, finite %s | "
                "rec:%.1e %s\n",
                compact<T>::name, V, fails ? "FAIL" : "OK", e_rec,
                e_rec <= tol_rec ? "OK" : "FAIL");
    return fails;
}

// --------------------- C API argument validation --------------------

static int test_validation()
{
    const int m = 8, n = 6, V = 4, nm = 4, ld = 8;
    std::vector<double> ap((size_t)ld * n * V, 0), tau((size_t)std::min(m, n) * V, 0);
    auto call = [&](char lay, int m_, int n_, int ldap_, int V_, int nm_) {
        return dgeqrf_compact(lay, m_, n_, ap.data(), ldap_, tau.data(), V_, nm_);
    };
    // clang-format off
    const ApiCheck t[] = {
        {"valid col",   call('C', m, n,  ld,  V, nm),   0},
        {"valid row",   call('R', m, n,  n,   V, nm),   0},   // row-major ld >= n
        {"bad layout",  call('X', m, n,  ld,  V, nm),  -1},
        {"m<0",         call('C', -1, n, ld,  V, nm),  -2},
        {"n<0",         call('C', m, -1, ld,  V, nm),  -3},
        {"ldap<m",      call('C', m, n,  m-1, V, nm),  -5},
        {"bad V",       call('C', m, n,  ld,  3, nm),  -7},
        {"nm<0",        call('C', m, n,  ld,  V, -1),  -8},
        {"empty m=0",   call('C', 0, n,  1,   V, nm),   0},
        {"empty nm=0",  call('C', m, n,  ld,  V, 0),    0},
    };
    // clang-format on
    return report_api_checks(t);
}

// ------------------------------- main --------------------------------

int main()
{
    int fails = 0;
    fails += test_validation();
    if (cbk_get_version() != CBK_VERSION) {
        std::printf("cbk_get_version() = %d, header CBK_VERSION = %d FAIL\n",
                    cbk_get_version(), CBK_VERSION);
        ++fails;
    }

    // square (the emphasis)
    fails += run_case<double, 2>(4, 30, 30);
    fails += run_case<double, 4>(8, 30, 30);
    fails += run_case<double, 8>(16, 60, 60);
    fails += run_case<double, 8>(11, 43, 43); // padded partial group
    fails += run_case<double, 4>(8, 3, 3);    // smallest supported
    fails += run_case<double, 4>(40, 24, 24); // 10 groups: OpenMP path when threads <= 10
    // rectangular
    fails += run_case<double, 4>(8, 64, 20); // tall
    fails += run_case<double, 4>(8, 20, 64); // wide
    // float
    fails += run_case<float, 8>(16, 30, 30);
    fails += run_case<float, 16>(32, 43, 17);
    // numerical scope: an underflowing column reads as already triangular
    fails += test_underflow<double, 4>();
    fails += test_underflow<float, 8>();

    // the dense-LAPACK contract (design 7.1): residual and orthogonality of
    // the materialized Q, over conditioning and structure, in both precisions
    fails += run_invariant_set<double, 4>();
    fails += run_invariant_set<float, 8>();

    return finish(fails);
}
