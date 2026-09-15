// test_gels_compact.cpp
//
// Validation of the templated compact least-squares / minimum-norm solve
// (dgels_compact / sgels_compact) against LAPACKE (test_lapack_util.hpp).
// The reference is ref_gels: LAPACKE_?gels for X and LAPACKE_?geqrf /
// ?gelqf for the factorization and tau (which is also what ?gels runs
// inside) -- the same steps the vectorized kernel executes V lanes at a
// time, blocked by LAPACK above its crossover -- and, independently of it,
// the properties that define the two solutions.
//
// Checks per (T, V, layout, trans, shape), the design's 7.1 gates, each a
// test ratio against THRESH (test_compact_util.hpp):
//   1. X (all max(m,n) rows of B, the residual rows included) == ref_gels,
//      relative to ||X_lapack||_1
//   2. the factorization left in A and the tau left in taup == ref_gels,
//      relative to ||A||_1
//   3. least squares: the normal equations op(A)^T (B - op(A) X) = 0, and the
//      residual sums of squares in rows n..m-1 of B equal ||B - op(A) X||^2
//      minimum norm: op(A) X = B, and X equals the minimum-norm solution formed
//      the other way, X = op(A)^T Z with (op(A) op(A)^T) Z = B
// over the suite's shapes and the small-dimension cross product (small_dims:
// m x n from 0 to 5 per (layout, trans), so every over-/underdetermined case
// meets the empty operand and the single row or column), plus LAPACK-style
// argument validation of the C API and the min(m,n) = 0 case (B := 0).
//
// Assisted-by: Claude:claude-fable-5

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#include "test_lapack_util.hpp" // compact<T>, ref_gels/geqr2/orm2r/trsm_upper, pack/unpack

using namespace cbk::test;

namespace {

// --------------------------- one test case --------------------------

template <class T, int V>
int run_case(char layout, char trans, int nm, int m, int n, int nrhs)
{
    const bool row = (layout == 'R');
    const bool tran = (trans == 'T');
    const bool tall = (m >= n);
    const int p = std::max(m, n), q = std::min(m, n);
    const bool overdet = (tall != tran);
    const int rows_op = tran ? n : m,
              cols_op = tran ? m : n; // op(A) is rows_op x cols_op

    // random A (diagonal-boosted, so op(A) is well conditioned) and random B;
    // the reference works on copies
    MatrixBatch<T> A(nm, m, n), Aref(nm, m, n), B(nm, p, nrhs), Bref(nm, p, nrhs),
        tau_ref(nm, q, 1);
    for (int idx = 0; idx < nm; ++idx) {
        gen_boosted(A.view(idx));
        for (size_t e = 0; e < B.stride(); ++e)
            B[idx][e] = frand<T>();
        std::copy(A[idx], A[idx] + A.stride(), Aref[idx]);
        std::copy(B[idx], B[idx] + B.stride(), Bref[idx]);
        ref_gels(trans, Aref.view(idx), Bref.view(idx), tau_ref[idx]);
    }

    // pack in the requested layout, solve, unpack
    const int lda = std::max(1, row ? n : m), ldb = std::max(1, row ? nrhs : p);
    std::vector<T> ap = pack_compact(A, lda, V, row);
    std::vector<T> bp = pack_compact(B, ldb, V, row);
    std::vector<T> tp = compact_buffer<T>(nm, q, 1, q, V); /* the kernel fills it */

    const int info = compact<T>::gels(layout, trans, m, n, nrhs, ap.data(), lda,
                                      bp.data(), ldb, tp.data(), V, nm);

    MatrixBatch<T> Aout(nm, m, n), Bout(nm, p, nrhs), tau_out(nm, q, 1);
    unpack_compact(Aout, ap.data(), lda, V, row);
    unpack_compact(Bout, bp.data(), ldb, V, row);
    unpack_tau(tau_out, tp.data(), V);

    // check 1 & 2: X (all p rows), the factorization, and tau vs the
    // reference, relative to the reference operand's L1 norm (tau, O(1)
    // reflector scalars, to 1)
    double r_x = 0, r_h = 0, r_t = 0;
    for (int idx = 0; idx < nm; ++idx) {
        const double na = norm1(A.view(idx));
        r_x = std::max(r_x, test_ratio<T>(diff_norm1(Bout.view(idx), Bref.view(idx)), p,
                                          norm1(Bref.view(idx))));
        r_h = std::max(r_h,
                       test_ratio<T>(diff_norm1(Aout.view(idx), Aref.view(idx)), p, na));
        r_t = std::max(r_t, test_ratio<T>(diff_norm1(mat_view(tau_out[idx], q, 1),
                                                     mat_view(tau_ref[idx], q, 1)),
                                          p));
    }

    // check 3: the defining properties, formed without the reference: r_prop
    // is the normal equations (least squares: dqrt17's ||op(A)^T r|| /
    // (||op(A)|| ||B|| max(m,n,nrhs) eps)) or the residual (minimum norm:
    // dqrt16's ||B - op(A) X|| / (max(m,n) ||op(A)|| ||X|| eps)), r_prop2 the
    // residual-sum-of-squares rows or the Gram-formed solution
    double r_prop = 0, r_prop2 = 0;
    std::vector<T> Rs((size_t)rows_op * nrhs), Ss((size_t)cols_op * nrhs);
    const auto R = mat_view(Rs.data(), rows_op, nrhs);
    const auto S = mat_view(Ss.data(), cols_op, nrhs);
    for (int idx = 0; idx < nm; ++idx) {
        const auto Aop = tran ? A.view(idx).transposed() : A.view(idx);
        const auto Bin = leading(B.view(idx), rows_op, nrhs);  // the right-hand sides
        const auto X = leading(Bout.view(idx), cols_op, nrhs); // the solution rows
        const double scale = norm1(Aop) * norm1(Bin);
        matmul(Aop, X, R); // R = op(A) X
        if (overdet) {
            // r = B - op(A) X; normal equations op(A)^T r = 0
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < rows_op; ++i)
                    R(i, j) = Bin(i, j) - R(i, j);
            matmul(Aop.transposed(), R, S);
            r_prop = std::max(r_prop, test_ratio<T>(norm1(S), std::max(p, nrhs), scale));
            // rows cols_op..rows_op-1 of B hold the residual: squared column
            // norms == ||r_j||^2 (relative to ||b_j||^2: a square system has no
            // residual rows and a zero residual). Not at min(m,n) = 0: that is
            // ?gels's quick return, B := 0 (checked against ref_gels above),
            // and the residual is B itself.
            for (int j = 0; q > 0 && j < nrhs; ++j) {
                double rss = 0, rss_b = 0, bb = 0;
                for (int i = 0; i < rows_op; ++i) {
                    rss += (double)R(i, j) * R(i, j);
                    bb += (double)Bin(i, j) * Bin(i, j);
                }
                for (int i = cols_op; i < rows_op; ++i)
                    rss_b += (double)Bout(idx, i, j) * Bout(idx, i, j);
                r_prop2 = std::max(
                    r_prop2,
                    test_ratio<T>(std::abs(rss - rss_b) / std::max(bb, norm_floor), p));
            }
        }
        else {
            // op(A) X = B
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < rows_op; ++i)
                    R(i, j) = Bin(i, j) - R(i, j);
            r_prop = std::max(r_prop, test_ratio<T>(norm1(R), p, norm1(Aop) * norm1(X)));
            // minimum norm the other way: X = op(A)^T Z, G Z = B with the Gram
            // matrix G = op(A) op(A)^T (rows_op x rows_op, SPD), solved by the
            // LAPACKE QR references. The two solutions differ by what the
            // Gram solve amplifies, cond(G) = cond(op(A))^2: dget04's forward
            // error, discounted by rcond(G).
            std::vector<T> Gs((size_t)rows_op * rows_op), Zs((size_t)rows_op * nrhs),
                tg(rows_op), Xmns((size_t)cols_op * nrhs);
            const auto G = mat_view(Gs.data(), rows_op, rows_op);
            const auto Z = mat_view(Zs.data(), rows_op, nrhs);
            const auto Xmn = mat_view(Xmns.data(), cols_op, nrhs);
            matmul(Aop, Aop.transposed(), G);
            const double rcond_g = rcond1(G);
            copy_matrix(Bin, Z);
            ref_geqr2(G, tg.data());
            ref_ormqr('T', rows_op, G, tg.data(), Z);
            ref_trsm_upper(G, Z);
            matmul(Aop.transposed(), Z, Xmn);
            r_prop2 = std::max(r_prop2, forward_ratio(Xmn, X, rcond_g));
        }
    }

    // One gate for X, the factorization and the defining properties: two
    // backward-stable implementations (LAPACK's blocked above its crossover)
    // of a well-conditioned op(A)'s factorization and solve (design 7.1).
    const bool ok_x = passes(r_x), ok_h = passes(r_h), ok_t = passes(r_t);
    const bool ok_p = passes(r_prop), ok_p2 = passes(r_prop2);

    std::printf(
        "T=%-6s V=%-2d %s trans=%c nm=%-2d m=%-3d n=%-3d nrhs=%d %-6s | X:%-5.2g %s "
        "H:%-5.2g %s tau:%-5.2g %s | %s:%-5.2g %s %s:%-5.2g %s | info=%d\n",
        compact<T>::name, V, row ? "row" : "col", trans, nm, m, n, nrhs,
        overdet ? "lstsq" : "minnrm", r_x, verdict(r_x), r_h, verdict(r_h), r_t,
        verdict(r_t), overdet ? "normal" : "resid", r_prop, verdict(r_prop),
        overdet ? "rss" : "minnorm", r_prop2, verdict(r_prop2), info);
    return (info != 0) + !ok_x + !ok_h + !ok_t + !ok_p + !ok_p2;
}

// --------------------- C API argument validation --------------------

int test_validation()
{
    const int m = 8, n = 6, nrhs = 3, V = 4, nm = 5, ng = (nm + V - 1) / V;
    const int lda = 8, ldb = 8; // column-major: ldap >= m, ldbp >= max(m,n)
    std::vector<double> ap((size_t)ng * lda * n * V, 0),
        bp((size_t)ng * ldb * nrhs * V, 0), tau((size_t)ng * n * V, 0);
    auto call = [&](char lay, char tr, int m_, int n_, int nrhs_, int ldap_, int ldbp_,
                    int V_, int nm_) {
        return dgels_compact(lay, tr, m_, n_, nrhs_, ap.data(), ldap_, bp.data(), ldbp_,
                             tau.data(), V_, nm_);
    };
    // clang-format off
    const ApiCheck t[] = {
        {"valid col",   call('C', 'N', m, n, nrhs, lda, ldb,  V, nm),   0},
        {"valid col T", call('C', 'T', m, n, nrhs, lda, ldb,  V, nm),   0},
        {"valid row",   call('R', 'N', m, n, nrhs, n,   nrhs, V, nm),   0}, // row-major lds
        {"bad layout",  call('X', 'N', m, n, nrhs, lda, ldb,  V, nm),  -1},
        {"bad trans",   call('C', 'X', m, n, nrhs, lda, ldb,  V, nm),  -2},
        {"m<0",         call('C', 'N', -1, n, nrhs, lda, ldb, V, nm),  -3},
        {"n<0",         call('C', 'N', m, -1, nrhs, lda, ldb, V, nm),  -4},
        {"nrhs<0",      call('C', 'N', m, n, -1, lda, ldb,    V, nm),  -5},
        {"ldap<m",      call('C', 'N', m, n, nrhs, m-1, ldb,  V, nm),  -7},
        {"ldbp<max",    call('C', 'N', m, n, nrhs, lda, m-1,  V, nm),  -9},
        {"ldbp<max T",  call('C', 'T', n, m, nrhs, n,   m-1,  V, nm),  -9}, // wide A, B has m rows
        {"bad V",       call('C', 'N', m, n, nrhs, lda, ldb,  3, nm), -11},
        {"nm<0",        call('C', 'N', m, n, nrhs, lda, ldb,  V, -1), -12},
        {"empty nrhs",  call('C', 'N', m, n, 0, lda, ldb,     V, nm),   0},
        {"empty nm=0",  call('C', 'N', m, n, nrhs, lda, ldb,  V, 0),    0},
    };
    // clang-format on
    return report_api_checks(t);
}

// min(m,n) = 0 with a non-empty B: the solution of the empty system, B := 0
// over all max(m,n) rows -- LAPACK ?gels's quick return -- in both layouts.
int test_empty_op()
{
    const int V = 4, nm = 6, n = 5, nrhs = 3;
    int bad = 0;
    for (char lay : {'C', 'R'}) {
        const bool row = (lay == 'R');
        const int ldb = row ? nrhs : n;
        std::vector<double> bp = compact_buffer<double>(nm, n, nrhs, ldb, V, row);
        std::fill(bp.begin(), bp.end(), 1.0);
        // m = 0: A and tau are empty and never touched, but Fortran semantics
        // want a (dummy) argument present, so hand in 1-element stand-ins
        double a_dummy = 0, tau_dummy = 0;
        const int lda = row ? n : 1; // ldap is still checked
        int info = dgels_compact(lay, 'N', 0, n, nrhs, &a_dummy, lda, bp.data(), ldb,
                                 &tau_dummy, V, nm);
        double mx = 0;
        for (double x : bp)
            mx = std::max(mx, std::abs(x));
        const bool ok = (info == 0) && (mx == 0.0);
        bad += !ok;
        std::printf("empty op(A) (m=0, n=%d) %s: B := 0 | max|B| %.1e info=%d %s\n", n,
                    row ? "row" : "col", mx, info, ok ? "OK" : "FAIL");
    }
    return bad;
}

} // namespace

// ------------------------------- main --------------------------------

int main()
{
    int fails = 0;
    fails += test_validation();
    fails += test_empty_op();

    // every (layout, trans) over square, tall and wide A, so all four
    // over-/underdetermined cases run in both layouts
    for (char lay : {'C', 'R'})
        for (char tr : {'N', 'T'}) {
            fails += run_case<double, 4>(lay, tr, 8, 30, 30, 3);  // square
            fails += run_case<double, 4>(lay, tr, 8, 40, 24, 5);  // tall
            fails += run_case<double, 4>(lay, tr, 8, 24, 40, 5);  // wide
            fails += run_case<double, 8>(lay, tr, 11, 43, 17, 4); // padded partial group
            fails += run_case<double, 2>(lay, tr, 4, 12, 20, 1);  // single RHS, V=2
            fails += run_case<float, 8>(lay, tr, 11, 32, 20, 3);  // padded partial group
            fails += run_case<float, 16>(lay, tr, 32, 20, 32, 2);
        }
    fails += run_case<double, 4>('C', 'N', 40, 24, 16, 3); // 10 groups: OpenMP path
    fails += run_case<double, 4>('C', 'T', 40, 24, 16, 3);
    fails += run_case<double, 4>('C', 'N', 8, 3, 3, 2);   // smallest supported
    fails += run_case<double, 4>('C', 'N', 8, 5, 1, 2);   // a single column
    fails += run_case<double, 4>('C', 'N', 8, 1, 5, 2);   // a single row
    fails += run_case<double, 4>('C', 'N', 8, 64, 20, 6); // wider RHS blocks (4+2 tail)
    fails += run_case<double, 4>('C', 'T', 8, 64, 20, 7); // (4+2+1 tail)

    // the small-dimension cross product (small_dims): m x n per (layout,
    // trans), so every over-/underdetermined case meets the empty operand, a
    // single row or column, and the orders around the register block; a
    // padded group
    for (int mm : small_dims)
        for (int nn : small_dims) {
            for (char lay : {'C', 'R'})
                for (char tr : {'N', 'T'})
                    fails += run_case<double, 4>(lay, tr, 5, mm, nn, 2);
            fails += run_case<float, 8>('R', 'T', 9, mm, nn, 1);
        }

    return finish(fails);
}
