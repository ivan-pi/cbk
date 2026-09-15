/* test_ormqr_compact.cpp
 *
 * Validation of the templated compact ormqr (dormqr_compact / sormqr_compact)
 * against LAPACKE (test_lapack_util.hpp): the reflectors come from
 * LAPACKE_?geqr2 (or LAPACKE_?geqp3 for the pivoted pipeline), the reference
 * application from LAPACKE_?ormqr, the explicit Q from LAPACKE_?orgqr, the
 * back substitution from cblas_?trsm.
 *
 * Checks per (T, V, layout, side, trans, m, n, k), each a test ratio against
 * THRESH (test_compact_util.hpp), LAPACK's QR path's:
 *   1. compact op(Q) C  ==  LAPACKE_?ormqr op(Q) C, relative to its norm (two
 *      backward-stable applications of the same reflectors, LAPACK's blocked
 *      above its crossover)
 *   2. dqrt03: op(Q) C (side 'L') or C op(Q) ('R') against ?gemm with the Q
 *      that ?orgqr forms explicitly, ||CC - op(Q) C||_1 / (nq ||C||_1 eps),
 *      nq the order of Q -- the four side/trans combinations in both layouts
 *   3. applying op(Q)^T after op(Q) recovers the original C  (Q Q^T = I)
 * and, in the solve pipeline (side 'L', trans 'T', column-major, the tuned
 * route), X(:,j) = j+1, B = A X:
 *   4. Q^T B then back substitution recovers X (dget04's forward error,
 *      rcond(A)), from ?geqr2's reflectors and, through a jpvt
 *      back-permutation, from ?geqp3's
 * over the suite's shapes and the small-dimension cross product (small_dims:
 * m x n from 0 to 5 with LAPACK's K values, the empty operand included), plus
 * LAPACK-style argument validation of the C API.
 *
 * Assisted-by: Claude:claude-fable-5 Claude:claude-opus-4-8
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <limits>
#include <algorithm>

#include "test_lapack_util.hpp" // compact<T>, ref_geqr2/ormqr/orgqr/geqp3/trsm_upper, pack/unpack

using namespace cbk::test;

/* ------------------- one (layout, side, trans) case ------------------ */
/* A is the nq x k reflector batch, nq = m (side 'L') or n ('R'); C is m x n. */

template <class T, int V>
static int run_case(char layout, char side, char trans, int nm, int m, int n, int k)
{
    const bool row = (layout == 'R');
    const bool left = (side == 'L');
    const bool tran = (trans == 'T');
    const int nq = left ? m : n; /* the order of Q, and the rows of A */

    /* (H, tau) of a random nq x k matrix, a random C, and the two references:
     * LAPACKE_?ormqr's op(Q) C, and the explicit Q of ?orgqr applied by ?gemm
     * (dqrt03) */
    MatrixBatch<T> H(nm, nq, k), tau(nm, k, 1), C(nm, m, n), Cref(nm, m, n), QC(nm, m, n);
    std::vector<T> Qs((size_t)nq * nq);
    const auto Q = mat_view(Qs.data(), nq, nq);
    for (int idx = 0; idx < nm; ++idx) {
        gen_boosted(H.view(idx)); /* diagonal boost tames cond for float */
        ref_geqr2(H.view(idx), tau[idx]);
        for (size_t e = 0; e < C.stride(); ++e)
            C[idx][e] = frand<T>();
        std::copy(C[idx], C[idx] + C.stride(), Cref[idx]);
        ref_ormqr(side, trans, k, H.view(idx), tau[idx], Cref.view(idx));

        std::fill(Qs.begin(), Qs.end(), T(0));
        copy_matrix(leading(H.view(idx), nq, k), leading(Q, nq, k));
        ref_orgqr(k, Q, tau[idx]); /* Q = H(0)..H(k-1), nq x nq */
        const auto opQ = tran ? Q.transposed() : Q;
        if (left)
            matmul(opQ, C.view(idx), QC.view(idx));
        else
            matmul(C.view(idx), opQ, QC.view(idx));
    }

    /* pack in the requested layout, apply, unpack */
    const int lda = std::max(1, row ? k : nq), ldc = std::max(1, row ? n : m);
    std::vector<T> ap = pack_compact(H, lda, V, row);
    std::vector<T> tp = pack_tau(tau, V);
    std::vector<T> cp = pack_compact(C, ldc, V, row);
    int bad = (compact<T>::ormqr(layout, side, trans, m, n, k, ap.data(), lda, tp.data(),
                                 cp.data(), ldc, V, nm) != 0);
    MatrixBatch<T> Cout(nm, m, n);
    unpack_compact(Cout, cp.data(), ldc, V, row);

    /* checks 1 and 2: vs LAPACKE_?ormqr, and vs the explicit Q (dqrt03) */
    double r_ref = 0, r_q = 0;
    for (int idx = 0; idx < nm; ++idx) {
        r_ref = std::max(r_ref, test_ratio<T>(diff_norm1(Cout.view(idx), Cref.view(idx)),
                                              nq, norm1(Cref.view(idx))));
        r_q = std::max(r_q, test_ratio<T>(diff_norm1(Cout.view(idx), QC.view(idx)), nq,
                                          norm1(C.view(idx))));
    }

    /* check 3: the inverse application recovers C */
    bad += (compact<T>::ormqr(layout, side, tran ? 'N' : 'T', m, n, k, ap.data(), lda,
                              tp.data(), cp.data(), ldc, V, nm) != 0);
    unpack_compact(Cout, cp.data(), ldc, V, row);
    double r_rt = 0;
    for (int idx = 0; idx < nm; ++idx)
        r_rt = std::max(r_rt, test_ratio<T>(diff_norm1(Cout.view(idx), C.view(idx)), nq,
                                            norm1(C.view(idx))));

    const bool ok = passes(r_ref) && passes(r_q) && passes(r_rt) && (bad == 0);
    std::printf(
        "T=%-6s V=%-2d nm=%-2d %s side=%c trans=%c m=%-3d n=%-3d k=%-3d | ref %-5.2g "
        "%s | Q*C %-5.2g %s | QQt=I %-5.2g %s | %s\n",
        compact<T>::name, V, nm, row ? "row" : "col", side, trans, m, n, k, r_ref,
        verdict(r_ref), r_q, verdict(r_q), r_rt, verdict(r_rt), ok ? "OK" : "FAIL");
    return !ok;
}

/* ------------------- the solve pipeline: Q^T B, then R X = Q^T B --------- */
/* From ?geqr2's reflectors of a square A, B = A X for a known X: the kernel
 * applies Q^T, cblas_?trsm back-substitutes, and X must come back (dget04). */

template <class T, int V> static int run_solve(int nm, int m, int nrhs)
{
    const int k = m;
    const std::vector<T> Xs = known_solution<T>(m, nrhs);
    const auto X = mat_view(Xs.data(), m, nrhs);

    MatrixBatch<T> A(nm, m, m), Afac(nm, m, m), B(nm, m, nrhs), Bout(nm, m, nrhs),
        tau(nm, m, 1);
    for (int kk = 0; kk < nm; ++kk) {
        gen_boosted(A.view(kk));           /* diagonal boost tames cond for float */
        matmul(A.view(kk), X, B.view(kk)); /* B = A X */
        std::copy(A[kk], A[kk] + A.stride(), Afac[kk]);
        ref_geqr2(Afac.view(kk), tau[kk]);
    }

    const int ld = std::max(1, m);
    std::vector<T> ap = pack_compact(Afac, ld, V);
    std::vector<T> tp = pack_tau(tau, V);
    std::vector<T> bp = pack_compact(B, ld, V);
    const int info = compact<T>::ormqr('C', 'L', 'T', m, nrhs, k, ap.data(), ld,
                                       tp.data(), bp.data(), ld, V, nm);
    unpack_compact(Bout, bp.data(), ld, V);

    double r = 0;
    for (int kk = 0; kk < nm; ++kk) {
        ref_trsm_upper(Afac.view(kk), Bout.view(kk));
        r = std::max(r, forward_ratio(Bout.view(kk), X, rcond1(A.view(kk))));
    }

    const bool ok = passes(r) && (info == 0);
    std::printf("T=%-6s V=%-2d nm=%-2d m=%-3d nrhs=%d | solve X: %-5.2g %s | info=%d\n",
                compact<T>::name, V, nm, m, nrhs, r, verdict(r), info);
    return !ok;
}

/* ------------------- pivoted-QR + back-permutation solve ------------ */
/* Salvaged from the former ArmPL cross-check: feed the kernel reflectors
 * from a column-pivoted (rank-revealing) QR, LAPACKE_?geqp3 through
 * ref_geqp3 (0-based jpvt), and recover X through a jpvt back-permutation,
 * i.e. solve A x = b with A(:,jpvt) = Q R:
 *   R y = Q^T b   (kernel applies Q^T),   x(jpvt(j)) = y(j).
 * Unpivoted tests never exercise this end-to-end permuted pipeline. */
template <class T, int V> static int run_case_pivoted(int nm, int m, int nrhs)
{
    const int k = m; /* square, full rank */

    const std::vector<T> Xs = known_solution<T>(m, nrhs);
    const auto X = mat_view(Xs.data(), m, nrhs);

    MatrixBatch<T> A(nm, m, m), Afac(nm, m, m), B(nm, m, nrhs), Bout(nm, m, nrhs),
        tau(nm, m, 1);
    MatrixBatch<lapack_int> jpvt(nm, m, 1);
    for (int kk = 0; kk < nm; ++kk) {
        gen_boosted(A.view(kk));           /* diagonal boost tames cond */
        matmul(A.view(kk), X, B.view(kk)); /* B = A X */

        std::copy(A[kk], A[kk] + A.stride(), Afac[kk]);
        ref_geqp3(Afac.view(kk), jpvt[kk], tau[kk]);
    }

    std::vector<T> ap = pack_compact(Afac, m, V);
    std::vector<T> tp = pack_tau(tau, V);
    std::vector<T> bp = pack_compact(B, m, V);

    /* kernel: c := Q^T b */
    compact<T>::ormqr('C', 'L', 'T', m, nrhs, k, ap.data(), m, tp.data(), bp.data(), m, V,
                      nm);
    unpack_compact(Bout, bp.data(), m, V);

    /* R y = c, then back-permute x(jpvt(j)) = y(j); compare against X (dget04,
     * discounted by rcond(A)) */
    double r = 0;
    std::vector<T> xs((size_t)m * nrhs);
    const auto x = mat_view(xs.data(), m, nrhs);
    for (int kk = 0; kk < nm; ++kk) {
        ref_trsm_upper(Afac.view(kk), Bout.view(kk));
        const auto y = Bout.view(kk);
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < m; ++i)
                x(jpvt[kk][i], j) = y(i, j);
        r = std::max(r, forward_ratio(x, X, rcond1(A.view(kk))));
    }

    const bool ok = passes(r);
    std::printf("T=%-6s V=%-2d nm=%-2d m=%-3d nrhs=%d | pivoted solve X: %-5.2g %s\n",
                compact<T>::name, V, nm, m, nrhs, r, verdict(r));
    return !ok;
}

/* --------------------------- micro-benchmark ------------------------ */

template <class T> static void bench(int V, int nm, int m, int nrhs, int reps)
{
    const int k = m;
    std::vector<T> ap = compact_buffer<T>(nm, m, m, m, V);
    std::vector<T> tp = compact_buffer<T>(nm, k, 1, k, V);
    std::vector<T> bp = compact_buffer<T>(nm, m, nrhs, m, V);
    for (auto &x : ap)
        x = frand<T>() * T(1e-3);
    for (auto &x : tp)
        x = frand<T>() * T(1e-3);
    for (auto &x : bp)
        x = frand<T>();

    timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < reps; ++r)
        compact<T>::ormqr('C', 'L', (r & 1) ? 'N' : 'T', m, nrhs, k, ap.data(), m,
                          tp.data(), bp.data(), m, V, nm);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);

    double fl_mat = 0;
    for (int kk = 0; kk < k; ++kk)
        fl_mat += 4.0 * (m - kk) * nrhs;
    std::printf("T=%-6s V=%-2d nm=%-2d m=%-3d nrhs=%d | %8.3f us/rep | %7.2f GFLOP/s\n",
                compact<T>::name, V, nm, m, nrhs, 1e6 * sec / reps,
                fl_mat * nm * reps / sec * 1e-9);
    volatile T sink = bp[0];
    (void)sink;
}

/* --------------------- C API argument validation -------------------- */
/* The public C entry points must reject illegal arguments LAPACK-style
 * (return -j for the j-th argument), not assert or miscompute. */
static int test_validation()
{
    const int m = 8, n = 6, k = 5, V = 4, nm = 4; /* A is nq x k, nq = m or n */
    std::vector<double> ap((size_t)m * m * V, 0), tau((size_t)m * V, 0),
        cp((size_t)m * m * V, 0);
    auto call = [&](char lay, char si, char tr, int m_, int n_, int k_, int ldap_,
                    int ldcp_, int V_, int nm_) {
        return dormqr_compact(lay, si, tr, m_, n_, k_, ap.data(), ldap_, tau.data(),
                              cp.data(), ldcp_, V_, nm_);
    };

    // clang-format off
    const ApiCheck t[] = {
        {"valid col L",    call('C', 'L', 'T', m, n, k,   m,   m,   V, nm),   0},
        {"valid col R",    call('C', 'R', 'N', m, n, k,   n,   m,   V, nm),   0}, // A is n x k
        {"valid row L",    call('R', 'L', 'T', m, n, k,   k,   n,   V, nm),   0}, // row-major lds
        {"bad layout",     call('X', 'L', 'T', m, n, k,   m,   m,   V, nm),  -1},
        {"bad side",       call('C', 'X', 'T', m, n, k,   m,   m,   V, nm),  -2},
        {"bad trans",      call('C', 'L', 'X', m, n, k,   m,   m,   V, nm),  -3},
        {"m<0",            call('C', 'L', 'T', -1, n, k,  m,   m,   V, nm),  -4},
        {"n<0",            call('C', 'L', 'T', m, -1, k,  m,   m,   V, nm),  -5},
        {"k>m (L)",        call('C', 'L', 'T', m, n, m+1, m,   m,   V, nm),  -6},
        {"k>n (R)",        call('C', 'R', 'T', m, n, n+1, n,   m,   V, nm),  -6},
        {"ldap<m (L)",     call('C', 'L', 'T', m, n, k,   m-1, m,   V, nm),  -8},
        {"ldap<n (R)",     call('C', 'R', 'T', m, n, k,   n-1, m,   V, nm),  -8},
        {"ldap<k (row)",   call('R', 'L', 'T', m, n, k,   k-1, n,   V, nm),  -8},
        {"ldcp<m (col)",   call('C', 'L', 'T', m, n, k,   m,   m-1, V, nm), -11},
        {"ldcp<n (row)",   call('R', 'L', 'T', m, n, k,   k,   n-1, V, nm), -11},
        {"bad V",          call('C', 'L', 'T', m, n, k,   m,   m,   3, nm), -12},
        {"nm<0",           call('C', 'L', 'T', m, n, k,   m,   m,   V, -1), -13},
        {"empty m=0",      call('C', 'L', 'T', 0, n, 0,   1,   1,   V, nm),   0},
        {"empty k=0",      call('C', 'L', 'T', m, n, 0,   m,   m,   V, nm),   0},
        {"empty nm=0",     call('C', 'L', 'T', m, n, k,   m,   m,   V, 0),    0},
    };
    // clang-format on
    return report_api_checks(t);
}

/* ------------------------------- main -------------------------------- */

int main(int argc, char **)
{
    int fails = 0;

    const int m = 43, nrhs = 5; /* nz=28 + npoly=15 */

    fails += test_validation();

    /* the full layout x side x trans product (dqrt03): A and Q are m x m
     * (k <= m) on the left, n x n (k <= n) on the right; a padded partial
     * group and k < nq among the shapes */
    for (char lay : {'C', 'R'})
        for (char side : {'L', 'R'})
            for (char tr : {'N', 'T'}) {
                fails +=
                    run_case<double, 4>(lay, side, tr, 8, 43, 5, side == 'L' ? 43 : 5);
                fails +=
                    run_case<double, 8>(lay, side, tr, 11, 32, 6, side == 'L' ? 20 : 4);
                fails +=
                    run_case<float, 8>(lay, side, tr, 16, 32, 8, side == 'L' ? 32 : 8);
            }
    /* the other widths, and 10 groups (the OpenMP path when threads <= 10),
     * on the tuned route */
    fails += run_case<double, 2>('C', 'L', 'T', 4, m, nrhs, m);
    fails += run_case<double, 4>('C', 'L', 'T', 40, m, nrhs, m);
    fails += run_case<float, 16>('C', 'L', 'T', 32, m, nrhs, m);

    /* the small-dimension cross product (small_dims): m x n with LAPACK's K
     * values {nq, 0, 1, nq/2} on the tuned route, k = nq on the transposed
     * (side 'R', row-major) one; the empty operand and the 1/2/3-column
     * blocks included, a padded group */
    for (int mm : small_dims)
        for (int nn : small_dims) {
            const int ks[] = {mm, 0, 1, mm / 2};
            for (int t = 0; t < 4; ++t) {
                const int kk = ks[t];
                if (kk > mm || std::find(ks, ks + t, kk) != ks + t) continue;
                fails += run_case<double, 4>('C', 'L', 'T', 5, mm, nn, kk);
            }
            fails += run_case<float, 8>('R', 'R', 'N', 9, mm, nn, nn);
        }

    /* the solve pipeline: Q^T B then back substitution recovers X */
    fails += run_solve<double, 4>(8, m, nrhs);
    fails += run_solve<double, 8>(11, m, nrhs); /* padded partial group */
    fails += run_solve<float, 8>(16, m, nrhs);

    /* column-pivoted QR + back-permutation solve (salvaged cross-check) */
    fails += run_case_pivoted<double, 4>(8, m, nrhs);
    fails += run_case_pivoted<double, 8>(11, m, nrhs); /* padded partial group */
    fails += run_case_pivoted<float, 8>(16, m, nrhs);

    if (argc > 1) { /* run benchmark only when asked (skip under qemu) */
        std::printf("\n-- micro-benchmark (single core) --\n");
        bench<double>(8, 8, 43, 8, 200000);
        bench<double>(4, 8, 43, 8, 200000);
        bench<double>(2, 8, 43, 8, 200000);
        bench<float>(16, 16, 43, 8, 100000);
        bench<float>(8, 16, 43, 8, 100000);
    }

    return finish(fails);
}
