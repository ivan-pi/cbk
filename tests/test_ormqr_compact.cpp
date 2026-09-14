/* test_ormqr_compact.cpp
 *
 * Validation of the templated compact ormqr against LAPACKE
 * (test_lapack_util.hpp): the reflectors come from LAPACKE_?geqr2 (or
 * LAPACKE_?geqp3 for the pivoted pipeline), the reference application from
 * LAPACKE_?ormqr, the back substitution from cblas_?trsm.
 *
 * Test design: X(:,j) = j+1 (ones, twos, threes, ...), B = A*X, so the
 * b columns are scaled row sums of A; the solve must recover X.
 *
 * Checks per (T, V), each a test ratio against THRESH (test_compact_util.hpp):
 *   1. compact Q^T B  ==  LAPACKE_?ormqr Q^T B, relative to ||Q^T B||_1 (the
 *      design's 7.1 gate: two backward-stable applications of the same
 *      reflectors, LAPACK's blocked above its crossover)
 *   2. back substitution recovers X (dget04's forward error, rcond(A))
 *   3. applying 'N' after 'T' recovers the original B  (Q Q^T = I)
 * over the suite's shapes and the small-dimension cross product (small_dims:
 * m x nrhs from 0 to 5, the empty operand and the 1/2/3-column right-hand
 * side blocks included).
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

#include "test_lapack_util.hpp" // compact<T>, ref_geqr2/orm2r/geqp3/trsm_upper, pack/unpack

using namespace cbk::test;

/* --------------------------- one test case -------------------------- */

template <class T, int V> static int run_case(int nm, int m, int nrhs)
{
    const int k = m;

    const std::vector<T> Xs = known_solution<T>(m, nrhs);
    const auto X = mat_view(Xs.data(), m, nrhs);

    MatrixBatch<T> A(nm, m, m), Afac(nm, m, m), B(nm, m, nrhs), Bref(nm, m, nrhs),
        Bout(nm, m, nrhs), tau(nm, m, 1);
    for (int kk = 0; kk < nm; ++kk) {
        gen_boosted(A.view(kk));           /* diagonal boost tames cond for float */
        matmul(A.view(kk), X, B.view(kk)); /* B = A X */

        std::copy(A[kk], A[kk] + (size_t)m * m, Afac[kk]);
        ref_geqr2(Afac.view(kk), tau[kk]);
        std::copy(B[kk], B[kk] + (size_t)m * nrhs, Bref[kk]);
        ref_ormqr('T', k, Afac.view(kk), tau[kk], Bref.view(kk));
    }

    const int ld = std::max(1, m);
    std::vector<T> ap = pack_compact(Afac, ld, V);
    std::vector<T> tp = pack_tau(tau, V);
    std::vector<T> bp = pack_compact(B, ld, V);

    /* check 1: compact Q^T B vs LAPACKE_?ormqr, relative to ||Q^T B||_1 */
    compact<T>::ormqr('T', m, nrhs, k, ap.data(), ld, tp.data(), bp.data(), ld, V, nm);
    unpack_compact(Bout, bp.data(), ld, V);
    double r1 = 0;
    for (int kk = 0; kk < nm; ++kk)
        r1 =
            std::max(r1, test_ratio<T>(max_abs_diff(Bout[kk], Bref[kk], (size_t)m * nrhs),
                                       m, norm1(Bref.view(kk))));

    /* check 2: solve recovers X (dget04, discounted by rcond(A)) */
    double r2 = 0;
    for (int kk = 0; kk < nm; ++kk) {
        ref_trsm_upper(Afac.view(kk), Bout.view(kk));
        r2 = std::max(
            r2, forward_ratio<T>(max_abs_diff(Bout[kk], Xs.data(), (size_t)m * nrhs),
                                 norm1(X), rcond1(A.view(kk))));
    }

    /* check 3: 'N' undoes 'T', relative to ||B||_1 */
    compact<T>::ormqr('N', m, nrhs, k, ap.data(), ld, tp.data(), bp.data(), ld, V, nm);
    unpack_compact(Bout, bp.data(), ld, V);
    double r3 = 0;
    for (int kk = 0; kk < nm; ++kk)
        r3 = std::max(r3, test_ratio<T>(max_abs_diff(Bout[kk], B[kk], (size_t)m * nrhs),
                                        m, norm1(B.view(kk))));

    const bool ok1 = passes(r1), ok2 = passes(r2), ok3 = passes(r3);
    std::printf(
        "T=%-6s V=%-2d nm=%-2d m=%-3d nrhs=%d | QtB: %-5.2g %s | solve X: %-5.2g %s "
        "| QQt=I: %-5.2g %s\n",
        compact<T>::name, V, nm, m, nrhs, r1, verdict(r1), r2, verdict(r2), r3,
        verdict(r3));
    return !ok1 + !ok2 + !ok3;
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
    compact<T>::ormqr('T', m, nrhs, k, ap.data(), m, tp.data(), bp.data(), m, V, nm);
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
        r = std::max(
            r, forward_ratio<T>(max_abs_diff(xs.data(), Xs.data(), (size_t)m * nrhs),
                                norm1(X), rcond1(A.view(kk))));
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
        compact<T>::ormqr((r & 1) ? 'N' : 'T', m, nrhs, k, ap.data(), m, tp.data(),
                          bp.data(), m, V, nm);
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
    const int m = 8, nrhs = 2, k = 8, V = 4, nm = 4, ld = 8;
    std::vector<double> ap((size_t)ld * k * V, 0), tau((size_t)k * V, 0),
        bp((size_t)ld * nrhs * V, 0);
    auto call = [&](char tr, int m_, int nrhs_, int k_, int ldap_, int ldbp_, int V_,
                    int nm_) {
        return dormqr_compact(tr, m_, nrhs_, k_, ap.data(), ldap_, tau.data(), bp.data(),
                              ldbp_, V_, nm_);
    };

    // clang-format off
    const ApiCheck t[] = {
        {"valid",          call('T', m, nrhs, k,   ld,    ld,    V, nm),   0},
        {"bad trans",      call('X', m, nrhs, k,   ld,    ld,    V, nm),  -1},
        {"m<0",            call('T', -1, nrhs, k,  ld,    ld,    V, nm),  -2},
        {"nrhs<0",         call('T', m, -1, k,     ld,    ld,    V, nm),  -3},
        {"k>m",            call('T', m, nrhs, m+1, ld,    ld,    V, nm),  -4},
        {"ldap<m",         call('T', m, nrhs, k,   m-1,   ld,    V, nm),  -6},
        {"ldbp<m",         call('T', m, nrhs, k,   ld,    m-1,   V, nm),  -9},
        {"bad V",          call('T', m, nrhs, k,   ld,    ld,    3, nm), -10},
        {"nm<0",           call('T', m, nrhs, k,   ld,    ld,    V, -1), -11},
        {"empty m=0",      call('T', 0, nrhs, 0,   1,     1,     V, nm),   0},
        {"empty nm=0",     call('T', m, nrhs, k,   ld,    ld,    V, 0),    0},
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

    fails += run_case<double, 2>(4, m, nrhs);
    fails += run_case<double, 4>(8, m, nrhs);
    fails += run_case<double, 8>(16, m, nrhs);
    fails += run_case<double, 8>(11, m, nrhs); /* padded partial group */
    fails +=
        run_case<double, 4>(40, m, nrhs); /* 10 groups: OpenMP path when threads <= 10 */
    fails += run_case<float, 4>(8, m, nrhs);
    fails += run_case<float, 8>(16, m, nrhs);
    fails += run_case<float, 16>(32, m, nrhs);

    /* the small-dimension cross product (small_dims): m x nrhs, the empty
     * operand and the 1/2/3-column RHS blocks included, a padded group */
    for (int mm : small_dims)
        for (int nr : small_dims) {
            fails += run_case<double, 4>(5, mm, nr);
            fails += run_case<float, 8>(9, mm, nr);
        }

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
