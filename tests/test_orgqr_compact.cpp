/* test_orgqr_compact.cpp
 *
 * Self-contained validation of the templated compact orgqr.
 * Reference: unblocked Householder QR (ref_geqr2, LAPACK reflector convention)
 * + a scalar dorg2r, both templated on T.
 *
 * Test design: factor a random m x k batch (diagonal-boosted), stage the
 * reflectors into the first k columns of an m x n buffer whose remaining
 * columns hold garbage (they must be overwritten), and generate Q.
 *
 * Checks per (T, V, layout):
 *   1. compact Q  ==  scalar dorg2r Q (elementwise, ~eps: same op sequence)
 *   2. Q^T Q = I (orthonormal columns, formed independently)
 *   3. Q(:, 0:k-1) R = A with R = triu of the factorization (k > 0)
 *   4. padded lanes of a partial last group come out exactly identity
 * plus k < n (unit-seeded extra columns), k = 0 (Q = leading columns of I),
 * row-major, and the C API's LAPACK-style argument validation.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#include "test_compact_util.hpp" // compact<T>, scalar references, MatrixBatch, pack/unpack

using namespace cbk::test;

/* ----------------------- reference kernel (scalar) ------------------ */

/* dorg2r: generate the first n columns of Q = H(0)..H(k-1) in place over the
 * reflectors of ref_geqr2 (columns 0..k-1 of A, m x n, m >= n >= k). Backward
 * accumulation: unit-seed columns k..n-1, then apply each H(kk) to the
 * trailing columns and form column kk = H(kk) e_kk. */
template <class T> static void ref_org2r(int k, MatrixView<T> A, const T *tau)
{
    const int m = A.rows, n = A.cols;
    assert(k <= n && n <= m);
    for (int j = k; j < n; ++j) {
        for (int i = 0; i < m; ++i)
            A(i, j) = T(0);
        A(j, j) = T(1);
    }
    for (int kk = k - 1; kk >= 0; --kk) {
        for (int j = kk + 1; j < n; ++j) {
            T w = A(kk, j);
            for (int i = kk + 1; i < m; ++i)
                w += A(i, kk) * A(i, j);
            A(kk, j) -= tau[kk] * w;
            for (int i = kk + 1; i < m; ++i)
                A(i, j) -= tau[kk] * A(i, kk) * w;
        }
        for (int i = kk + 1; i < m; ++i)
            A(i, kk) *= -tau[kk];
        A(kk, kk) = T(1) - tau[kk];
        for (int i = 0; i < kk; ++i)
            A(i, kk) = T(0);
    }
}

/* --------------------------- one test case -------------------------- */

template <class T, int V>
static int run_case(int nm, int m, int n, int k, bool rowmajor = false)
{
    assert(m >= n && n >= k);
    const T eps = std::numeric_limits<T>::epsilon();
    const double tol_exact = 100.0 * eps;    /* same op sequence as the reference */
    const double tol_orth = 100.0 * eps * m; /* independent invariants */
    const double tol_rec = 100.0 * eps * m;

    const int ld = rowmajor ? n : m;

    /* factor a random m x k batch (empty at k = 0); stage its reflectors
     * into m x n */
    MatrixBatch<T> A0(nm, m, k), Afac(nm, m, k), tau(nm, k, 1), Hn(nm, m, n),
        Qref(nm, m, n), Qout(nm, m, n);
    for (int kk = 0; kk < nm; ++kk) {
        gen_boosted(A0.view(kk)); /* diagonal boost tames cond for float */
        std::copy(A0[kk], A0[kk] + A0.stride(), Afac[kk]);
        if (k > 0) ref_geqr2(Afac.view(kk), tau[kk]);

        const auto H = Hn.view(kk);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                H(i, j) = (j < k) ? Afac(kk, i, j) : frand<T>(); /* garbage cols */

        std::copy(Hn[kk], Hn[kk] + Hn.stride(), Qref[kk]);
        ref_org2r(k, Qref.view(kk), tau[kk]);
    }

    std::vector<T> ap = pack_compact(Hn, ld, V, rowmajor);
    std::vector<T> tp = pack_tau(tau, V); /* empty at k = 0, never read then */

    const int info =
        compact<T>::orgqr(rowmajor ? 'R' : 'C', m, n, k, ap.data(), ld, tp.data(), V, nm);
    int fails = 0;
    if (info != 0) {
        ++fails;
        std::printf("    info = %d (expected 0)\n", info);
    }
    unpack_compact(Qout, ap.data(), ld, V, rowmajor);

    /* check 1: elementwise vs scalar dorg2r */
    double e1 = 0;
    for (int kk = 0; kk < nm; ++kk)
        e1 = std::max(e1, max_abs_diff(Qout[kk], Qref[kk], Qout.stride()));

    /* check 2: Q^T Q = I, formed densely */
    double e2 = 0;
    for (int kk = 0; kk < nm; ++kk)
        e2 = std::max(e2, orth_error(Qout.view(kk)));

    /* check 3: Q(:, 0:k-1) R = A0, R = triu of the factorization */
    double e3 = 0;
    if (k > 0) {
        std::vector<T> Rs((size_t)k * k), Recs((size_t)m * k);
        const auto R = mat_view(Rs.data(), k, k);
        const auto Rec = mat_view(Recs.data(), m, k);
        for (int kk = 0; kk < nm; ++kk) {
            for (int j = 0; j < k; ++j)
                for (int i = 0; i < k; ++i)
                    R(i, j) = (i <= j) ? Afac(kk, i, j) : T(0);
            matmul(leading(Qout.view(kk), m, k), R, Rec);
            e3 = std::max(e3, max_abs_diff(Recs.data(), A0[kk], (size_t)m * k) /
                                  std::max(norm1(A0.view(kk)), norm_floor));
        }
    }

    /* check 4: padded lanes of a partial last group come out exactly identity */
    double e4 = 0;
    if (nm % V) {
        const int ng = (nm + V - 1) / V;
        const std::size_t gstride = group_stride(rowmajor, ld, m, n, V);
        const auto P = make_const_view<T, V>(ap.data() + (std::size_t)(ng - 1) * gstride,
                                             rowmajor, ld);
        for (int v = nm % V; v < V; ++v)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    e4 = std::max(e4,
                                  (double)std::abs(P(i, j)[v] - (i == j ? T(1) : T(0))));
    }

    bool ok1 = e1 <= tol_exact, ok2 = e2 <= tol_orth, ok3 = e3 <= tol_rec, ok4 = e4 == 0;
    fails += !ok1 + !ok2 + !ok3 + !ok4;
    std::printf("T=%-6s V=%-2d nm=%-2d %s m=%-3d n=%-3d k=%-3d | ref: %.2e %s | QtQ=I: "
                "%.2e %s | QR=A: %.2e %s | pad: %.1e %s\n",
                compact<T>::name, V, nm, rowmajor ? "row" : "col", m, n, k, e1,
                ok1 ? "OK" : "FAIL", e2, ok2 ? "OK" : "FAIL", e3, ok3 ? "OK" : "FAIL", e4,
                ok4 ? "OK" : "FAIL");
    return fails;
}

/* --------------------- C API argument validation -------------------- */
/* The public C entry points must reject illegal arguments LAPACK-style
 * (return -j for the j-th argument), not assert or miscompute. */
static int test_validation()
{
    const int m = 8, n = 8, k = 8, V = 4, nm = 4, ld = 8;
    std::vector<double> ap((size_t)ld * n * V, 0), tau((size_t)k * V, 0);
    auto call = [&](char lay, int m_, int n_, int k_, int ldap_, int V_, int nm_) {
        return dorgqr_compact(lay, m_, n_, k_, ap.data(), ldap_, tau.data(), V_, nm_);
    };

    // clang-format off
    const ApiCheck t[] = {
        {"valid",          call('C', m, n, k,   ld,  V, nm),   0},
        {"bad layout",     call('X', m, n, k,   ld,  V, nm),  -1},
        {"m<0",            call('C', -1, n, k,  ld,  V, nm),  -2},
        {"n>m",            call('C', m, m+1, k, ld,  V, nm),  -3},
        {"k>n",            call('C', m, n, n+1, ld,  V, nm),  -4},
        {"ldap<m",         call('C', m, n, k,   m-1, V, nm),  -6},
        {"ldap<n (row)",   call('R', m, n, k,   n-1, V, nm),  -6},
        {"bad V",          call('C', m, n, k,   ld,  3, nm),  -8},
        {"nm<0",           call('C', m, n, k,   ld,  V, -1),  -9},
        {"empty m=0",      call('C', 0, 0, 0,   1,   V, nm),   0},
        {"empty n=0",      call('C', m, 0, 0,   ld,  V, nm),   0},
        {"empty nm=0",     call('C', m, n, k,   ld,  V, 0),    0},
    };
    // clang-format on
    return report_api_checks(t);
}

/* ------------------------------- main -------------------------------- */

int main()
{
    int fails = 0;

    fails += test_validation();

    /* square thin Q (n = k = m), the geqrf output consumed directly */
    fails += run_case<double, 2>(4, 43, 43, 43);
    fails += run_case<double, 4>(8, 43, 43, 43);
    fails += run_case<double, 8>(16, 43, 43, 43);
    fails += run_case<double, 8>(11, 43, 43, 43); /* padded partial group */
    fails += run_case<double, 4>(40, 43, 43, 43); /* 10 groups: OpenMP path */
    fails += run_case<float, 4>(8, 43, 43, 43);
    fails += run_case<float, 8>(16, 43, 43, 43);
    fails += run_case<float, 16>(32, 43, 43, 43);

    /* tall thin Q (n = k < m) */
    fails += run_case<double, 4>(8, 43, 20, 20);
    fails += run_case<float, 8>(16, 43, 20, 20);

    /* k < n: extra unit-seeded columns */
    fails += run_case<double, 4>(8, 43, 30, 12);
    fails += run_case<double, 8>(11, 50, 24, 10); /* padded partial group too */
    fails += run_case<float, 4>(8, 43, 30, 12);

    /* k = 0: Q = leading columns of the identity */
    fails += run_case<double, 4>(8, 10, 6, 0);

    /* smallest size */
    fails += run_case<double, 2>(3, 1, 1, 1);

    /* row-major */
    fails += run_case<double, 4>(8, 43, 43, 43, true);
    fails += run_case<double, 8>(11, 43, 20, 20, true); /* padded partial group */
    fails += run_case<float, 8>(16, 30, 30, 30, true);
    fails += run_case<float, 4>(8, 40, 24, 9, true); /* k < n */

    return finish(fails);
}
