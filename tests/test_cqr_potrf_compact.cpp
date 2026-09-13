// test_cqr_potrf_compact.cpp
//
// Self-contained validation of the templated compact Cholesky factorization
// (dpotrf_compact / spotrf_compact), its solve companion (dpotrs_compact /
// spotrs_compact) and the fused factor-and-solve (dposv_compact /
// sposv_compact), with no BLAS dependency. The reference is the unblocked
// Cholesky (LAPACK ?potf2) implemented in scalar form -- the same algorithm
// the vectorized kernel executes V lanes at a time, so a correct kernel
// matches it to working precision.
//
// Checks per (T, V, uplo, layout):
//   1. named-triangle factor  ==  scalar potf2 factor   (elementwise, ~eps*scale)
//   2. reconstruction  L L^T == A  (lower) / U^T U == A  (upper)
//   3. the strictly-opposite triangle of the compact buffer is bit-for-bit
//      unchanged from the input (the routine must not reference or write it)
//   4. end-to-end solve: factor + ?potrs_compact recovers a known X, and
//      ?posv_compact reproduces that factor and X bit-for-bit
//   5. nrhs = 0: ?posv_compact still factors (LAPACK ?posv), bit-identical to
//      ?potrf_compact, with a null bp
// plus LAPACK-style argument validation of the three C APIs.
//
// Assisted-by: Claude:claude-opus-4.8 Claude

#include <cstdio>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#include "test_compact_util.hpp" // compact<T>, frand, gen_spd, MatrixBatch, pack/unpack

using namespace cqr::test;

// ----------------------- reference kernel (scalar) ------------------
// Unblocked right-looking potf2 on a dense column-major n x n matrix, in place.
// Lower: A = L L^T, factor in the lower triangle. Upper: A = U^T U, factor in
// the upper triangle. Only the named triangle is read or written. No SPD check
// (mirrors the routine under test): a bad pivot yields NaN/Inf.

template <class T> static void ref_potf2(char uplo, MatrixView<T> A)
{
    assert(A.rows == A.cols);
    const bool upper = (uplo == 'U' || uplo == 'u');
    const int n = A.rows;
    if (!upper) {
        for (int j = 0; j < n; ++j) {
            T d = std::sqrt(A(j, j));
            A(j, j) = d;
            T invd = T(1) / d;
            for (int i = j + 1; i < n; ++i)
                A(i, j) *= invd;               // scale pivot column
            for (int jj = j + 1; jj < n; ++jj) // rank-1 trailing update, lower
                for (int i = jj; i < n; ++i)
                    A(i, jj) -= A(i, j) * A(jj, j);
        }
    }
    else {
        for (int j = 0; j < n; ++j) {
            T d = std::sqrt(A(j, j));
            A(j, j) = d;
            T invd = T(1) / d;
            for (int c = j + 1; c < n; ++c)
                A(j, c) *= invd;            // scale pivot row
            for (int c = j + 1; c < n; ++c) // rank-1 trailing update, upper
                for (int r = j + 1; r <= c; ++r)
                    A(r, c) -= A(j, r) * A(j, c);
        }
    }
}

// --------------------------- one test case --------------------------

template <class T, int V> static int run_case(int nm, int n, char uplo, char layout)
{
    const T eps = std::numeric_limits<T>::epsilon();
    const bool rowmajor = (layout == 'R' || layout == 'r');
    const bool upper = (uplo == 'U' || uplo == 'u');

    // random SPD batch + scalar reference factor for this uplo
    MatrixBatch<T> A(nm, n, n), Aref(nm, n, n);
    for (int idx = 0; idx < nm; ++idx) {
        gen_spd(A.view(idx));
        std::copy(A[idx], A[idx] + (size_t)n * n, Aref[idx]);
        ref_potf2(uplo, Aref.view(idx));
    }

    // pack the full symmetric A, factor with the routine under test, unpack
    std::vector<T> ap = pack_compact(A, n, V, rowmajor);
    int info = compact<T>::potrf(layout, uplo, n, ap.data(), n, V, nm);
    MatrixBatch<T> Aout(nm, n, n);
    unpack_compact(Aout, ap.data(), n, V, rowmajor);

    double e_fac = 0, e_rec = 0, e_untouched = 0;
    std::vector<T> Recs((size_t)n * n);
    const auto Rec = mat_view(Recs.data(), n, n);
    for (int idx = 0; idx < nm; ++idx) {
        // check 1: named-triangle factor vs scalar reference (elementwise)
        // check 3: strictly-opposite triangle unchanged from the input A
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const bool named = upper ? (i <= j) : (i >= j);
                if (named)
                    e_fac = std::max(e_fac,
                                     (double)std::abs(Aout(idx, i, j) - Aref(idx, i, j)));
                else
                    e_untouched = std::max(
                        e_untouched, (double)std::abs(Aout(idx, i, j) - A(idx, i, j)));
            }

        // check 2: reconstruction of A from the named triangle's factor
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0;
                if (!upper) // A = L L^T : sum_l L(i,l) L(j,l), l <= min(i,j)
                    for (int l = 0; l <= std::min(i, j); ++l)
                        s += (double)Aout(idx, i, l) * (double)Aout(idx, j, l);
                else // A = U^T U : sum_l U(l,i) U(l,j), l <= min(i,j)
                    for (int l = 0; l <= std::min(i, j); ++l)
                        s += (double)Aout(idx, l, i) * (double)Aout(idx, l, j);
                Rec(i, j) = (T)s;
            }
        e_rec = std::max(e_rec, max_abs_diff(Recs.data(), A[idx], (size_t)n * n));
    }

    const double scale = std::max(1, n);
    const double tol_fac = 20.0 * eps * scale;         // same op sequence, unique factor
    const double tol_rec = 40.0 * eps * scale * scale; // O(n) accumulation in recon
    bool ok_f = e_fac <= tol_fac;
    bool ok_r = e_rec <= tol_rec;
    bool ok_u = e_untouched == 0.0; // must be bit-for-bit unchanged
    bool ok_i = (info == 0);

    std::printf("T=%-6s V=%-2d uplo=%c lay=%c nm=%-2d n=%-3d | fac:%.1e %s rec:%.1e %s "
                "untouched:%s | info=%d %s\n",
                compact<T>::name, V, uplo, layout, nm, n, e_fac, ok_f ? "OK" : "FAIL",
                e_rec, ok_r ? "OK" : "FAIL", ok_u ? "OK" : "FAIL", info,
                ok_i ? "OK" : "FAIL");
    return (!ok_f) + (!ok_r) + (!ok_u) + (!ok_i);
}

// --------------------- end-to-end solve A X = B ---------------------
// Two-step (potrf, then potrs) against a known X, and the fused posv against
// the two-step result: the fused driver runs the same group kernels in the
// same order on the same data, so its factor and X must match bit-for-bit.

template <class T, int V>
static int run_solve(int nm, int n, int nrhs, char uplo, char layout)
{
    const T eps = std::numeric_limits<T>::epsilon();
    const bool rowmajor = (layout == 'R' || layout == 'r');
    const int ldb = rowmajor ? nrhs : n;

    // known X, B = A X densely
    MatrixBatch<T> A(nm, n, n), B(nm, n, nrhs);
    const std::vector<T> Xs = known_solution<T>(n, nrhs);
    const auto X = mat_view(Xs.data(), n, nrhs);
    for (int idx = 0; idx < nm; ++idx) {
        gen_spd(A.view(idx));
        matmul(A.view(idx), X, B.view(idx));
    }

    std::vector<T> ap = pack_compact(A, n, V, rowmajor);
    std::vector<T> bp = pack_compact(B, ldb, V, rowmajor);
    std::vector<T> ap2 = ap, bp2 = bp; // the fused call's copies

    int info_f = compact<T>::potrf(layout, uplo, n, ap.data(), n, V, nm);
    int info_s =
        compact<T>::potrs(layout, uplo, n, nrhs, ap.data(), n, bp.data(), ldb, V, nm);
    int info_v =
        compact<T>::posv(layout, uplo, n, nrhs, ap2.data(), n, bp2.data(), ldb, V, nm);

    MatrixBatch<T> Xhat(nm, n, nrhs);
    unpack_compact(Xhat, bp.data(), ldb, V, rowmajor);

    const auto [e_fwd, e_res] = solve_errors(A, B, Xhat, X);
    // fused vs two-step, on the raw compact buffers (padded lanes included)
    const bool fused_same = (ap2 == ap) && (bp2 == bp);

    // The tame SPD batches keep cond(A) modest, so residual and forward error
    // share one backward-stability-sized gate.
    const double rtol = 100.0 * std::max(1, n) * eps;
    bool ok = (e_fwd <= rtol) && (e_res <= rtol) && fused_same && (info_f == 0) &&
              (info_s == 0) && (info_v == 0);
    std::printf("T=%-6s V=%-2d uplo=%c lay=%c nm=%-2d n=%-3d nrhs=%d solve | fwd:%.1e "
                "res:%.1e (%.1e) posv==trf+trs:%s info=%d/%d/%d %s\n",
                compact<T>::name, V, uplo, layout, nm, n, nrhs, e_fwd, e_res, rtol,
                fused_same ? "yes" : "NO", info_f, info_s, info_v, ok ? "OK" : "FAIL");
    return !ok;
}

// ------------- nrhs = 0: the fused driver still factors --------------
// LAPACK ?posv calls ?potrf unconditionally -- the nrhs = 0 quick return is
// ?potrs's -- so the fused driver must factor ap even with no right-hand
// sides, bit-identically to ?potrf_compact, without referencing bp (null
// here).

template <class T, int V> static int run_nrhs0(int nm, int n, char uplo, char layout)
{
    const bool rowmajor = (layout == 'R' || layout == 'r');
    MatrixBatch<T> A(nm, n, n);
    for (int idx = 0; idx < nm; ++idx)
        gen_spd(A.view(idx));
    std::vector<T> ap = pack_compact(A, n, V, rowmajor);
    std::vector<T> ap2 = ap;

    int info_f = compact<T>::potrf(layout, uplo, n, ap.data(), n, V, nm);
    int info_v = compact<T>::posv(layout, uplo, n, 0, ap2.data(), n, nullptr, n, V, nm);

    const bool same = (ap == ap2);
    bool ok = (info_f == 0) && (info_v == 0) && same;
    std::printf("T=%-6s V=%-2d uplo=%c lay=%c nm=%-2d n=%-3d nrhs=0 | posv==potrf:%s "
                "info=%d/%d %s\n",
                compact<T>::name, V, uplo, layout, nm, n, same ? "yes" : "NO", info_f,
                info_v, ok ? "OK" : "FAIL");
    return !ok;
}

// ------------- non-SPD lane isolation (design section 6.2) ----------
// A single non-SPD matrix shares a pack with SPD siblings. The routine takes no
// safeguarded path and reports no error (info stays 0): the bad lane simply
// hits a non-positive pivot, so sqrt/reciprocal poison *its own* factor with
// NaN/Inf -- while every sibling lane, computed with the same unmasked SIMD
// instructions, must stay finite and bit-exact vs the scalar reference. This is
// the "graceful garbage in, garbage out" contract and, crucially, gates that a
// poisoned lane never contaminates its neighbors.

template <class T, int V> static int run_nonspd(int n, char uplo, char layout)
{
    const bool rowmajor = (layout == 'R' || layout == 'r');
    const bool upper = (uplo == 'U' || uplo == 'u');
    const int nm = V;      // one full pack, so all V lanes are exercised together
    const int badlane = 1; // this lane is non-SPD; the rest are SPD

    MatrixBatch<T> A(nm, n, n), Aref(nm, n, n);
    for (int idx = 0; idx < nm; ++idx) {
        if (idx == badlane) {
            // symmetric but indefinite: identity with a negative leading pivot,
            // so the very first sqrt(A(0,0)) = sqrt(-1) = NaN.
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    A(idx, i, j) = (i == j) ? T(1) : T(0);
            A(idx, 0, 0) = T(-1);
        }
        else {
            gen_spd(A.view(idx));
            std::copy(A[idx], A[idx] + (size_t)n * n, Aref[idx]);
            ref_potf2(uplo, Aref.view(idx));
        }
    }

    std::vector<T> ap = pack_compact(A, n, V, rowmajor);
    int info = compact<T>::potrf(layout, uplo, n, ap.data(), n, V, nm);
    MatrixBatch<T> Aout(nm, n, n);
    unpack_compact(Aout, ap.data(), n, V, rowmajor);

    double e_spd = 0; // worst error over the SPD sibling lanes
    bool spd_finite = true;
    bool bad_poisoned = false; // the non-SPD lane must carry NaN/Inf
    for (int idx = 0; idx < nm; ++idx)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const bool named = upper ? (i <= j) : (i >= j);
                if (!named) continue;
                const T x = Aout(idx, i, j);
                if (idx == badlane) {
                    if (!std::isfinite((double)x)) bad_poisoned = true;
                }
                else {
                    if (!std::isfinite((double)x)) spd_finite = false;
                    e_spd = std::max(e_spd, (double)std::abs(x - Aref(idx, i, j)));
                }
            }

    const double tol = 20.0 * std::numeric_limits<T>::epsilon() * std::max(1, n);
    bool ok_spd = spd_finite && (e_spd <= tol); // siblings uncontaminated & correct
    bool ok_bad = bad_poisoned;                 // poison confined but present
    bool ok_info = (info == 0);                 // GIGO: no early-exit, no error

    std::printf("T=%-6s V=%-2d uplo=%c lay=%c n=%-3d non-SPD lane | siblings:%.1e %s "
                "poison:%s info=%d %s\n",
                compact<T>::name, V, uplo, layout, n, e_spd, ok_spd ? "OK" : "FAIL",
                ok_bad ? "OK" : "FAIL", info, ok_info ? "OK" : "FAIL");
    return (!ok_spd) + (!ok_bad) + (!ok_info);
}

// --------------------- C API argument validation --------------------

// The factor signature is shared with ?sytrfnp and the solve signature with
// ?sytrsnp/?sysvnp -- one validator each in cqr_compact.cpp -- so the tables
// live in test_compact_util.hpp and are run here against this family's three
// entry points.
static int test_validation()
{
    std::vector<ApiCheck> t;
    append_factor_api_checks(t, "trf", dpotrf_compact);
    append_solve_api_checks(t, "trs", dpotrs_compact);
    append_solve_api_checks(t, "sv", dposv_compact);
    return report_api_checks(t.data(), t.size());
}

// ------------------------------- main --------------------------------

int main()
{
    int fails = 0;
    fails += test_validation();

    // Full feature matrix: both uplo x both layouts, several (V, n, nm),
    // including padded partial final groups (nm not a multiple of V).
    const char uplos[] = {'L', 'U'};
    const char lays[] = {'C', 'R'};
    for (char u : uplos)
        for (char l : lays) {
            fails += run_case<double, 2>(4, 16, u, l);
            fails += run_case<double, 4>(8, 30, u, l);
            fails += run_case<double, 8>(16, 40, u, l);
            fails += run_case<double, 8>(11, 43, u, l); // padded partial group
            fails += run_case<double, 4>(3, 3, u, l);   // smallest, padded
            fails += run_case<float, 8>(16, 30, u, l);
            fails += run_case<float, 16>(17, 24, u, l); // padded partial group
        }

    // End-to-end solve A X = B on SPD batches, closing the pipeline (factor +
    // potrs, and the fused posv), over uplo/layout/precision, padded groups,
    // and RHS counts that exercise trsm's 4/2/1 column blocks.
    for (char u : uplos)
        for (char l : lays) {
            fails += run_solve<double, 4>(8, 30, 5, u, l);
            fails += run_solve<double, 8>(11, 43, 4, u, l); // padded partial group
            fails += run_solve<double, 2>(6, 17, 1, u, l);  // single RHS
            fails += run_solve<float, 8>(16, 24, 3, u, l);
        }

    // nrhs = 0 must factor anyway (LAPACK ?posv), bit-identical to potrf,
    // with a null bp.
    fails += run_nrhs0<double, 4>(6, 20, 'L', 'C');
    fails += run_nrhs0<float, 8>(9, 16, 'U', 'R');

    // Non-SPD lane isolation (design 6.2): a poisoned lane must not contaminate
    // its SPD siblings, over both uplo, both layouts, and both precisions.
    for (char u : uplos)
        for (char l : lays) {
            fails += run_nonspd<double, 4>(20, u, l);
            fails += run_nonspd<float, 8>(16, u, l);
        }

    // 10 groups: takes the OpenMP group loop when the team has <= 10 threads.
    fails += run_case<double, 4>(40, 20, 'L', 'C');
    fails += run_solve<double, 4>(40, 20, 4, 'L', 'C');

    return finish(fails);
}
