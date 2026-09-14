// test_potrf_compact.cpp
//
// Validation of the templated compact Cholesky factorization (dpotrf_compact
// / spotrf_compact), its solve companion (dpotrs_compact / spotrs_compact)
// and the fused factor-and-solve (dposv_compact / sposv_compact) against
// LAPACKE (test_lapack_util.hpp): the reference factor is LAPACKE_?potrf
// (recursive / blocked, so its rounding differs from the kernel's
// right-looking sweep, but the SPD factor is unique).
//
// Checks per (T, V, uplo, layout), each a test ratio against THRESH
// (test_compact_util.hpp):
//   1. named-triangle factor  ==  LAPACKE_?potrf factor, relative to the
//      reference factor's L1 norm (design 7.1: the unique factor agrees to a
//      few n eps between backward-stable implementations)
//   2. reconstruction  L L^T == A  (lower) / U^T U == A  (upper), by ?trmm
//      (dpot01's residual)
//   3. the strictly-opposite triangle of the compact buffer is bit-for-bit
//      unchanged from the input (the routine must not reference or write it)
//   4. end-to-end solve: factor + ?potrs_compact recovers a known X, and
//      ?posv_compact reproduces that factor and X bit-for-bit
//   5. nrhs = 0: ?posv_compact still factors (LAPACK ?posv), bit-identical to
//      ?potrf_compact, with a 1-element dummy bp (never referenced)
// over the suite's orders and the small-dimension sweep (small_dims: every
// order from 0 to 5 through the factorization checks, n x nrhs through the
// solve pipeline, the empty operand included), plus LAPACK-style argument
// validation of the three C APIs.
//
// Assisted-by: Claude:claude-opus-4-8 Claude:claude-fable-5

#include <cstdio>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#include "test_lapack_util.hpp" // compact<T>, ref_potrf, tri_apply, gen_spd, pack/unpack

using namespace cbk::test;

// --------------------------- one test case --------------------------

template <class T, int V>
static int run_case(int nm, int n, char uplo, char layout, double cond = 0.0)
{
    const bool rowmajor = (layout == 'R' || layout == 'r');
    const bool upper = (uplo == 'U' || uplo == 'u');

    // random SPD batch (cond > 0 squeezes its spectrum) + LAPACKE's factor
    // for this uplo
    MatrixBatch<T> A(nm, n, n), Aref(nm, n, n);
    for (int idx = 0; idx < nm; ++idx) {
        gen_spd(A.view(idx), cond);
        std::copy(A[idx], A[idx] + (size_t)n * n, Aref[idx]);
        ref_potrf(uplo, Aref.view(idx));
    }

    // pack the full symmetric A, factor with the routine under test, unpack
    const int lda = std::max(1, n);
    std::vector<T> ap = pack_compact(A, lda, V, rowmajor);
    int info = compact<T>::potrf(layout, uplo, n, ap.data(), lda, V, nm);
    MatrixBatch<T> Aout(nm, n, n);
    unpack_compact(Aout, ap.data(), lda, V, rowmajor);

    double r_fac = 0, r_rec = 0, e_untouched = 0;
    std::vector<T> Fs((size_t)n * n), Recs((size_t)n * n);
    const auto F = mat_view(Fs.data(), n, n), Rec = mat_view(Recs.data(), n, n);
    for (int idx = 0; idx < nm; ++idx) {
        const auto Fac = Aout.view(idx), Ref = Aref.view(idx), Ain = A.view(idx);
        // check 1: named-triangle factor vs LAPACKE_?potrf, relative to the
        // reference factor's L1 norm (the unique SPD factor: a sharp signal)
        // check 3: strictly-opposite triangle unchanged from the input A
        double el = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const bool named = upper ? (i <= j) : (i >= j);
                if (named)
                    el = std::max(el, (double)std::abs(Fac(i, j) - Ref(i, j)));
                else
                    e_untouched =
                        std::max(e_untouched, (double)std::abs(Fac(i, j) - Ain(i, j)));
            }
        r_fac = std::max(r_fac, test_ratio<T>(el, n, norm1(Ref)));

        // check 2: reconstruction of A from the named triangle's factor, by
        // ?trmm: F is the factor with the other triangle zeroed, so
        // L L^T = F L^T (side 'R') and U^T U = U^T F (side 'L') -- dpot01's
        // ||L L^T - A|| / (n ||A|| eps)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                F(i, j) = (upper ? (i <= j) : (i >= j)) ? Fac(i, j) : T(0);
        if (!upper)
            tri_apply('R', 'L', 'T', 'N', Fac, F, Rec);
        else
            tri_apply('L', 'U', 'T', 'N', Fac, F, Rec);
        r_rec = std::max(r_rec,
                         test_ratio<T>(max_abs_diff(Recs.data(), A[idx], (size_t)n * n),
                                       n, norm1(Ain)));
    }

    const bool ok_f = passes(r_fac), ok_r = passes(r_rec);
    const bool ok_u = e_untouched == 0.0; // must be bit-for-bit unchanged
    const bool ok_i = (info == 0);

    std::printf("T=%-6s V=%-2d uplo=%c lay=%c nm=%-2d n=%-3d cond=%.0f | fac:%-5.2g %s "
                "rec:%-5.2g %s untouched:%s | info=%d %s\n",
                compact<T>::name, V, uplo, layout, nm, n, cond, r_fac, verdict(r_fac),
                r_rec, verdict(r_rec), ok_u ? "OK" : "FAIL", info, ok_i ? "OK" : "FAIL");
    return (!ok_f) + (!ok_r) + (!ok_u) + (!ok_i);
}

// --------------------- end-to-end solve A X = B ---------------------
// Two-step (potrf, then potrs) against a known X, and the fused posv against
// the two-step result: the fused driver runs the same group kernels in the
// same order on the same data, so its factor and X must match bit-for-bit.

template <class T, int V>
static int run_solve(int nm, int n, int nrhs, char uplo, char layout)
{
    const bool rowmajor = (layout == 'R' || layout == 'r');
    const int lda = std::max(1, n), ldb = std::max(1, rowmajor ? nrhs : n);

    // known X, B = A X densely
    MatrixBatch<T> A(nm, n, n), B(nm, n, nrhs);
    const std::vector<T> Xs = known_solution<T>(n, nrhs);
    const auto X = mat_view(Xs.data(), n, nrhs);
    for (int idx = 0; idx < nm; ++idx) {
        gen_spd(A.view(idx));
        matmul(A.view(idx), X, B.view(idx));
    }

    std::vector<T> ap = pack_compact(A, lda, V, rowmajor);
    std::vector<T> bp = pack_compact(B, ldb, V, rowmajor);
    std::vector<T> ap2 = ap, bp2 = bp; // the fused call's copies

    int info_f = compact<T>::potrf(layout, uplo, n, ap.data(), lda, V, nm);
    int info_s =
        compact<T>::potrs(layout, uplo, n, nrhs, ap.data(), lda, bp.data(), ldb, V, nm);
    int info_v =
        compact<T>::posv(layout, uplo, n, nrhs, ap2.data(), lda, bp2.data(), ldb, V, nm);

    MatrixBatch<T> Xhat(nm, n, nrhs);
    unpack_compact(Xhat, bp.data(), ldb, V, rowmajor);

    // dget02's residual and dget04's forward error
    const auto [r_res, r_fwd] = solve_ratios(A, B, Xhat, X);
    // fused vs two-step, on the raw compact buffers (padded lanes included)
    const bool fused_same = (ap2 == ap) && (bp2 == bp);

    const bool ok = passes(r_res) && passes(r_fwd) && fused_same && (info_f == 0) &&
                    (info_s == 0) && (info_v == 0);
    std::printf(
        "T=%-6s V=%-2d uplo=%c lay=%c nm=%-2d n=%-3d nrhs=%d solve | res:%-5.2g %s "
        "fwd:%-5.2g %s posv==trf+trs:%s info=%d/%d/%d %s\n",
        compact<T>::name, V, uplo, layout, nm, n, nrhs, r_res, verdict(r_res), r_fwd,
        verdict(r_fwd), fused_same ? "yes" : "NO", info_f, info_s, info_v,
        ok ? "OK" : "FAIL");
    return !ok;
}

// ------------- nrhs = 0: the fused driver still factors --------------
// LAPACK ?posv calls ?potrf unconditionally -- the nrhs = 0 quick return is
// ?potrs's -- so the fused driver must factor ap even with no right-hand
// sides, bit-identically to ?potrf_compact, without referencing bp (a
// 1-element dummy here: Fortran semantics want the argument present).

template <class T, int V> static int run_nrhs0(int nm, int n, char uplo, char layout)
{
    const bool rowmajor = (layout == 'R' || layout == 'r');
    MatrixBatch<T> A(nm, n, n);
    for (int idx = 0; idx < nm; ++idx)
        gen_spd(A.view(idx));
    std::vector<T> ap = pack_compact(A, n, V, rowmajor);
    std::vector<T> ap2 = ap;

    int info_f = compact<T>::potrf(layout, uplo, n, ap.data(), n, V, nm);
    T b_dummy = 0; // never referenced at nrhs = 0, present per Fortran semantics
    int info_v = compact<T>::posv(layout, uplo, n, 0, ap2.data(), n, &b_dummy, n, V, nm);

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
            ref_potrf(uplo, Aref.view(idx));
        }
    }

    std::vector<T> ap = pack_compact(A, n, V, rowmajor);
    int info = compact<T>::potrf(layout, uplo, n, ap.data(), n, V, nm);
    MatrixBatch<T> Aout(nm, n, n);
    unpack_compact(Aout, ap.data(), n, V, rowmajor);

    double r_spd = 0; // worst ratio over the SPD sibling lanes
    bool spd_finite = true;
    bool bad_poisoned = false; // the non-SPD lane must carry NaN/Inf
    for (int idx = 0; idx < nm; ++idx) {
        double el = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const bool named = upper ? (i <= j) : (i >= j);
                if (!named) continue;
                const T x = Aout(idx, i, j);
                if (idx == badlane) {
                    bad_poisoned = bad_poisoned || !std::isfinite((double)x);
                    continue;
                }
                spd_finite = spd_finite && std::isfinite((double)x);
                el = std::max(el, (double)std::abs(x - Aref(idx, i, j)));
            }
        if (idx != badlane) // vs LAPACKE_?potrf, relative to its factor's norm
            r_spd = std::max(r_spd, test_ratio<T>(el, n, norm1(Aref.view(idx))));
    }

    const bool ok_spd = spd_finite && passes(r_spd); // siblings uncontaminated & correct
    const bool ok_bad = bad_poisoned;                // poison confined but present
    const bool ok_info = (info == 0);                // GIGO: no early-exit, no error

    std::printf("T=%-6s V=%-2d uplo=%c lay=%c n=%-3d non-SPD lane | siblings:%-5.2g %s "
                "poison:%s info=%d %s\n",
                compact<T>::name, V, uplo, layout, n, r_spd, ok_spd ? "OK" : "FAIL",
                ok_bad ? "OK" : "FAIL", info, ok_info ? "OK" : "FAIL");
    return (!ok_spd) + (!ok_bad) + (!ok_info);
}

// --------------------- C API argument validation --------------------

// The factor signature is shared with ?sytrfnp and the solve signature with
// ?sytrsnp/?sysvnp -- one validator each in cbk.cpp -- so the tables
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
            fails += run_case<float, 16>(17, 24, u, l);     // padded partial group
            fails += run_case<double, 4>(8, 40, u, l, 2.0); // dynamic range (cond knob)
        }
    fails += run_case<double, 8>(8, 128, 'L', 'C'); // LAPACK's blocked potrf
    fails += run_case<float, 8>(8, 128, 'U', 'R');

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

    // The small-dimension sweep (small_dims): every order over uplo x layout,
    // then n x nrhs through the solve pipeline, a padded group each.
    for (int n : small_dims)
        for (char u : uplos)
            for (char l : lays)
                fails += run_case<double, 4>(5, n, u, l);
    for (int n : small_dims)
        for (int nrhs : small_dims) {
            fails += run_solve<double, 4>(5, n, nrhs, 'L', 'C');
            fails += run_solve<float, 8>(9, n, nrhs, 'U', 'R');
        }

    // nrhs = 0 must factor anyway (LAPACK ?posv), bit-identical to potrf,
    // bp a never-referenced dummy.
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
