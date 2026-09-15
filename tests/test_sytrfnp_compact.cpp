// test_sytrfnp_compact.cpp
//
// Validation of the templated compact unpivoted LDL^T factorization
// (dsytrfnp_compact / ssytrfnp_compact), its solve companion (dsytrsnp_compact
// / ssytrsnp_compact) and the fused factor-and-solve (dsysvnp_compact /
// ssysvnp_compact). LAPACK has no unpivoted LDL^T (?sytrf pivots, so its
// factors differ elementwise), so the reference is the one scalar routine the
// suites still hand-roll, ref_sytf2np below: the unblocked unpivoted
// factorization -- the same algorithm the vectorized kernel executes V lanes
// at a time, so a correct kernel matches it to working precision. The
// reference itself is validated first, against the library: its (L, D) must
// reconstruct A through cblas_?trmm / ?gemm (test_reference).
//
// Checks per (T, V, uplo, layout), on symmetric *indefinite* batches, each a
// test ratio against THRESH (test_compact_util.hpp):
//   1. named-triangle factor  ==  scalar reference factor  (elementwise)
//   2. reconstruction  L D L^T == A  (lower) / U^T D U == A  (upper)
//   3. the strictly-opposite triangle of the compact buffer is bit-for-bit
//      unchanged from the input (the routine must not reference or write it)
//   4. end-to-end solve: factor + ?sytrsnp_compact recovers a known X, and
//      ?sysvnp_compact reproduces that factor and X bit-for-bit
//   5. nrhs = 0: ?sysvnp_compact still factors (LAPACK ?sysv), bit-identical
//      to ?sytrfnp_compact, with a 1-element dummy bp (never referenced)
//   6. ?sytrsnp_compact == the unit-diagonal ?trsm_compact sweeps around
//      D^-1 it is built from (dget04 of the one solution against the other)
// over the suite's orders and the small-dimension sweep (small_dims: every
// order from 0 to 5 through the factorization checks, n x nrhs through the
// solve pipeline, the empty operand included), plus:
//   - a zero on the *input* diagonal with nonsingular leading minors factors
//     fine (the pivots are the updated Schur-complement entries),
//   - a zero-*pivot* lane poisons itself with Inf/NaN without contaminating
//     its siblings (design section 6.2),
//   - LAPACK-style argument validation of the three C APIs.
//
// Assisted-by: Claude:claude-fable-5

#include <cstdio>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#include "test_lapack_util.hpp" // compact<T>, matmul, tri_apply, gen_sym_ldlt, pack/unpack

using namespace cbk::test;

// ----------------------- reference kernel (scalar) ------------------
// Unblocked right-looking unpivoted LDL^T on a dense column-major n x n matrix,
// in place. Lower: A = L D L^T; upper: A = U^T D U (the transpose dual, NOT
// LAPACK sytrf's U D U^T). D lands on the diagonal, the unit factor strictly
// off it. Only the named triangle is read or written. No singularity check
// (mirrors the routine under test): a zero pivot yields Inf/NaN.

template <class T> static void ref_sytf2np(char uplo, MatrixView<T> As)
{
    assert(As.rows == As.cols);
    const bool upper = (uplo == 'U' || uplo == 'u');
    const int n = As.rows;
    // One sweep serves both triangles, as in the kernel: factor the lower
    // triangle of A for uplo lower and of A^T for upper (A is symmetric, so
    // U^T D U of A is L D L^T of A^T). Same operand pairing either way, so the
    // reference and the kernel stay bit-comparable.
    const auto A = upper ? As.transposed() : As;
    for (int j = 0; j < n; ++j) {
        const T d = A(j, j);
        const T invd = T(1) / d;
        for (int i = j + 1; i < n; ++i)
            A(i, j) *= invd;                 // scale pivot column -> L(:,j)
        for (int jj = j + 1; jj < n; ++jj) { // rank-1 trailing update, lower
            const T w = A(jj, j) * d;
            for (int i = jj; i < n; ++i)
                A(i, jj) -= A(i, j) * w;
        }
    }
}

// ------------- the reference itself, vs the library ------------------
// ref_sytf2np is the only hand-rolled reference left in the suites, so it is
// checked against LAPACKE/CBLAS before anything is checked against it: its
// (L, D) must reconstruct A. W = L D (the columns of the unit-lower L scaled
// by D), then W := W L^T by ?trmm (side 'R', unit diagonal); the upper
// factor U^T D U is the same product read through the transposed view, as in
// ref_sytf2np itself, and A is symmetric. Gated on the test ratio
// ||L D L^T - A|| / (n ||A|| eps), the design's 7.1 residual.

template <class T> static int test_reference(int n, char uplo)
{
    const bool upper = (uplo == 'U' || uplo == 'u');
    std::vector<T> As((size_t)n * n), Fs((size_t)n * n), Ws((size_t)n * n),
        Rs((size_t)n * n);
    const auto A = mat_view(As.data(), n, n), F = mat_view(Fs.data(), n, n),
               W = mat_view(Ws.data(), n, n), R = mat_view(Rs.data(), n, n);
    gen_sym_ldlt(A);
    copy_matrix(A, F);
    ref_sytf2np(uplo, F);
    const auto L = upper ? F.transposed() : F; // the unit-lower factor, D on its diagonal
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            W(i, j) = (i == j) ? L(j, j) : (i > j) ? L(i, j) * L(j, j) : T(0); // L D
    tri_apply('R', 'L', 'T', 'U', L, W, R); // (L D) L^T
    const double r = test_ratio<T>(diff_norm1(R, A), n, norm1(A));
    const bool ok = passes(r);
    std::printf("T=%-6s reference ref_sytf2np uplo=%c n=%-3d | ||LDL^T-A||/(n ||A|| eps) "
                "%-5.2g %s\n",
                compact<T>::name, uplo, n, r, verdict(r));
    return !ok;
}

// --------------------------- one test case --------------------------

template <class T, int V> static int run_case(int nm, int n, char uplo, char layout)
{
    const bool rowmajor = (layout == 'R' || layout == 'r');
    const bool upper = (uplo == 'U' || uplo == 'u');

    // random symmetric-indefinite batch + scalar reference factor for this uplo
    MatrixBatch<T> A(nm, n, n), Aref(nm, n, n);
    for (int idx = 0; idx < nm; ++idx) {
        gen_sym_ldlt(A.view(idx));
        std::copy(A[idx], A[idx] + (size_t)n * n, Aref[idx]);
        ref_sytf2np(uplo, Aref.view(idx));
    }

    // pack the full symmetric A, factor with the routine under test, unpack
    const int lda = std::max(1, n);
    std::vector<T> ap = pack_compact(A, lda, V, rowmajor);
    int info = compact<T>::sytrfnp(layout, uplo, n, ap.data(), lda, V, nm);
    MatrixBatch<T> Aout(nm, n, n);
    unpack_compact(Aout, ap.data(), lda, V, rowmajor);

    double r_fac = 0, r_rec = 0, e_untouched = 0;
    for (int idx = 0; idx < nm; ++idx) {
        const auto Ain = A.view(idx), Fac = Aout.view(idx), Ref = Aref.view(idx);
        const double na = norm1(Ain);
        // check 1: named-triangle factor vs scalar reference (the same
        // operation sequence; the 1-norm of the difference over the triangle,
        // relative to ||A||)
        // check 2: reconstruction of A from the named triangle's (D, L|U):
        // dsyt01's ||L D L^T - A||_1 / (n ||A||_1 eps)
        // check 3: strictly-opposite triangle unchanged from the input A
        double e_fac = 0, e_rec = 0;
        for (int j = 0; j < n; ++j) {
            double cs_fac = 0, cs_rec = 0;
            for (int i = 0; i < n; ++i) {
                const bool named = upper ? (i <= j) : (i >= j);
                if (named)
                    cs_fac += std::abs((double)Fac(i, j) - Ref(i, j));
                else
                    e_untouched =
                        std::max(e_untouched, (double)std::abs(Fac(i, j) - Ain(i, j)));
                cs_rec +=
                    std::abs(ldlt_reconstruct(Fac, i, j, upper) - (double)Ain(i, j));
            }
            e_fac = std::max(e_fac, cs_fac);
            e_rec = std::max(e_rec, cs_rec);
        }
        r_fac = std::max(r_fac, test_ratio<T>(e_fac, n, na));
        r_rec = std::max(r_rec, test_ratio<T>(e_rec, n, na));
    }

    const bool ok_f = passes(r_fac), ok_r = passes(r_rec);
    const bool ok_u = e_untouched == 0.0; // must be bit-for-bit unchanged
    const bool ok_i = (info == 0);

    std::printf(
        "T=%-6s V=%-2d uplo=%c lay=%c nm=%-2d n=%-3d | fac:%-5.2g %s rec:%-5.2g %s "
        "untouched:%s | info=%d %s\n",
        compact<T>::name, V, uplo, layout, nm, n, r_fac, verdict(r_fac), r_rec,
        verdict(r_rec), ok_u ? "OK" : "FAIL", info, ok_i ? "OK" : "FAIL");
    return (!ok_f) + (!ok_r) + (!ok_u) + (!ok_i);
}

// --------------------- end-to-end solve A X = B ---------------------
// Two-step (sytrfnp, then sytrsnp) against a known X, and the fused sysvnp
// against the two-step result: the fused driver runs the same group kernels in
// the same order on the same data, so its factor and X must match bit-for-bit.

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
        gen_sym_ldlt(A.view(idx));
        matmul(A.view(idx), X, B.view(idx));
    }

    std::vector<T> ap = pack_compact(A, lda, V, rowmajor);
    std::vector<T> bp = pack_compact(B, ldb, V, rowmajor);
    std::vector<T> ap2 = ap, bp2 = bp; // the fused call's copies

    int info_f = compact<T>::sytrfnp(layout, uplo, n, ap.data(), lda, V, nm);
    int info_s =
        compact<T>::sytrsnp(layout, uplo, n, nrhs, ap.data(), lda, bp.data(), ldb, V, nm);
    int info_v = compact<T>::sysvnp(layout, uplo, n, nrhs, ap2.data(), lda, bp2.data(),
                                    ldb, V, nm);

    MatrixBatch<T> Xhat(nm, n, nrhs);
    unpack_compact(Xhat, bp.data(), ldb, V, rowmajor);

    // dget02's residual and dget04's forward error (the latter discounted by
    // rcond(A), which the indefinite batch's conditioning enters through)
    const auto [r_res, r_fwd] = solve_ratios(A, B, Xhat, X);
    // fused vs two-step, on the raw compact buffers (padded lanes included)
    const bool fused_same = (ap2 == ap) && (bp2 == bp);

    const bool ok = passes(r_res) && passes(r_fwd) && fused_same && (info_f == 0) &&
                    (info_s == 0) && (info_v == 0);
    std::printf(
        "T=%-6s V=%-2d uplo=%c lay=%c nm=%-2d n=%-3d nrhs=%d solve | res:%-5.2g %s "
        "fwd:%-5.2g %s sysv==trf+trs:%s info=%d/%d/%d %s\n",
        compact<T>::name, V, uplo, layout, nm, n, nrhs, r_res, verdict(r_res), r_fwd,
        verdict(r_fwd), fused_same ? "yes" : "NO", info_f, info_s, info_v,
        ok ? "OK" : "FAIL");
    return !ok;
}

// ------------- sytrsnp vs the trsm sweeps it is built from ------------
// ?sytrsnp_compact is two unit-diagonal ?trsm_compact sweeps around a
// division by D (lower: L z = B, z := D^-1 z, L^T X = z; upper: U^T z = B,
// z := D^-1 z, U X = z), so the public trsm entry point, with the diagonal
// scaling done densely in between, must solve the same system from the same
// factor. Gated as dget04's forward error of the one solution against the
// other (rcond(A)); the two are not expected bit-identical (the kernel's
// scaling is its own), so only the ratio is gated.

template <class T, int V>
static int run_compose(int nm, int n, int nrhs, char uplo, char layout)
{
    const bool rowmajor = (layout == 'R' || layout == 'r');
    const bool upper = (uplo == 'U' || uplo == 'u');
    const int lda = std::max(1, n), ldb = std::max(1, rowmajor ? nrhs : n);

    MatrixBatch<T> A(nm, n, n), B(nm, n, nrhs);
    const std::vector<T> Xs = known_solution<T>(n, nrhs);
    const auto X = mat_view(Xs.data(), n, nrhs);
    for (int idx = 0; idx < nm; ++idx) {
        gen_sym_ldlt(A.view(idx));
        matmul(A.view(idx), X, B.view(idx));
    }
    std::vector<T> ap = pack_compact(A, lda, V, rowmajor);
    std::vector<T> bp = pack_compact(B, ldb, V, rowmajor);
    std::vector<T> bp2 = bp;

    int bad = (compact<T>::sytrfnp(layout, uplo, n, ap.data(), lda, V, nm) != 0);
    bad += (compact<T>::sytrsnp(layout, uplo, n, nrhs, ap.data(), lda, bp.data(), ldb, V,
                                nm) != 0);
    // the same solve as two unit-diagonal trsm sweeps around D^-1
    const char t1 = upper ? 'T' : 'N', t2 = upper ? 'N' : 'T';
    bad += (compact<T>::trsm(layout, 'L', uplo, t1, 'U', n, nrhs, T(1), ap.data(), lda,
                             bp2.data(), ldb, V, nm) != 0);
    MatrixBatch<T> F(nm, n, n), Z(nm, n, nrhs);
    unpack_compact(F, ap.data(), lda, V, rowmajor);
    unpack_compact(Z, bp2.data(), ldb, V, rowmajor);
    for (int idx = 0; idx < nm; ++idx)
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i)
                Z(idx, i, j) /= F(idx, i, i);
    pack_compact(Z, bp2.data(), ldb, V, rowmajor);
    bad += (compact<T>::trsm(layout, 'L', uplo, t2, 'U', n, nrhs, T(1), ap.data(), lda,
                             bp2.data(), ldb, V, nm) != 0);

    MatrixBatch<T> X1(nm, n, nrhs), X2(nm, n, nrhs);
    unpack_compact(X1, bp.data(), ldb, V, rowmajor);
    unpack_compact(X2, bp2.data(), ldb, V, rowmajor);
    double r = 0;
    for (int idx = 0; idx < nm; ++idx)
        r = std::max(r, forward_ratio(X1.view(idx), X2.view(idx), rcond1(A.view(idx))));

    const bool ok = passes(r) && (bad == 0);
    std::printf("T=%-6s V=%-2d uplo=%c lay=%c nm=%-2d n=%-3d nrhs=%d sytrsnp vs "
                "trsm+D+trsm | %-5.2g %s %s\n",
                compact<T>::name, V, uplo, layout, nm, n, nrhs, r, verdict(r),
                ok ? "OK" : "FAIL");
    return !ok;
}

// ------------- nrhs = 0: the fused driver still factors --------------
// LAPACK ?sysv calls ?sytrf unconditionally -- the nrhs = 0 quick return is
// ?sytrs's -- so the fused driver must factor ap even with no right-hand
// sides, bit-identically to ?sytrfnp_compact, without referencing bp (a
// 1-element dummy here: Fortran semantics want the argument present).

template <class T, int V> static int run_nrhs0(int nm, int n, char uplo, char layout)
{
    const bool rowmajor = (layout == 'R' || layout == 'r');
    MatrixBatch<T> A(nm, n, n);
    for (int idx = 0; idx < nm; ++idx)
        gen_sym_ldlt(A.view(idx));
    std::vector<T> ap = pack_compact(A, n, V, rowmajor);
    std::vector<T> ap2 = ap;

    int info_f = compact<T>::sytrfnp(layout, uplo, n, ap.data(), n, V, nm);
    T b_dummy = 0; // never referenced at nrhs = 0, present per Fortran semantics
    int info_v =
        compact<T>::sysvnp(layout, uplo, n, 0, ap2.data(), n, &b_dummy, n, V, nm);

    const bool same = (ap == ap2);
    bool ok = (info_f == 0) && (info_v == 0) && same;
    std::printf("T=%-6s V=%-2d uplo=%c lay=%c nm=%-2d n=%-3d nrhs=0 | sysv==trf:%s "
                "info=%d/%d %s\n",
                compact<T>::name, V, uplo, layout, nm, n, same ? "yes" : "NO", info_f,
                info_v, ok ? "OK" : "FAIL");
    return !ok;
}

// -------- zero on the input diagonal, nonsingular leading minors -----
// The pivots are the *updated* Schur-complement diagonal entries, so a zero on
// the original diagonal is harmless as long as no leading principal minor
// vanishes. Handcrafted: A = L D L^T with L(1,0) = 1, D = diag(2,-2,1,-1,...)
// gives A(1,1) = 1*2*1 + (-2) = 0 while the minors are prod(d) != 0. The
// factorization must stay finite and recover (L, D) exactly as the scalar
// reference does; Cholesky would already have failed on d_1 < 0.

template <class T, int V> static int run_zerodiag(char uplo, char layout)
{
    const int n = 4, nm = V; // one full pack
    const bool rowmajor = (layout == 'R' || layout == 'r');
    const bool upper = (uplo == 'U' || uplo == 'u');

    // The factor as ?sytrfnp stores it: D on the diagonal, the unit-diagonal L
    // strictly below it. D = diag(2, -2, 1, -1) with L(1,0) = 1 puts a zero on
    // A's diagonal, and prod(D) != 0 keeps every leading minor nonsingular.
    const T diag[] = {T(2), T(-2), T(1), T(-1)};
    std::vector<T> f((size_t)n * n, T(0));
    const auto F = mat_view(f.data(), n, n);
    for (int j = 0; j < n; ++j)
        F(j, j) = diag[j];
    F(1, 0) = T(1); // makes A(1,1) = d0 * 1 + d1 = 0
    F(2, 0) = T(0.5);
    F(3, 1) = T(-0.75);
    F(3, 2) = T(0.25);

    MatrixBatch<T> A(nm, n, n), Aref(nm, n, n);
    for (int idx = 0; idx < nm; ++idx) {
        const auto Ain = A.view(idx);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                Ain(i, j) = (T)ldlt_reconstruct(F, i, j, false);
        std::copy(A[idx], A[idx] + (size_t)n * n, Aref[idx]);
        ref_sytf2np(uplo, Aref.view(idx));
    }

    std::vector<T> ap = pack_compact(A, n, V, rowmajor);
    int info = compact<T>::sytrfnp(layout, uplo, n, ap.data(), n, V, nm);
    MatrixBatch<T> Aout(nm, n, n);
    unpack_compact(Aout, ap.data(), n, V, rowmajor);

    double r = 0;
    bool finite = true;
    for (int idx = 0; idx < nm; ++idx) {
        const auto Fac = Aout.view(idx), Ref = Aref.view(idx);
        double e = 0;
        for (int j = 0; j < n; ++j) {
            double cs = 0;
            for (int i = 0; i < n; ++i) {
                const bool named = upper ? (i <= j) : (i >= j);
                if (!named) continue;
                if (!std::isfinite((double)Fac(i, j))) finite = false;
                cs += std::abs((double)Fac(i, j) - Ref(i, j));
            }
            e = std::max(e, cs);
        }
        r = std::max(r, test_ratio<T>(e, n, norm1(A.view(idx))));
    }
    const bool ok = finite && passes(r) && (info == 0);
    std::printf("T=%-6s V=%-2d uplo=%c lay=%c zero-diagonal A(1,1)=0 | err:%-5.2g "
                "finite:%s %s\n",
                compact<T>::name, V, uplo, layout, r, finite ? "yes" : "NO",
                ok ? "OK" : "FAIL");
    return !ok;
}

// ------------- zero-pivot lane isolation (design section 6.2) --------
// A single lane with a singular leading 1x1 minor (A(0,0) = 0, A(0,1) != 0)
// shares a pack with factorable siblings. The routine takes no safeguarded path
// and reports no error (info stays 0): the bad lane's 1/0 poisons *its own*
// factor with Inf/NaN -- while every sibling lane, computed with the same
// unmasked SIMD instructions, must stay finite and match the scalar reference.

template <class T, int V> static int run_zeropivot(int n, char uplo, char layout)
{
    const bool rowmajor = (layout == 'R' || layout == 'r');
    const bool upper = (uplo == 'U' || uplo == 'u');
    const int nm = V;      // one full pack, so all V lanes are exercised together
    const int badlane = 1; // this lane has a zero pivot; the rest factor fine

    MatrixBatch<T> A(nm, n, n), Aref(nm, n, n);
    for (int idx = 0; idx < nm; ++idx) {
        if (idx == badlane) {
            // symmetric, but the leading 1x1 minor is singular: the very first
            // pivot is A(0,0) = 0, so invd = 1/0 = Inf poisons the lane.
            const auto Bad = A.view(idx);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    Bad(i, j) = (i == j) ? T(1) : T(0);
            Bad(0, 0) = T(0);
            Bad(0, 1) = Bad(1, 0) = T(1);
        }
        else {
            gen_sym_ldlt(A.view(idx));
            std::copy(A[idx], A[idx] + (size_t)n * n, Aref[idx]);
            ref_sytf2np(uplo, Aref.view(idx));
        }
    }

    std::vector<T> ap = pack_compact(A, n, V, rowmajor);
    int info = compact<T>::sytrfnp(layout, uplo, n, ap.data(), n, V, nm);
    MatrixBatch<T> Aout(nm, n, n);
    unpack_compact(Aout, ap.data(), n, V, rowmajor);

    double r_sib = 0; // worst ratio over the factorable sibling lanes
    bool sib_finite = true;
    bool bad_poisoned = false; // the zero-pivot lane must carry Inf/NaN
    for (int idx = 0; idx < nm; ++idx) {
        const auto Fac = Aout.view(idx), Ref = Aref.view(idx);
        double e_sib = 0;
        for (int j = 0; j < n; ++j) {
            double cs = 0;
            for (int i = 0; i < n; ++i) {
                const bool named = upper ? (i <= j) : (i >= j);
                if (!named) continue;
                const T x = Fac(i, j);
                if (idx == badlane) {
                    if (!std::isfinite((double)x)) bad_poisoned = true;
                }
                else {
                    if (!std::isfinite((double)x)) sib_finite = false;
                    cs += std::abs((double)x - Ref(i, j));
                }
            }
            e_sib = std::max(e_sib, cs);
        }
        if (idx != badlane)
            r_sib = std::max(r_sib, test_ratio<T>(e_sib, n, norm1(A.view(idx))));
    }

    const bool ok_sib = sib_finite && passes(r_sib); // siblings uncontaminated & correct
    const bool ok_bad = bad_poisoned;                // poison confined but present
    const bool ok_info = (info == 0);                // GIGO: no early-exit, no error

    std::printf(
        "T=%-6s V=%-2d uplo=%c lay=%c n=%-3d zero-pivot lane | siblings:%-5.2g %s "
        "poison:%s info=%d %s\n",
        compact<T>::name, V, uplo, layout, n, r_sib, ok_sib ? "OK" : "FAIL",
        ok_bad ? "OK" : "FAIL", info, ok_info ? "OK" : "FAIL");
    return (!ok_sib) + (!ok_bad) + (!ok_info);
}

// --------------------- C API argument validation --------------------

// The factor signature is shared with ?potrf and the solve signature with
// ?potrs/?posv -- one validator each in cbk.cpp -- so the tables live
// in test_compact_util.hpp and are run here against this family's three entry
// points.
static int test_validation()
{
    std::vector<ApiCheck> t;
    append_factor_api_checks(t, "trf", dsytrfnp_compact);
    append_solve_api_checks(t, "trs", dsytrsnp_compact);
    append_solve_api_checks(t, "sv", dsysvnp_compact);
    return report_api_checks(t.data(), t.size());
}

// ------------------------------- main --------------------------------

int main()
{
    int fails = 0;
    fails += test_validation();

    // The hand-rolled reference against the library first.
    const char uplos[] = {'L', 'U'};
    const char lays[] = {'C', 'R'};
    for (char u : uplos) {
        fails += test_reference<double>(30, u);
        fails += test_reference<double>(43, u);
        fails += test_reference<float>(24, u);
    }

    // Full feature matrix: both uplo x both layouts, several (V, n, nm),
    // including padded partial final groups (nm not a multiple of V).
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

    // End-to-end solve A X = B on indefinite batches, closing the pipeline
    // (factor + sytrsnp, and the fused sysvnp), over uplo/layout/precision,
    // padded groups, and RHS counts that exercise trsm's 4/2/1 column blocks.
    for (char u : uplos)
        for (char l : lays) {
            fails += run_solve<double, 4>(8, 30, 5, u, l);
            fails += run_solve<double, 8>(11, 43, 4, u, l); // padded partial group
            fails += run_solve<double, 2>(6, 17, 1, u, l);  // single RHS
            fails += run_solve<float, 8>(16, 24, 3, u, l);
        }

    // sytrsnp against the trsm sweeps it is built from, over uplo/layout.
    for (char u : uplos)
        for (char l : lays) {
            fails += run_compose<double, 4>(8, 30, 5, u, l);
            fails += run_compose<float, 8>(11, 24, 3, u, l); // padded partial group
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

    // nrhs = 0 must factor anyway (LAPACK ?sysv), bit-identical to sytrfnp,
    // bp a never-referenced dummy.
    fails += run_nrhs0<double, 4>(6, 20, 'L', 'C');
    fails += run_nrhs0<float, 8>(9, 16, 'U', 'R');

    // Zero on the input diagonal (nonsingular minors): must factor cleanly.
    for (char u : uplos)
        for (char l : lays) {
            fails += run_zerodiag<double, 4>(u, l);
            fails += run_zerodiag<float, 8>(u, l);
        }

    // Zero-pivot lane isolation (design 6.2): a poisoned lane must not
    // contaminate its siblings, over both uplo, both layouts, both precisions.
    for (char u : uplos)
        for (char l : lays) {
            fails += run_zeropivot<double, 4>(20, u, l);
            fails += run_zeropivot<float, 8>(16, u, l);
        }

    // 10 groups: takes the OpenMP group loop when the team has <= 10 threads.
    fails += run_case<double, 4>(40, 20, 'L', 'C');
    fails += run_solve<double, 4>(40, 20, 4, 'L', 'C');

    return finish(fails);
}
