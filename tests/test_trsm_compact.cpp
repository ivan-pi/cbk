// test_trsm_compact.cpp
//
// Validation of the templated compact triangular solve (dtrsm_compact /
// strsm_compact) against CBLAS (test_lapack_util.hpp): the reference is
// cblas_?trsm -- the same math the vectorized kernel executes V lanes at a
// time -- and the residual is formed by cblas_?trmm.
//
// Two parts:
//   1. C API argument validation -- exercises the LAPACK/BLAS-style info = -j
//      contract of the portable entry points.
//   2. Numerical correctness over side / uplo / transa / diag, precisions and
//      interleave widths (column-major, the tuned path), as test ratios
//      against THRESH (test_compact_util.hpp), LAPACK's TR path's: dget04's
//      forward error vs the exact solution (B = op(A) X for a random X, as
//      dlarhs) and vs the library's solve, plus dtrt02's residual
//      ||alpha B - op(A) X|| formed with the library's triangular multiply
//      (so a bug shared by the reference and the kernel cannot pass unseen),
//      over the suite's shapes
//      and the small-dimension cross product (small_dims: m x n from 0 to 5,
//      the empty operand and the 1/2/3-column route included).
//
// Assisted-by: Claude:claude-opus-4-8 Claude:claude-fable-5

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#include "test_lapack_util.hpp" // compact<T>, ref_trsm, tri_apply, gen_tri, pack/unpack

using namespace cbk::test;

// The reference is ref_trsm of test_lapack_util.hpp (cblas_?trsm over views,
// in place); tri_apply (cblas_?trmm) forms the residual independently.
// Padded pack slots carry the identity -- for a triangular A that is a unit
// diagonal, so the kernel's divisions never hit a zero pivot in the padding.

// --------------------------- one numerical case ---------------------
// Column-major; A is the order-s triangular factor, B is m x n.

template <class T, int V>
static int run_case(char side, char uplo, char transa, char diag, int nm, int m, int n)
{
    const bool left = (side == 'L');
    const int s = left ? m : n;
    const T alpha = T(0.5) + frand<T>(); // a non-trivial, non-zero scalar

    // triangular A (s x s), column-major: random in the referenced triangle,
    // diagonal boosted for conditioning, the other triangle zeroed.
    MatrixBatch<T> A(nm, s, s);
    const bool up = (uplo == 'U');
    for (int idx = 0; idx < nm; ++idx)
        gen_tri(A.view(idx), up);

    // a random solution X (dlarhs's), the right-hand side B = op(A) X (side
    // 'L') or X op(A) ('R') by the library's triangular multiply -- so the
    // solve of op(A) X = alpha B has the exact solution alpha X -- and the
    // library's own solution of the same system
    MatrixBatch<T> X(nm, m, n), B(nm, m, n), Xref(nm, m, n);
    for (int idx = 0; idx < nm; ++idx) {
        for (size_t e = 0; e < (size_t)m * n; ++e)
            X[idx][e] = frand<T>();
        tri_apply(side, uplo, transa, diag, A.view(idx), X.view(idx), B.view(idx));
        std::copy(B[idx], B[idx] + (size_t)m * n, Xref[idx]);
        ref_trsm(side, uplo, transa, diag, alpha, A.view(idx), Xref.view(idx));
    }

    // pack, solve with the routine under test, unpack
    const int lda = std::max(1, s), ldb = std::max(1, m);
    std::vector<T> ap = pack_compact(A, lda, V);
    std::vector<T> bp = pack_compact(B, ldb, V);

    int info = compact<T>::trsm('C', side, uplo, transa, diag, m, n, alpha, ap.data(),
                                lda, bp.data(), ldb, V, nm);

    MatrixBatch<T> Bout(nm, m, n);
    unpack_compact(Bout, bp.data(), ldb, V);

    // Three gates, LAPACK's TR path's: dget04's forward error against the
    // exact solution alpha X and against the library's solution, both
    // discounted by rcond of the operator actually applied -- E = op(A) with
    // the unit diagonal honored, formed by the library's triangular multiply
    // of the identity -- and dtrt02's residual, per right-hand side,
    // ||alpha b_j - op(A) x_j||_1 / (||op(A)||_1 ||x_j||_1 eps), formed with
    // the same multiply, so a bug shared by ref_trsm and the kernel cannot
    // slip through. Side 'R' is the transposed system op(A)^T X^T = alpha B^T:
    // its right-hand sides are the rows, read through transposed views, and
    // its operator op(A)^T.
    double r_fwd = 0, r_ref = 0, r_res = 0;
    std::vector<T> Rs((size_t)m * n), aXs((size_t)m * n), Is((size_t)s * s, T(0)),
        Es((size_t)s * s);
    const auto R = mat_view(Rs.data(), m, n), aX = mat_view(aXs.data(), m, n);
    const auto I = mat_view(Is.data(), s, s), E = mat_view(Es.data(), s, s);
    for (int d = 0; d < s; ++d)
        I(d, d) = T(1);
    const auto sys = [left](MatrixView<T> M) { return left ? M : M.transposed(); };
    for (int idx = 0; idx < nm; ++idx) {
        tri_apply('L', uplo, transa, diag, A.view(idx), I, E); // E = op(A)
        const auto Eop = sys(E);
        const double rc = rcond1(Eop), anorm = norm1(Eop);
        for (size_t e = 0; e < (size_t)m * n; ++e)
            aXs[e] = alpha * X[idx][e];
        r_fwd = std::max(r_fwd, forward_ratio(sys(Bout.view(idx)), sys(aX), rc));
        r_ref =
            std::max(r_ref, forward_ratio(sys(Bout.view(idx)), sys(Xref.view(idx)), rc));
        tri_apply(side, uplo, transa, diag, A.view(idx), Bout.view(idx), R);
        for (size_t e = 0; e < (size_t)m * n; ++e)
            Rs[e] = alpha * B[idx][e] - Rs[e]; // alpha B - op(A) Xhat
        r_res = std::max(r_res, residual_ratio(anorm, sys(R), sys(Bout.view(idx))));
    }

    const bool ok = passes(r_fwd) && passes(r_ref) && passes(r_res);
    std::printf("  T=%-6s V=%-2d side=%c uplo=%c tr=%c diag=%c nm=%-2d m=%-3d n=%-3d | "
                "fwd %-5.2g %s ref %-5.2g %s res %-5.2g %s info=%d %s\n",
                compact<T>::name, V, side, uplo, transa, diag, nm, m, n, r_fwd,
                verdict(r_fwd), r_ref, verdict(r_ref), r_res, verdict(r_res), info,
                (ok && info == 0) ? "OK" : "FAIL");

    return (info != 0) + !ok;
}

// --------------------- C API argument validation --------------------

static int test_validation()
{
    const int m = 8, n = 6, V = 4, nm = 4;
    // buffers sized for the largest valid case (A up to m x m, B up to m x n)
    std::vector<double> A((size_t)m * m * V, 0), B((size_t)m * n * V, 0);
    auto call = [&A, &B](char lay, char si, char up, char tr, char di, int m_, int n_,
                         int lda_, int ldb_, int V_, int nm_) {
        return dtrsm_compact(lay, si, up, tr, di, m_, n_, 1.0, A.data(), lda_, B.data(),
                             ldb_, V_, nm_);
    };
    // clang-format off
    const ApiCheck t[] = {
        {"valid L col",  call('C','L','U','N','N', m, n,  m,   m,   V, nm),   0},
        {"valid R col",  call('C','R','L','T','U', m, n,  n,   m,   V, nm),   0},
        {"valid row",    call('R','L','U','N','N', m, n,  m,   n,   V, nm),   0},
        {"transa=C",     call('C','L','U','C','N', m, n,  m,   m,   V, nm),   0},
        {"bad layout",   call('X','L','U','N','N', m, n,  m,   m,   V, nm),  -1},
        {"bad side",     call('C','X','U','N','N', m, n,  m,   m,   V, nm),  -2},
        {"bad uplo",     call('C','L','X','N','N', m, n,  m,   m,   V, nm),  -3},
        {"bad transa",   call('C','L','U','X','N', m, n,  m,   m,   V, nm),  -4},
        {"bad diag",     call('C','L','U','N','X', m, n,  m,   m,   V, nm),  -5},
        {"m<0",          call('C','L','U','N','N', -1, n, m,   m,   V, nm),  -6},
        {"n<0",          call('C','L','U','N','N', m, -1, m,   m,   V, nm),  -7},
        {"ldap<m (L)",   call('C','L','U','N','N', m, n,  m-1, m,   V, nm), -10},
        {"ldap<n (R)",   call('C','R','U','N','N', m, n,  n-1, m,   V, nm), -10},
        {"ldbp<m (col)", call('C','L','U','N','N', m, n,  m,   m-1, V, nm), -12},
        {"ldbp<n (row)", call('R','L','U','N','N', m, n,  m,   n-1, V, nm), -12},
        {"bad V",        call('C','L','U','N','N', m, n,  m,   m,   3, nm), -13},
        {"nm<0",         call('C','L','U','N','N', m, n,  m,   m,   V, -1), -14},
        {"empty m=0",    call('C','L','U','N','N', 0, n,  1,   1,   V, nm),   0},
        {"empty n=0",    call('C','L','U','N','N', m, 0,  m,   m,   V, nm),   0},
        {"empty nm=0",   call('C','L','U','N','N', m, n,  m,   m,   V, 0),    0},
    };
    // clang-format on
    return report_api_checks(t);
}

// ------------------------------- main --------------------------------

int main()
{
    std::printf("compact trsm portable test\n");

    int fails = 0;
    fails += test_validation();

    std::printf("numerical cases (column-major, vs cblas_?trsm):\n");
    // full side x uplo x transa x diag matrix at a representative shape/width
    for (char side : {'L', 'R'})
        for (char uplo : {'U', 'L'})
            for (char tr : {'N', 'T'})
                for (char di : {'N', 'U'})
                    fails += run_case<double, 4>(side, uplo, tr, di, 8, 12, 5);

    // precisions, widths, and a padded partial final group
    fails += run_case<double, 2>('L', 'U', 'N', 'N', 6, 16, 4);
    fails += run_case<double, 8>('L', 'L', 'T', 'N', 16, 24, 3);
    fails += run_case<double, 8>('L', 'U', 'N', 'N', 11, 20, 4); // padded last group
    fails += run_case<float, 8>('L', 'U', 'N', 'N', 16, 16, 4);
    fails += run_case<float, 16>('R', 'L', 'N', 'U', 32, 10, 7);
    fails += run_case<double, 4>('L', 'U', 'N', 'N', 40, 16, 4); // 10 groups: OpenMP path

    // few-RHS no-transpose left (n = 1,2,3): the column-axpy kernel route
    // (n >= 4 above routes to the row-dot kernel), across uplo / diag / width
    for (int nrhs : {1, 2, 3})
        for (char uplo : {'U', 'L'}) {
            fails += run_case<double, 4>('L', uplo, 'N', 'N', 8, 16, nrhs);
            fails += run_case<double, 8>('L', uplo, 'N', 'U', 11, 20, nrhs); // padded
        }
    fails += run_case<float, 8>('L', 'U', 'N', 'N', 16, 16, 1);
    fails += run_case<float, 16>('L', 'L', 'N', 'N', 32, 12, 2);

    // the small-dimension cross product (small_dims): m x n for one flag
    // combination per side, the empty operand and the 1/2/3-column route
    // included, a padded group
    for (int mm : small_dims)
        for (int nn : small_dims) {
            fails += run_case<double, 4>('L', 'U', 'N', 'N', 5, mm, nn);
            fails += run_case<float, 8>('R', 'L', 'T', 'U', 9, mm, nn);
        }

    return finish(fails);
}
