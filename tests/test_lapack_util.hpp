// test_lapack_util.hpp
//
// The dense references of every test suite, on a real LAPACKE + CBLAS stack
// (issue #27): lapack<T>, a struct of static forwarders to the LAPACKE_?* and
// cblas_?* entry points specialized for double (d) and float (s), and over it
// the reference procedures the suites call by name -- ref_geqr2, ref_ormqr,
// ref_potrf, ref_trsm, ref_gels, ..., matmul, tri_apply, solve_errors -- each
// taking the project's MatrixView operands and forwarding their `.data` and
// leading dimensions to the library. The only reference the suites still
// hand-roll is the unpivoted LDL^T of test_sytrfnp_compact.cpp (LAPACK has no
// such routine), and that one is validated against this library too.
//
// Which headers to include is the stack's business, decided by
// cmake/FindLAPACKE.cmake through one definition on the LAPACKE::LAPACKE
// target: MKL's own <mkl_lapacke.h>/<mkl_cblas.h> when the tests are built
// with the MKL extension (a test binary must never link two LAPACK stacks),
// the plain <lapacke.h>/<cblas.h> of OpenBLAS or Netlib otherwise, and on
// macOS accelerate-lapacke's <lapacke.h> with Accelerate's CBLAS.
// Integer arguments are lapack_int (MKL_INT under MKL, so an ilp64 MKL build
// gets its 64-bit integers); the CBLAS calls take the same lapack_int and
// convert to the stack's own BLAS integer, which matches on every stack the
// tests support (the other stacks are used at their 32-bit interface).
//
// Assisted-by: Claude:claude-fable-5

#ifndef TEST_LAPACK_UTIL_HPP
#define TEST_LAPACK_UTIL_HPP

#include "test_compact_util.hpp"

#if defined(CBK_LAPACKE_MKL)
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#elif defined(CBK_LAPACKE_ACCELERATE)
// macOS: LAPACKE from accelerate-lapacke (Netlib's LAPACKE over Accelerate's
// LAPACK 3.9.1 interface), CBLAS from Accelerate's vecLib sub-framework
// directly -- not through the <Accelerate/Accelerate.h> umbrella, whose LAPACK
// prototypes would collide with the ones lapacke.h pulls in. The framework
// search path for <vecLib/...> and ACCELERATE_NEW_LAPACK (the LAPACK 3.9.1
// interface the shim is built for) are on the LAPACKE::LAPACKE target.
#include <lapacke.h>
#include <vecLib/cblas.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

namespace cbk::test {

// ----------------------- LAPACKE / CBLAS, by scalar type -------------
// lapack<T>::geqr2 / geqrf / gelqf / orgqr / ormqr / geqp3 / potrf / getrf /
// gecon / gels forward to LAPACKE_?*, gemm / trsm / trmm to cblas_?*. One macro
// generates both specializations so the two cannot drift apart.

template <class T> struct lapack;

// clang-format off
// NOLINTBEGIN(bugprone-macro-parentheses): T is a type name, p a token to paste
#define CBK_TEST_LAPACK_DISPATCH(T, p)                                                     \
template <> struct lapack<T> {                                                             \
    static lapack_int geqr2(int layout, lapack_int m, lapack_int n, T *a, lapack_int lda, \
                            T *tau)                                                        \
    { return LAPACKE_##p##geqr2(layout, m, n, a, lda, tau); }                              \
    static lapack_int geqrf(int layout, lapack_int m, lapack_int n, T *a, lapack_int lda, \
                            T *tau)                                                        \
    { return LAPACKE_##p##geqrf(layout, m, n, a, lda, tau); }                              \
    static lapack_int gelqf(int layout, lapack_int m, lapack_int n, T *a, lapack_int lda, \
                            T *tau)                                                        \
    { return LAPACKE_##p##gelqf(layout, m, n, a, lda, tau); }                              \
    static lapack_int orgqr(int layout, lapack_int m, lapack_int n, lapack_int k, T *a,   \
                            lapack_int lda, const T *tau)                                  \
    { return LAPACKE_##p##orgqr(layout, m, n, k, a, lda, tau); }                           \
    static lapack_int ormqr(int layout, char side, char trans, lapack_int m,              \
                            lapack_int n, lapack_int k, const T *a, lapack_int lda,        \
                            const T *tau, T *c, lapack_int ldc)                            \
    { return LAPACKE_##p##ormqr(layout, side, trans, m, n, k, a, lda, tau, c, ldc); }      \
    static lapack_int geqp3(int layout, lapack_int m, lapack_int n, T *a, lapack_int lda, \
                            lapack_int *jpvt, T *tau)                                      \
    { return LAPACKE_##p##geqp3(layout, m, n, a, lda, jpvt, tau); }                        \
    static lapack_int potrf(int layout, char uplo, lapack_int n, T *a, lapack_int lda)    \
    { return LAPACKE_##p##potrf(layout, uplo, n, a, lda); }                                \
    static lapack_int getrf(int layout, lapack_int m, lapack_int n, T *a, lapack_int lda, \
                            lapack_int *ipiv)                                              \
    { return LAPACKE_##p##getrf(layout, m, n, a, lda, ipiv); }                             \
    static lapack_int gecon(int layout, char norm, lapack_int n, const T *a,              \
                            lapack_int lda, T anorm, T *rcond)                             \
    { return LAPACKE_##p##gecon(layout, norm, n, a, lda, anorm, rcond); }                  \
    static lapack_int gels(int layout, char trans, lapack_int m, lapack_int n,            \
                           lapack_int nrhs, T *a, lapack_int lda, T *b, lapack_int ldb)   \
    { return LAPACKE_##p##gels(layout, trans, m, n, nrhs, a, lda, b, ldb); }               \
    static void gemm(CBLAS_ORDER layout, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb,         \
                     lapack_int m, lapack_int n, lapack_int k, T alpha, const T *a,        \
                     lapack_int lda, const T *b, lapack_int ldb, T beta, T *c,             \
                     lapack_int ldc)                                                       \
    { cblas_##p##gemm(layout, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }     \
    static void trsm(CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,               \
                     CBLAS_TRANSPOSE ta, CBLAS_DIAG diag, lapack_int m, lapack_int n,      \
                     T alpha, const T *a, lapack_int lda, T *b, lapack_int ldb)            \
    { cblas_##p##trsm(layout, side, uplo, ta, diag, m, n, alpha, a, lda, b, ldb); }        \
    static void trmm(CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,               \
                     CBLAS_TRANSPOSE ta, CBLAS_DIAG diag, lapack_int m, lapack_int n,      \
                     T alpha, const T *a, lapack_int lda, T *b, lapack_int ldb)            \
    { cblas_##p##trmm(layout, side, uplo, ta, diag, m, n, alpha, a, lda, b, ldb); }        \
};
// NOLINTEND(bugprone-macro-parentheses)
// clang-format on

CBK_TEST_LAPACK_DISPATCH(double, d)
CBK_TEST_LAPACK_DISPATCH(float, s)
#undef CBK_TEST_LAPACK_DISPATCH

// ----------------------- views as LAPACKE / CBLAS arguments ----------
// A MatrixView carries its layout as strides (si, sj); LAPACKE and CBLAS take
// one layout flag per call plus a leading dimension per operand (the CBLAS
// one is spelled CBLAS_ORDER, the name every CBLAS header agrees on). A view maps
// without a copy when one stride is 1 and the other is at least the extent
// along it -- column-major (si == 1, ld = sj) or row-major (sj == 1, ld = si);
// the ambiguous degenerate shapes (a single row or column) are read as
// whichever description satisfies the library's ld >= extent rule. An empty
// extent constrains nothing (mat_view of a 0 x n matrix has sj = 0); the
// library's ld >= max(1, extent) is then lapack_ld's business.

template <class Mv> bool is_rowmajor(const Mv &M)
{
    if (M.si == 1 && (M.rows == 0 || M.sj >= std::max(M.rows, 1))) return false;
    assert(M.sj == 1 && (M.cols == 0 || M.si >= std::max(M.cols, 1)) &&
           "view is not a BLAS layout");
    return true;
}

template <class Mv> lapack_int lapack_ld(const Mv &M)
{
    return std::max<lapack_int>(M.ld(), 1);
}

template <class Mv> int lapack_layout(const Mv &M)
{
    return is_rowmajor(M) ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;
}

template <class Mv> CBLAS_ORDER cblas_layout(const Mv &M)
{
    return is_rowmajor(M) ? CblasRowMajor : CblasColMajor;
}

// The BLAS character arguments as CBLAS enumerators.
inline CBLAS_TRANSPOSE cblas_trans(char t)
{
    return (t == 'T' || t == 't' || t == 'C' || t == 'c') ? CblasTrans : CblasNoTrans;
}
inline CBLAS_TRANSPOSE cblas_flip(CBLAS_TRANSPOSE t)
{
    return t == CblasNoTrans ? CblasTrans : CblasNoTrans;
}
inline CBLAS_UPLO cblas_uplo(char u)
{
    return (u == 'U' || u == 'u') ? CblasUpper : CblasLower;
}
inline CBLAS_UPLO cblas_flip(CBLAS_UPLO u)
{
    return u == CblasUpper ? CblasLower : CblasUpper;
}
inline CBLAS_SIDE cblas_side(char s)
{
    return (s == 'L' || s == 'l') ? CblasLeft : CblasRight;
}
inline CBLAS_DIAG cblas_diag(char d)
{
    return (d == 'U' || d == 'u') ? CblasUnit : CblasNonUnit;
}

// The writable operand of a call whose layout another operand fixes: the
// view itself when its layout already matches, else a contiguous staging
// copy in the call's layout, copied back on destruction. Only the LAPACKE
// routines without a transpose flag need it (ormqr's and gels's right-hand
// sides, staged into the layout of the factor); the BLAS-3 calls read a
// mismatched operand as its transpose instead.
template <class T> class Staged {
  public:
    Staged(MatrixView<T> M, bool rowmajor) : orig_(M), use_(M)
    {
        if (is_rowmajor(M) == rowmajor) return;
        buf_.resize((std::size_t)M.rows * M.cols);
        use_ =
            mat_view(buf_.data(), M.rows, M.cols, rowmajor ? M.cols : M.rows, rowmajor);
        copy_matrix(M, use_);
    }
    ~Staged()
    {
        if (!buf_.empty()) copy_matrix(use_, orig_);
    }
    Staged(const Staged &) = delete;
    Staged &operator=(const Staged &) = delete;

    T *data() const { return use_.data; }
    lapack_int ld() const { return lapack_ld(use_); }

  private:
    MatrixView<T> orig_, use_;
    std::vector<T> buf_;
};

// ----------------------- reference procedures ------------------------
// The names the suites call, each forwarded to the library. LAPACK's blocked
// drivers (?geqrf, ?ormqr, ?orgqr, ?potrf, ?gels) run their unblocked
// counterparts below the crossover order their ilaenv settings pick (which
// covers most of the suites' sizes) and a blocked variant above it, so a
// comparison against them is a comparison of two backward-stable
// implementations of the same factorization, not of the same operation
// sequence: the suites gate such comparisons relative to the operand norms,
// at a multiple of n * eps. Only ?geqr2 is unblocked by definition, and only
// it is compared elementwise at ~eps.
//
// Each takes LAPACK's own quick return on an empty operand before the call:
// the routine would do nothing, but LAPACKE's NaN check runs first and an
// empty batch's operand is a possibly null pointer (MatrixBatch hands out
// std::vector storage), which not every stack's check guards.

template <class Mv> bool is_empty(const Mv &M)
{
    return M.rows == 0 || M.cols == 0;
}

// ?geqr2: unblocked Householder QR, (H, tau) in the LAPACK convention.
template <class T> void ref_geqr2(MatrixView<T> A, T *tau)
{
    if (is_empty(A)) return;
    lapack<T>::geqr2(lapack_layout(A), A.rows, A.cols, A.data, lapack_ld(A), tau);
}

// ?geqrf: the blocked driver, for the invariants suites.
template <class T> void ref_geqrf(MatrixView<T> A, T *tau)
{
    if (is_empty(A)) return;
    lapack<T>::geqrf(lapack_layout(A), A.rows, A.cols, A.data, lapack_ld(A), tau);
}

// ?ormqr, side='L': B := Q^T B (trans 'T') or Q B ('N') from (H, tau), with
// k reflectors read from A (m x k or wider) and applied to B (m x nrhs).
template <class T, class Av>
void ref_ormqr(char trans, int k, Av A, const T *tau, MatrixView<T> B)
{
    assert(A.rows == B.rows && k <= std::min(A.rows, A.cols));
    if (is_empty(B) || k == 0) return;
    const Staged<T> Bs(B, is_rowmajor(A));
    lapack<T>::ormqr(lapack_layout(A), 'L', trans, B.rows, B.cols, k, A.data,
                     lapack_ld(A), tau, Bs.data(), Bs.ld());
}

// ?orgqr: generate the first n columns of Q = H(0)..H(k-1) in place over the
// reflectors in columns 0..k-1 of A (m x n, m >= n >= k).
template <class T> void ref_orgqr(int k, MatrixView<T> A, const T *tau)
{
    assert(k <= A.cols && A.cols <= A.rows);
    if (is_empty(A)) return; /* k = 0 is not empty: Q = I(:, 0:n-1) is written */
    lapack<T>::orgqr(lapack_layout(A), A.rows, A.cols, k, A.data, lapack_ld(A), tau);
}

// ?geqp3: column-pivoted QR. On exit A holds the reflectors below the diagonal
// and R on/above it for the *permuted* matrix A(:,jpvt); jpvt[j] is the
// ORIGINAL (0-based) column index placed at position j, so A(:,jpvt) = Q R.
// (LAPACK's jpvt is 1-based and takes "free column" markers on input; both
// are handled here.)
template <class T> void ref_geqp3(MatrixView<T> A, lapack_int *jpvt, T *tau)
{
    std::fill(jpvt, jpvt + A.cols, 0);
    lapack<T>::geqp3(lapack_layout(A), A.rows, A.cols, A.data, lapack_ld(A), jpvt, tau);
    for (int j = 0; j < A.cols; ++j)
        jpvt[j] -= 1;
}

// ?potrf on a dense n x n matrix, in place. Lower: A = L L^T, factor in the
// lower triangle. Upper: A = U^T U, factor in the upper triangle. Only the
// named triangle is read or written. Returns LAPACK's info (> 0: not SPD).
template <class T> lapack_int ref_potrf(char uplo, MatrixView<T> A)
{
    assert(A.rows == A.cols);
    if (is_empty(A)) return 0;
    return lapack<T>::potrf(lapack_layout(A), uplo, A.rows, A.data, lapack_ld(A));
}

// The triangular operand of a BLAS-3 call in the layout of the call: an A
// stored the other way round is read as its transpose, with the transpose
// flag and the triangle flipped to compensate.
template <class Av>
void tri_operand(bool rowmajor, Av &A, CBLAS_TRANSPOSE &ta, CBLAS_UPLO &uplo)
{
    if (is_rowmajor(A) != rowmajor) {
        A = A.transposed();
        ta = cblas_flip(ta);
        uplo = cblas_flip(uplo);
    }
}

// BLAS ?trsm: solves op(A) X = alpha B (side='L') or X op(A) = alpha B
// (side='R') in place, A the order-s triangular factor; alpha == 0 gives
// B := 0. Only the referenced triangle of A is touched; the diagonal is not
// referenced when diag='U'. The ?trsm suite's reference, and the triangular
// steps of ref_gels.
template <class T, class Av>
void ref_trsm(char side, char uplo, char transa, char diag, T alpha, Av A,
              MatrixView<T> B)
{
    const bool left = (side == 'L' || side == 'l');
    assert(A.rows == A.cols && A.rows == (left ? B.rows : B.cols));
    const bool row = is_rowmajor(B);
    CBLAS_TRANSPOSE ta = cblas_trans(transa);
    CBLAS_UPLO ul = cblas_uplo(uplo);
    tri_operand(row, A, ta, ul);
    lapack<T>::trsm(cblas_layout(B), cblas_side(side), ul, ta, cblas_diag(diag), B.rows,
                    B.cols, alpha, A.data, lapack_ld(A), B.data, lapack_ld(B));
}

// Back substitution R X = B with R the upper triangle of an n x n array.
template <class T, class Rv> void ref_trsm_upper(Rv R, MatrixView<T> B)
{
    ref_trsm('L', 'U', 'N', 'N', T(1), R, B);
}

// ?gels: the least-squares (op(A) with more rows than columns) or minimum-norm
// (more columns than rows) solution of op(A) X = B, op(A) = A ('N') or A^T
// ('T'), A m x n and B max(m,n) x nrhs, each in the layout its view carries.
// On exit B holds X (least squares: rows n..m-1 keep the residual), A the
// factorization of the tall orientation -- the QR of A (m >= n) or its LQ in
// ?gelqf storage (m < n) -- and tau its min(m,n) reflector scalars. LAPACK's
// ?gels keeps tau in its workspace, so it runs on a copy of A for X, and
// (A, tau) come from ?geqrf / ?gelqf on A itself, the same call ?gels makes.
// min(m, n) = 0 is ?gels's quick return, B := 0 over all max(m, n) rows (no
// factorization, no least-squares solution: the residual rows hold zeros too).
template <class T> void ref_gels(char trans, MatrixView<T> A, MatrixView<T> B, T *tau)
{
    const int m = A.rows, n = A.cols, nrhs = B.cols;
    assert(B.rows == std::max(m, n));
    if (is_empty(A)) {
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < B.rows; ++i)
                B(i, j) = T(0);
        return;
    }
    const bool row = is_rowmajor(A);

    std::vector<T> A0s((std::size_t)m * n);
    const auto A0 = mat_view(A0s.data(), m, n, row ? n : m, row);
    copy_matrix(A, A0);
    const Staged<T> Bs(B, row);
    lapack<T>::gels(lapack_layout(A0), trans, m, n, nrhs, A0.data, lapack_ld(A0),
                    Bs.data(), Bs.ld());

    if (m >= n)
        lapack<T>::geqrf(lapack_layout(A), m, n, A.data, lapack_ld(A), tau);
    else
        lapack<T>::gelqf(lapack_layout(A), m, n, A.data, lapack_ld(A), tau);
}

// C (m x n) := A (m x k) * B (k x n) by ?gemm, for forming right-hand sides
// and residuals. The shapes come from the views, which also carry the
// layout: the call runs in C's, and an operand stored the other way round is
// passed as its transpose.
template <class Av, class Bv, class Cv> void matmul(Av A, Bv B, Cv C)
{
    using T = elem_t<Cv>;
    assert(A.rows == C.rows && B.cols == C.cols && A.cols == B.rows);
    const int k = A.cols; // the inner extent, before any operand is transposed
    const bool row = is_rowmajor(C);
    CBLAS_TRANSPOSE ta = CblasNoTrans, tb = CblasNoTrans;
    if (is_rowmajor(A) != row) {
        A = A.transposed();
        ta = CblasTrans;
    }
    if (is_rowmajor(B) != row) {
        B = B.transposed();
        tb = CblasTrans;
    }
    lapack<T>::gemm(cblas_layout(C), ta, tb, C.rows, C.cols, k, T(1), A.data,
                    lapack_ld(A), B.data, lapack_ld(B), T(0), C.data, lapack_ld(C));
}

// Apply a triangular operator to a general matrix by ?trmm -- the "forward"
// direction of a ?trsm, for checking a solve's defining residual
// ||op(A) X - alpha B||. R (m x n) := op(A) X (side 'L') or X op(A) (side
// 'R'), with A the order-s (s = m for 'L', n for 'R') triangular factor:
// uplo 'U'/'L', op(A) = A ('N') or A^T ('T'/'C'), unit ('U') or non-unit
// ('N') diagonal. Each operand carries its own layout in its view.
template <class Av, class Xv, class Rv>
void tri_apply(char side, char uplo, char transa, char diag, Av A, Xv X, Rv R)
{
    using T = elem_t<Rv>;
    assert(A.rows == A.cols &&
           A.rows == ((side == 'L' || side == 'l') ? R.rows : R.cols) &&
           X.rows == R.rows && X.cols == R.cols);
    copy_matrix(X, R);
    const bool row = is_rowmajor(R);
    CBLAS_TRANSPOSE ta = cblas_trans(transa);
    CBLAS_UPLO ul = cblas_uplo(uplo);
    tri_operand(row, A, ta, ul);
    lapack<T>::trmm(cblas_layout(R), cblas_side(side), ul, ta, cblas_diag(diag), R.rows,
                    R.cols, T(1), A.data, lapack_ld(A), R.data, lapack_ld(R));
}

// ----------------------- shared checks -------------------------------

// rcond = 1 / (||A||_1 ||A^-1||_1) of a square matrix, by ?getrf + ?gecon:
// the RCOND LAPACK's dget04 discounts a forward error by (forward_ratio).
// An empty matrix is perfectly conditioned; a singular one returns 0, which
// makes any forward error pass -- the ratio then says nothing, as in LAPACK.
template <class Av> double rcond1(Av A)
{
    using T = elem_t<Av>;
    assert(A.rows == A.cols);
    const int n = A.rows;
    if (n == 0) return 1.0;
    std::vector<T> LUs((std::size_t)n * n);
    std::vector<lapack_int> ipiv(n);
    const auto LU = mat_view(LUs.data(), n, n);
    copy_matrix(A, LU);
    const double anorm = norm1(LU);
    lapack<T>::getrf(LAPACK_COL_MAJOR, n, n, LU.data, n, ipiv.data());
    T rcond = 0;
    lapack<T>::gecon(LAPACK_COL_MAJOR, '1', n, LU.data, n, (T)anorm, &rcond);
    return (double)rcond;
}

// The end-to-end solve gate every solving suite closes with, as test ratios:
// dget02's / dpot02's residual, per right-hand side, ||b_j - A xhat_j||_1 /
// (||A||_1 ||xhat_j||_1 eps), the residual formed by ?gemm so a solver bug
// cannot hide behind the factorization that produced Xhat; and dget04's
// forward error against the known X, discounted by rcond(A). The worst over
// the batch of each.
struct SolveRatios {
    double res, fwd;
};

template <class T>
SolveRatios solve_ratios(const MatrixBatch<T> &A, const MatrixBatch<T> &B,
                         const MatrixBatch<T> &Xhat, ConstMatrixView<T> X)
{
    assert(B.rows() == Xhat.rows() && B.cols() == Xhat.cols() && X.rows == Xhat.rows() &&
           X.cols == Xhat.cols());
    std::vector<T> Rs(Xhat.stride());
    const auto R = mat_view(Rs.data(), Xhat.rows(), Xhat.cols());
    SolveRatios r{0, 0};
    for (int v = 0; v < Xhat.count(); ++v) {
        const auto Av = A.view(v), Bv = B.view(v), Xv = Xhat.view(v);
        matmul(Av, Xv, R); // R := B - A Xhat
        for (int j = 0; j < R.cols; ++j)
            for (int i = 0; i < R.rows; ++i)
                R(i, j) = Bv(i, j) - R(i, j);
        r.res = std::max(r.res, residual_ratio(norm1(Av), R, Xv));
        r.fwd = std::max(r.fwd, forward_ratio(Xv, X, rcond1(Av)));
    }
    return r;
}

// The same two, as relative errors (the MKL suites' gates).
struct SolveErrors {
    double fwd, res;
};

template <class T>
SolveErrors solve_errors(const MatrixBatch<T> &A, const MatrixBatch<T> &B,
                         const MatrixBatch<T> &Xhat, ConstMatrixView<T> X)
{
    assert(B.rows() == Xhat.rows() && B.cols() == Xhat.cols() && X.rows == Xhat.rows() &&
           X.cols == Xhat.cols());
    const size_t sB = Xhat.stride();
    std::vector<T> AXs(sB);
    const auto AX = mat_view(AXs.data(), Xhat.rows(), Xhat.cols());
    const double nX = std::max(norm1(X), norm_floor);
    SolveErrors e{0, 0};
    for (int v = 0; v < Xhat.count(); ++v) {
        e.fwd = std::max(e.fwd, max_abs_diff(Xhat[v], X.data, sB) / nX);
        matmul(A.view(v), Xhat.view(v), AX);
        e.res = std::max(e.res, max_abs_diff(AXs.data(), B[v], sB) /
                                    std::max(norm1(B.view(v)), norm_floor));
    }
    return e;
}

} // namespace cbk::test

#endif // TEST_LAPACK_UTIL_HPP
