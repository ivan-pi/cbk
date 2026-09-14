// test_lapack_util.hpp
//
// The dense references of every test suite, on a real LAPACKE + CBLAS stack
// (issue #27): lapack<T>, a struct of static forwarders to the LAPACKE_?* and
// cblas_?* entry points specialized for double (d) and float (s), and over it
// the reference procedures the suites call by name -- ref_geqr2, ref_orm2r,
// ref_potf2, ref_trsm, ref_gels, ..., matmul, tri_apply, solve_errors -- each
// taking the project's MatrixView operands and forwarding their `.data` and
// leading dimensions to the library. The only reference the suites still
// hand-roll is the unpivoted LDL^T of test_sytrfnp_compact.cpp (LAPACK has no
// such routine), and that one is validated against this library too.
//
// Which headers to include is the stack's business, decided by
// cmake/FindLAPACKE.cmake through one definition on the LAPACKE::LAPACKE
// target: MKL's own <mkl_lapacke.h>/<mkl_cblas.h> when the tests are built
// with the MKL extension (a test binary must never link two LAPACK stacks),
// the plain <lapacke.h>/<cblas.h> of OpenBLAS or Netlib otherwise.
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
#include <Accelerate/Accelerate.h>
#include <lapacke.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace cbk::test {

// ----------------------- LAPACKE / CBLAS, by scalar type -------------
// lapack<T>::geqr2 / geqrf / gelqf / orgqr / ormqr / geqp3 / potrf / gels /
// larfg forward to LAPACKE_?*, gemm / trsm / trmm to cblas_?*. One macro
// generates both specializations so the two cannot drift apart.

template <class T> struct lapack;

// clang-format off
// NOLINTBEGIN(bugprone-macro-parentheses): T is a type name, p a token to paste
#define CBK_TEST_LAPACK_DISPATCH(T, p)                                                     \
template <> struct lapack<T> {                                                             \
    static lapack_int larfg(lapack_int n, T *alpha, T *x, lapack_int incx, T *tau)        \
    { return LAPACKE_##p##larfg(n, alpha, x, incx, tau); }                                 \
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
    static lapack_int gels(int layout, char trans, lapack_int m, lapack_int n,            \
                           lapack_int nrhs, T *a, lapack_int lda, T *b, lapack_int ldb)   \
    { return LAPACKE_##p##gels(layout, trans, m, n, nrhs, a, lda, b, ldb); }               \
    static void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb,         \
                     lapack_int m, lapack_int n, lapack_int k, T alpha, const T *a,        \
                     lapack_int lda, const T *b, lapack_int ldb, T beta, T *c,             \
                     lapack_int ldc)                                                       \
    { cblas_##p##gemm(layout, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }     \
    static void trsm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,               \
                     CBLAS_TRANSPOSE ta, CBLAS_DIAG diag, lapack_int m, lapack_int n,      \
                     T alpha, const T *a, lapack_int lda, T *b, lapack_int ldb)            \
    { cblas_##p##trsm(layout, side, uplo, ta, diag, m, n, alpha, a, lda, b, ldb); }        \
    static void trmm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,               \
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
// one layout flag per call plus a leading dimension per operand. A view maps
// without a copy when one stride is 1 and the other is at least the extent
// along it -- column-major (si == 1, ld = sj) or row-major (sj == 1, ld = si);
// the ambiguous degenerate shapes (a single row or column) are read as
// whichever description satisfies the library's ld >= extent rule.

template <class Mv> bool is_rowmajor(const Mv &M)
{
    if (M.si == 1 && M.sj >= std::max(M.rows, 1)) return false;
    assert(M.sj == 1 && M.si >= std::max(M.cols, 1) && "view is not a BLAS layout");
    return true;
}

template <class Mv> lapack_int lapack_ld(const Mv &M)
{
    return std::max<lapack_int>(is_rowmajor(M) ? M.si : M.sj, 1);
}

template <class Mv> int lapack_layout(const Mv &M)
{
    return is_rowmajor(M) ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;
}

template <class Mv> CBLAS_LAYOUT cblas_layout(const Mv &M)
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

// An operand of a call whose layout is fixed by another operand: the view
// itself when its layout already matches, else a contiguous staging copy in
// the call's layout -- copied back on destruction when the operand is
// writable (T non-const). Only the LAPACKE routines without a transpose flag
// for the operand need it (ormqr's reflectors, gels's right-hand side); the
// BLAS-3 calls read a mismatched operand as its transpose instead.
template <class T> class Staged {
  public:
    Staged(MatrixView<T> M, bool rowmajor) : orig_(M), copied_(is_rowmajor(M) != rowmajor)
    {
        if (!copied_) {
            use_ = M;
            return;
        }
        buf_.resize((std::size_t)M.rows * M.cols);
        const auto w = mat_view(buf_.data(), M.rows, M.cols, rowmajor ? M.cols : M.rows,
                                rowmajor);
        copy_matrix(M, w);
        use_ = MatrixView<T>{w.data, w.si, w.sj, w.rows, w.cols};
    }
    ~Staged()
    {
        if constexpr (!std::is_const_v<T>) {
            if (copied_) copy_matrix(use_, orig_);
        }
    }
    Staged(const Staged &) = delete;
    Staged &operator=(const Staged &) = delete;

    T *data() const { return use_.data; }
    lapack_int ld() const { return lapack_ld(use_); }

  private:
    MatrixView<T> orig_;
    bool copied_;
    std::vector<std::remove_const_t<T>> buf_;
    MatrixView<T> use_{};
};

// A read-only view of the same storage, whatever the constness of the view
// handed in (distinct MatrixView instantiations do not convert).
template <class Mv> ConstMatrixView<elem_t<Mv>> as_const_view(const Mv &M)
{
    return {M.data, M.si, M.sj, M.rows, M.cols};
}

// ----------------------- reference procedures ------------------------
// The names the suites call, each forwarded to the library. LAPACK's blocked
// drivers (?geqrf, ?ormqr, ?orgqr, ?potrf, ?gels) run their unblocked
// counterparts below the crossover order their ilaenv settings pick (which
// covers most of the suites' sizes) and a blocked variant above it, so a
// comparison against them is a comparison of two backward-stable
// implementations of the same factorization, not of the same operation
// sequence: the suites gate such comparisons relative to the operand norms,
// at a multiple of n * eps. Only ?geqr2 and ?larfg are unblocked by
// definition, and only those two are compared elementwise at ~eps.

// ?larfg: reflector from (alpha, x[0..m-2]); alpha := beta on exit.
template <class T> void ref_larfg(int m, T *alpha, T *x, T *tau)
{
    lapack<T>::larfg(m, alpha, x, 1, tau);
}

// ?geqr2: unblocked Householder QR, (H, tau) in the LAPACK convention.
template <class T> void ref_geqr2(MatrixView<T> A, T *tau)
{
    lapack<T>::geqr2(lapack_layout(A), A.rows, A.cols, A.data, lapack_ld(A), tau);
}

// ?geqrf: the blocked driver, for the invariants suites.
template <class T> void ref_geqrf(MatrixView<T> A, T *tau)
{
    lapack<T>::geqrf(lapack_layout(A), A.rows, A.cols, A.data, lapack_ld(A), tau);
}

// ?ormqr, side='L': B := Q^T B (trans 'T') or Q B ('N') from (H, tau), with
// k reflectors read from A (m x k or wider) and applied to B (m x nrhs).
template <class T, class Av>
void ref_orm2r(char trans, int k, Av A, const T *tau, MatrixView<T> B)
{
    assert(A.rows == B.rows && k <= std::min(A.rows, A.cols));
    const bool row = is_rowmajor(B);
    const Staged<const elem_t<Av>> As(as_const_view(A), row);
    lapack<T>::ormqr(lapack_layout(B), 'L', trans, B.rows, B.cols, k, As.data(), As.ld(),
                     tau, B.data, lapack_ld(B));
}

// ?orgqr: generate the first n columns of Q = H(0)..H(k-1) in place over the
// reflectors in columns 0..k-1 of A (m x n, m >= n >= k).
template <class T> void ref_org2r(int k, MatrixView<T> A, const T *tau)
{
    assert(k <= A.cols && A.cols <= A.rows);
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
template <class T> lapack_int ref_potf2(char uplo, MatrixView<T> A)
{
    assert(A.rows == A.cols);
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
// ?gels leaves that factorization in A but keeps tau in its workspace, so
// (A, tau) are taken from ?geqrf / ?gelqf on a copy of the input, which is
// the same call ?gels makes.
template <class T> void ref_gels(char trans, MatrixView<T> A, MatrixView<T> B, T *tau)
{
    const int m = A.rows, n = A.cols, nrhs = B.cols;
    assert(B.rows == std::max(m, n));
    const bool row = is_rowmajor(A);

    std::vector<T> A0s((std::size_t)m * n);
    const auto A0 = mat_view(A0s.data(), m, n, row ? n : m, row);
    copy_matrix(A, A0);

    {
        const Staged<T> Bs(B, row);
        lapack<T>::gels(lapack_layout(A), trans, m, n, nrhs, A.data, lapack_ld(A),
                        Bs.data(), Bs.ld());
    }
    if (m >= n)
        lapack<T>::geqrf(lapack_layout(A0), m, n, A0.data, lapack_ld(A0), tau);
    else
        lapack<T>::gelqf(lapack_layout(A0), m, n, A0.data, lapack_ld(A0), tau);
    copy_matrix(A0, A);
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
    const bool left = (side == 'L' || side == 'l');
    const int s = left ? R.rows : R.cols;
    assert(A.rows == s && A.cols == s && X.rows == R.rows && X.cols == R.cols);
    (void)s;
    copy_matrix(X, R);
    const bool row = is_rowmajor(R);
    CBLAS_TRANSPOSE ta = cblas_trans(transa);
    CBLAS_UPLO ul = cblas_uplo(uplo);
    tri_operand(row, A, ta, ul);
    lapack<T>::trmm(cblas_layout(R), cblas_side(side), ul, ta, cblas_diag(diag), R.rows,
                    R.cols, T(1), A.data, lapack_ld(A), R.data, lapack_ld(R));
}

// ----------------------- shared checks -------------------------------

// The end-to-end solve gate every solving suite closes with: the worst
// relative forward error of Xhat against the known X, and the worst relative
// residual ||A Xhat - B|| formed by ?gemm -- so a solver bug cannot hide
// behind the factorization that produced Xhat.
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
