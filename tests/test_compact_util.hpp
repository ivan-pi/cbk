// test_compact_util.hpp
//
// Shared helpers for the compact-format test suites: a seeded RNG, error
// metrics, input generation, scalar LAPACK-style reference kernels, precision
// overloads of the portable C API, and Compact pack/unpack. Header-only and
// MKL-free, so the BLAS-free portable tests use it too.
//
// Assisted-by: Claude:claude-opus-4.8

#ifndef TEST_COMPACT_UTIL_HPP
#define TEST_COMPACT_UTIL_HPP

#include "cqr_compact.h"
#include "cqr_compact_common.hpp"
#include "cqr_matrix_batch.hpp"
#include "cqr_matrix_view.hpp"

#include <cstdio>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>
#include <algorithm>

namespace cqr::test {

// The dense strided view every suite addresses its host-side matrices through
// (src/cqr_matrix_view.hpp), and the compact analogue the kernels use, which
// the pack/unpack helpers below address the interleaved side through
// (src/cqr_compact_common.hpp).
using cqr::detail::ConstMatrixView;
using cqr::detail::mat_view;
using cqr::detail::MatrixBatch;
using cqr::detail::MatrixView;

using cqr::detail::for_vlen;
using cqr::detail::group_stride;
using cqr::detail::make_const_view;
using cqr::detail::make_view;

// The scalar a view addresses. Helpers that only read take the view type
// itself, so one signature serves MatrixView<T> and ConstMatrixView<T>. The
// latter is just MatrixView<const T>, but distinct template instantiations do
// not convert into each other (the view is an aggregate, with no converting
// constructor), so a MatrixView<T> argument cannot bind a MatrixView<const T>
// parameter.
template <class Mv>
using elem_t = std::remove_const_t<
    std::remove_reference_t<decltype(std::declval<const Mv &>()(0, 0))>>;

// One RNG per test binary (each test is a separate executable, so there is no
// cross-test coupling); the seed only has to be fixed, not unique.
inline std::mt19937_64 &rng()
{
    static std::mt19937_64 g(2026);
    return g;
}

template <class T> T frand()
{
    static std::uniform_real_distribution<T> dist(T(-1), T(1));
    return dist(rng());
}

// Denominator floor for relative errors, err / max(norm, norm_floor): a
// nonzero norm in these suites is O(1) or larger, so the floor never shifts a
// ratio -- it only keeps a zero reference norm from producing 0/0 = NaN. It is
// a divide-by-zero guard (a normal double near DBL_MIN), not a tolerance.
constexpr double norm_floor = 1e-300;

// max |a - b| over n elements.
template <class T> double max_abs_diff(const T *a, const T *b, size_t n)
{
    double d = 0;
    for (size_t i = 0; i < n; ++i)
        d = std::max(d, (double)std::abs(a[i] - b[i]));
    return d;
}

// The leading rows x cols block of a matrix, as a view of the same storage.
template <class Mv> Mv leading(Mv M, int rows, int cols)
{
    assert(rows >= 0 && rows <= M.rows && cols >= 0 && cols <= M.cols);
    return {M.data, M.si, M.sj, rows, cols};
}

// L1 (max column sum) norm of a matrix, in any layout its view describes.
template <class Mv> double norm1(Mv M)
{
    double mx = 0;
    for (int j = 0; j < M.cols; ++j) {
        double s = 0;
        for (int i = 0; i < M.rows; ++i)
            s += std::abs(M(i, j));
        mx = std::max(mx, s);
    }
    return mx;
}

// Deviation of Q's columns from orthonormality, max |(Q^T Q - I)(i,j)| over
// the upper triangle (Q^T Q is symmetric), accumulated in double so the gate
// does not inherit float rounding. BLAS-free; shared by the orgqr suites.
template <class Qv> double orth_error(Qv Q)
{
    double e = 0;
    for (int j = 0; j < Q.cols; ++j)
        for (int i = 0; i <= j; ++i) {
            double s = 0;
            for (int l = 0; l < Q.rows; ++l)
                s += (double)Q(l, i) * Q(l, j);
            e = std::max(e, std::abs(s - (i == j ? 1.0 : 0.0)));
        }
    return e;
}

// C (m x n) := A (m x k) * B (k x n) -- the plain triple loop, for forming
// right-hand sides and residuals. The shapes come from the views, which also
// carry the layout, so the three operands need not share one.
template <class Av, class Bv, class Cv> void matmul(Av A, Bv B, Cv C)
{
    assert(A.rows == C.rows && B.cols == C.cols && A.cols == B.rows);
    for (int j = 0; j < C.cols; ++j)
        for (int i = 0; i < C.rows; ++i) {
            elem_t<Cv> s = 0;
            for (int l = 0; l < A.cols; ++l)
                s += A(i, l) * B(l, j);
            C(i, j) = s;
        }
}

// Elementwise copy dst <- src, same shape, each side in its own layout -- the
// staging step between a column-major batch and a row-major API surface.
template <class Sv, class T> void copy_matrix(Sv src, MatrixView<T> dst)
{
    assert(src.rows == dst.rows && src.cols == dst.cols);
    for (int j = 0; j < dst.cols; ++j)
        for (int i = 0; i < dst.rows; ++i)
            dst(i, j) = src(i, j);
}

// The known solution the solve checks recover: X(:,j) = j+1 (ones, twos, ...),
// column-major n x nrhs.
template <class T> std::vector<T> known_solution(int n, int nrhs)
{
    std::vector<T> Xs((size_t)n * nrhs);
    const auto X = mat_view(Xs.data(), n, nrhs);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i)
            X(i, j) = T(j + 1);
    return Xs;
}

// ----------------------- reference kernels (scalar) -----------------
// Column-major, in place: the unblocked LAPACK algorithms the vectorized
// kernels execute V lanes at a time, so a correct kernel matches them to
// working precision. Matrix operands are views; the vector operands of ?larfg
// stay pointers, since a reflector is a contiguous column segment and not a
// matrix (which is also why these references are column-major only -- the
// views they are handed must have unit row stride, as their col() asserts).

// dlarfg: reflector from (alpha, x[0..m-2]); alpha := beta on exit.
template <class T> void ref_larfg(int m, T *alpha, T *x, T *tau)
{
    T xnorm = 0;
    for (int i = 0; i < m - 1; ++i)
        xnorm = std::hypot(xnorm, x[i]);
    if (xnorm == T(0)) {
        *tau = 0;
        return;
    }
    T beta = -std::copysign(std::hypot(*alpha, xnorm), *alpha);
    *tau = (beta - *alpha) / beta;
    T scal = T(1) / (*alpha - beta);
    for (int i = 0; i < m - 1; ++i)
        x[i] *= scal;
    *alpha = beta;
}

// dgeqr2: unblocked Householder QR, (H, tau) in the LAPACK convention.
template <class T> void ref_geqr2(MatrixView<T> A, T *tau)
{
    const int m = A.rows, n = A.cols, k = std::min(m, n);
    for (int kk = 0; kk < k; ++kk) {
        // The reflector runs down column kk from the diagonal; A.col() gives
        // the contiguous column, so the tail below the diagonal is addressable
        // even when it is empty (kk + 1 == m on the last row).
        ref_larfg(m - kk, &A(kk, kk), A.col(kk) + kk + 1, &tau[kk]);
        for (int j = kk + 1; j < n; ++j) {
            T w = A(kk, j);
            for (int i = kk + 1; i < m; ++i)
                w += A(i, kk) * A(i, j);
            A(kk, j) -= tau[kk] * w;
            for (int i = kk + 1; i < m; ++i)
                A(i, j) -= tau[kk] * A(i, kk) * w;
        }
    }
}

// dorm2r, side='L': B := Q^T B (trans 'T') or Q B ('N') from (H, tau), with
// k reflectors read from A (m x k or wider) and applied to B (m x nrhs).
template <class T, class Av>
void ref_orm2r(char trans, int k, Av A, const T *tau, MatrixView<T> B)
{
    const int m = B.rows, nrhs = B.cols;
    assert(A.rows == m && k <= std::min(A.rows, A.cols));
    bool fwd = (trans == 'T');
    for (int s = 0; s < k; ++s) {
        int kk = fwd ? s : k - 1 - s;
        for (int j = 0; j < nrhs; ++j) {
            T w = B(kk, j);
            for (int i = kk + 1; i < m; ++i)
                w += A(i, kk) * B(i, j);
            B(kk, j) -= tau[kk] * w;
            for (int i = kk + 1; i < m; ++i)
                B(i, j) -= tau[kk] * A(i, kk) * w;
        }
    }
}

// Back substitution R X = B with R the upper triangle of an n x n array.
template <class T, class Rv> void ref_trsm_upper(Rv R, MatrixView<T> B)
{
    const int n = B.rows, nrhs = B.cols;
    assert(R.rows == n && R.cols == n);
    for (int j = 0; j < nrhs; ++j)
        for (int i = n - 1; i >= 0; --i) {
            T s = B(i, j);
            for (int l = i + 1; l < n; ++l)
                s -= R(i, l) * B(l, j);
            B(i, j) = s / R(i, i);
        }
}

// Dense BLAS ?trsm: solves op(A) X = alpha B (side='L') or X op(A) = alpha B
// (side='R') in place, A the order-s triangular factor. B is pre-scaled by
// alpha (so alpha == 0 gives B := 0), then a unit-alpha substitution runs. Only
// the referenced triangle of A is touched; the diagonal is skipped entirely
// when diag='U'. The ?trsm suite's reference, and the triangular steps of
// ref_gels.
template <class T, class Av>
void ref_trsm(char side, char uplo, char transa, char diag, T alpha, Av A,
              MatrixView<T> B)
{
    const bool left = (side == 'L' || side == 'l');
    const bool upper = (uplo == 'U' || uplo == 'u');
    const bool tran = (transa == 'T' || transa == 't' || transa == 'C' || transa == 'c');
    const bool unit = (diag == 'U' || diag == 'u');
    const int m = B.rows, n = B.cols;
    assert(A.rows == A.cols && A.rows == (left ? m : n));

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            B(i, j) *= alpha; // B := alpha B (alpha == 0 -> B := 0)

    if (left) {
        // solve op(A) X = B column by column; A is m x m
        const bool back = (upper != tran);
        for (int j = 0; j < n; ++j)
            for (int t = 0; t < m; ++t) {
                int i = back ? m - 1 - t : t;
                T s = B(i, j);
                if (back)
                    for (int l = i + 1; l < m; ++l)
                        s -= (tran ? A(l, i) : A(i, l)) * B(l, j);
                else
                    for (int l = 0; l < i; ++l)
                        s -= (tran ? A(l, i) : A(i, l)) * B(l, j);
                B(i, j) = unit ? s : s / A(i, i);
            }
    }
    else {
        // solve X op(A) = B column of X at a time; A is n x n
        const bool fwd = (upper != tran);
        for (int t = 0; t < n; ++t) {
            int j = fwd ? t : n - 1 - t;
            if (fwd)
                for (int l = 0; l < j; ++l) {
                    T a = tran ? A(j, l) : A(l, j);
                    for (int i = 0; i < m; ++i)
                        B(i, j) -= a * B(i, l);
                }
            else
                for (int l = j + 1; l < n; ++l) {
                    T a = tran ? A(j, l) : A(l, j);
                    for (int i = 0; i < m; ++i)
                        B(i, j) -= a * B(i, l);
                }
            if (!unit) {
                T d = A(j, j);
                for (int i = 0; i < m; ++i)
                    B(i, j) /= d;
            }
        }
    }
}

// dgels, unblocked: the least-squares (op(A) with more rows than columns) or
// minimum-norm (more columns than rows) solution of op(A) X = B, op(A) = A
// ('N') or A^T ('T'), A m x n and B max(m,n) x nrhs, each in the layout its
// view carries. Factors the tall orientation F (A, or A^T when m < n: that is
// the LQ of A in ?gelqf storage) with ref_geqr2 -- on exit A holds the
// factorization and tau its min(m,n) reflector scalars -- then B := Q^T B,
// R X = B (least squares: rows n..m-1 of B keep the residual) or R^T Y = B,
// X = Q [Y; 0] (minimum norm). The same steps the compact kernel runs V lanes
// at a time.
template <class T> void ref_gels(char trans, MatrixView<T> A, MatrixView<T> B, T *tau)
{
    const bool tran = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');
    const int m = A.rows, n = A.cols, nrhs = B.cols;
    const bool tall = (m >= n);
    const int p = tall ? m : n, q = tall ? n : m;
    const bool overdet = (tall != tran);
    assert(B.rows == p);

    // F: the tall orientation, p x q, column-major (the references need it so)
    std::vector<T> Fs((size_t)p * q);
    const auto F = mat_view(Fs.data(), p, q);
    copy_matrix(tall ? A : A.transposed(), F);

    ref_geqr2(F, tau);
    const auto R = leading(F, q, q), Bq = leading(B, q, nrhs);
    if (overdet) {
        ref_orm2r('T', q, F, tau, B);
        ref_trsm('L', 'U', 'N', 'N', T(1), R, Bq);
    }
    else {
        ref_trsm('L', 'U', 'T', 'N', T(1), R, Bq);
        for (int j = 0; j < nrhs; ++j)
            for (int i = q; i < p; ++i)
                B(i, j) = T(0);
        ref_orm2r('N', q, F, tau, B);
    }

    // hand the factorization back in A's orientation
    copy_matrix(F, tall ? A : A.transposed());
}

// ----------------------- portable C API, by scalar type ------------
// compact<T>::geqrf / ormqr / orgqr / potrf / sytrfnp / sytrsnp / sysvnp / trsm / gels forward
// to the d/s entry points of cqr_compact.h, so the templated suites call one name for both precisions;
// compact<T>::name labels their output.

template <class T> struct compact;

// clang-format off
// NOLINTBEGIN(bugprone-macro-parentheses): T is a type name, p a token to paste
#define CQR_TEST_COMPACT_DISPATCH(T, p, label)                                             \
template <> struct compact<T> {                                                            \
    static constexpr const char *name = label;                                             \
    static int geqrf(char lay, int m, int n, T *a, int ld, T *tau, int V, int nm)          \
    { return p##geqrf_compact(lay, m, n, a, ld, tau, V, nm); }                             \
    static int ormqr(char tr, int m, int nrhs, int k, const T *a, int lda, const T *tau,   \
                     T *b, int ldb, int V, int nm)                                         \
    { return p##ormqr_compact(tr, m, nrhs, k, a, lda, tau, b, ldb, V, nm); }               \
    static int orgqr(char lay, int m, int n, int k, T *a, int lda, const T *tau,           \
                     int V, int nm)                                                        \
    { return p##orgqr_compact(lay, m, n, k, a, lda, tau, V, nm); }                         \
    static int potrf(char lay, char up, int n, T *a, int ld, int V, int nm)                \
    { return p##potrf_compact(lay, up, n, a, ld, V, nm); }                                 \
    static int sytrfnp(char lay, char up, int n, T *a, int ld, int V, int nm)              \
    { return p##sytrfnp_compact(lay, up, n, a, ld, V, nm); }                               \
    static int sytrsnp(char lay, char up, int n, int nrhs, const T *a, int lda, T *b,      \
                       int ldb, int V, int nm)                                             \
    { return p##sytrsnp_compact(lay, up, n, nrhs, a, lda, b, ldb, V, nm); }                \
    static int sysvnp(char lay, char up, int n, int nrhs, T *a, int lda, T *b, int ldb,    \
                      int V, int nm)                                                       \
    { return p##sysvnp_compact(lay, up, n, nrhs, a, lda, b, ldb, V, nm); }                 \
    static int trsm(char lay, char si, char up, char tr, char di, int m, int n, T alpha,   \
                    const T *a, int lda, T *b, int ldb, int V, int nm)                     \
    { return p##trsm_compact(lay, si, up, tr, di, m, n, alpha, a, lda, b, ldb, V, nm); }   \
    static int gels(char lay, char tr, int m, int n, int nrhs, T *a, int lda, T *b,       \
                    int ldb, T *tau, int V, int nm)                                        \
    { return p##gels_compact(lay, tr, m, n, nrhs, a, lda, b, ldb, tau, V, nm); }           \
};
// NOLINTEND(bugprone-macro-parentheses)
// clang-format on

CQR_TEST_COMPACT_DISPATCH(double, d, "double")
CQR_TEST_COMPACT_DISPATCH(float, s, "float")
#undef CQR_TEST_COMPACT_DISPATCH

// ----------------------- input generation ----------------------------

// Random m x n matrix with the leading diagonal boosted by `boost`, which
// tames the conditioning of the square/tall QR and solve tests.
template <class T> void gen_boosted(MatrixView<T> A, T boost = T(2))
{
    for (int j = 0; j < A.cols; ++j)
        for (int i = 0; i < A.rows; ++i)
            A(i, j) = frand<T>();
    for (int d = 0; d < std::min(A.rows, A.cols); ++d)
        A(d, d) += boost;
}

// Symmetric positive-definite n x n matrix (column-major): A = M^T M + n*I, a
// tame condition number. cond > 0 squeezes the spectrum by a symmetric
// congruence D A D, D = diag(10^{-cond*i/(n-1)}) -- dynamic range, still SPD.
//
// The result is symmetric to the bit, not just to working precision. M^T M is
// symmetric in exact arithmetic, but A(i,j) and A(j,i) come from two separately
// evaluated dot products, so a value-unsafe FP model (icpx defaults to
// -fp-model=fast) can round them one ULP apart. The row-major "opposite triangle
// untouched" checks are sensitive to this: they read a row-major factor back
// through its transpose, comparing A(j,i) with A(i,j), so a sub-ULP asymmetry
// there reads as the kernel having written the wrong triangle. The trailing
// mirror pins A(j,i) == A(i,j) exactly, on every compiler and FP model.
template <class T> void gen_spd(MatrixView<T> A, double cond = 0.0)
{
    assert(A.rows == A.cols);
    const int n = A.rows;
    std::vector<T> Ms((size_t)n * n);
    const auto M = mat_view(Ms.data(), n, n);
    for (auto &x : Ms)
        x = frand<T>();
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            T s = 0;
            for (int l = 0; l < n; ++l)
                s += M(l, i) * M(l, j);
            A(i, j) = s + (i == j ? T(n) : T(0));
        }
    if (cond > 0.0)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                double si = std::pow(10.0, -cond * (n > 1 ? (double)i / (n - 1) : 0.0));
                double sj = std::pow(10.0, -cond * (n > 1 ? (double)j / (n - 1) : 0.0));
                A(i, j) *= (T)(si * sj);
            }
    // Mirror the lower triangle onto the upper so A(j,i) == A(i,j) bit-for-bit
    // (the congruence above scales A(i,j) and A(j,i) by the same si*sj, so it
    // preserves whatever symmetry M^T M produced; enforce it exactly here).
    for (int j = 0; j < n; ++j)
        for (int i = j + 1; i < n; ++i)
            A(j, i) = A(i, j);
}

// Symmetric *indefinite* n x n matrix (column-major) with a known-good
// unpivoted LDL^T: A = L D L^T built from a random unit-lower L (entries in
// (-0.5, 0.5), so element growth in re-factorization stays mild) and a diagonal
// D with |d| in [0.5, 2.5) and mixed signs (d_1 is forced negative for n >= 2,
// so the matrix is genuinely indefinite -- Cholesky would fail on it). Every
// leading principal minor is prod(d_1..d_k) != 0, so the unpivoted
// factorization exists and is exactly this (L, D). As in gen_spd, the upper
// triangle is mirrored from the lower so A is symmetric to the bit.
template <class T> void gen_sym_ldlt(MatrixView<T> A)
{
    assert(A.rows == A.cols);
    const int n = A.rows;
    std::vector<T> Ls((size_t)n * n, T(0)), d(n);
    const auto L = mat_view(Ls.data(), n, n);
    for (int j = 0; j < n; ++j) {
        L(j, j) = T(1);
        for (int i = j + 1; i < n; ++i)
            L(i, j) = T(0.5) * frand<T>();
        T mag = T(0.5) + std::abs(frand<T>()) * T(2);
        d[j] = (j == 1 || frand<T>() < 0) ? -mag : mag; // mixed signs, d_1 < 0
    }
    for (int j = 0; j < n; ++j)
        for (int i = j; i < n; ++i) {
            T s = 0;
            for (int l = 0; l <= j; ++l)
                s += L(i, l) * d[l] * L(j, l);
            A(i, j) = s;
        }
    for (int j = 0; j < n; ++j)
        for (int i = j + 1; i < n; ++i)
            A(j, i) = A(i, j);
}

// Element (i,j) of an n x n unpivoted LDL^T factor as stored by ?sytrfnp:
// A(i,j) = sum_{l <= min(i,j)} F(i,l) D(l) F(j,l) with F the unit factor read
// from the strict lower triangle (uplo 'L') or, for the transpose dual
// A = U^T D U, from the strict upper one (F(i,l) = U(l,i)). `at(i,j)` reads the
// stored factor. Shared by the reconstruction checks of both ?sytrfnp suites.
template <class At> double ldlt_reconstruct(const At &at, int i, int j, bool upper)
{
    double s = 0;
    for (int l = 0; l <= std::min(i, j); ++l) {
        const double fi = (l == i) ? 1.0 : (upper ? (double)at(l, i) : (double)at(i, l));
        const double fj = (l == j) ? 1.0 : (upper ? (double)at(l, j) : (double)at(j, l));
        s += fi * (double)at(l, l) * fj;
    }
    return s;
}

// Fill one square triangular matrix: random in the referenced triangle, the
// diagonal boosted away from zero for conditioning, the other (never
// referenced) triangle zeroed. Shared by the ?trsm suites; the layout rides on
// the view's strides, so one body serves column-major and row-major.
template <class T> void gen_tri(MatrixView<T> A, bool upper)
{
    assert(A.rows == A.cols);
    const int s = A.rows;
    for (int i = 0; i < s; ++i)
        for (int j = 0; j < s; ++j) {
            bool ref = upper ? (i <= j) : (i >= j);
            A(i, j) = ref ? frand<T>() : T(0);
        }
    for (int d = 0; d < s; ++d)
        A(d, d) = (A(d, d) >= 0 ? T(1) : T(-1)) * (T(2) + std::abs(frand<T>()));
}

// Apply a triangular operator to a general matrix -- the "forward" direction of
// a ?trsm, for checking a solve's defining residual ||op(A) X - alpha B||.
// R (m x n) := op(A) X (side 'L') or X op(A) (side 'R'), with A the order-s
// (s = m for 'L', n for 'R') triangular factor: uplo 'U'/'L', op(A) = A ('N')
// or A^T ('T'/'C'), unit ('U') or non-unit ('N') diagonal. The shapes come
// from R and A; each operand carries its own layout in its view.
template <class Av, class Xv, class Rv>
void tri_apply(char side, char uplo, char transa, char diag, Av A, Xv X, Rv R)
{
    using T = elem_t<Rv>;
    const bool left = (side == 'L' || side == 'l');
    const bool upper = (uplo == 'U' || uplo == 'u');
    const bool tran = (transa == 'T' || transa == 't' || transa == 'C' || transa == 'c');
    const bool unit = (diag == 'U' || diag == 'u');
    const int m = R.rows, n = R.cols, s = left ? m : n;
    assert(A.rows == s && A.cols == s && X.rows == m && X.cols == n);
    // op(A)(i,k): unit or A(i,i) on the diagonal; off-diagonal is the referenced
    // entry of A (op = A) or of its transpose (op = A^T), else zero.
    auto Mop = [&](int i, int k) -> T {
        if (i == k) return unit ? T(1) : A(i, i);
        const bool ref = tran ? (upper ? i > k : i < k) : (upper ? k > i : k < i);
        if (!ref) return T(0);
        return tran ? A(k, i) : A(i, k);
    };
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            T acc = 0;
            if (left)
                for (int k = 0; k < s; ++k)
                    acc += Mop(i, k) * X(k, j);
            else
                for (int k = 0; k < s; ++k)
                    acc += X(i, k) * Mop(k, j);
            R(i, j) = acc;
        }
}

// ----------------------- dense batches and compact packing -----------
// The dense batches are MatrixBatch (src/cqr_matrix_batch.hpp), shared with
// the benchmarks; the suites allocate with the default allocator.

// Compact pack/unpack (matches mkl_?gepack_compact). group g = idx/V, slot
// v = idx%V; element (i,j) of every matrix in a group is one V-wide pack, and
// the packs of a group sit at the leading dimension ldp in the given layout.
// Padded slots (idx >= nm) carry the identity.
//
// Both sides are addressed through the project's views: the dense side through
// MatrixBatch (hence MatrixView), the interleaved side through the kernels'
// own BatchView, whose element is the pack and whose strides are in packs --
// `P(i, j)[v]` is element (i,j) of slot v. The layout is the view's strides,
// so one body serves column-major and row-major, and the group offset is the
// library's group_stride. BatchView is templated on the interleave width, so
// for_vlen turns the runtime V into the compile-time one (2, 4, 8 or 16 --
// the widths the C API accepts). Sharing the kernels' views here is
// deliberate; see "Two views, one idea" in .claude/CLAUDE.md.
//
// A padded slot carries pad_diag on its diagonal, zero elsewhere: the identity
// for matrix batches (so kernels run the padding unmasked), zero for tau
// batches (the identity's reflectors).

// A zeroed compact buffer sized for nm rows x cols matrices at leading
// dimension ldp -- the one place the ng * gstride sizing is written. For a
// buffer the kernel fills (tau, say), use it directly; for one packed from a
// batch, the pack_compact/pack_tau overloads below return it filled.
template <class T>
std::vector<T> compact_buffer(int nm, int rows, int cols, int ldp, int V,
                              bool rowmajor = false)
{
    const int ng = (nm + V - 1) / V;
    return std::vector<T>((std::size_t)ng * group_stride(rowmajor, ldp, rows, cols, V));
}

template <class T>
void pack_compact(const MatrixBatch<T> &Mk, T *p, int ldp, int V, bool rowmajor = false,
                  T pad_diag = T(1))
{
    const int m = Mk.rows(), n = Mk.cols(), nm = Mk.count();
    const int ng = (nm + V - 1) / V;
    const std::size_t gstride = group_stride(rowmajor, ldp, m, n, V);
    const bool width_ok = for_vlen(V, [&](auto vw) {
        constexpr int VV = decltype(vw)::value;
        for (int g = 0; g < ng; ++g) {
            const auto P = make_view<T, VV>(p + (std::size_t)g * gstride, rowmajor, ldp);
            for (int v = 0; v < VV; ++v) {
                const int idx = g * VV + v;
                if (idx < nm) {
                    const auto M = Mk.view(idx);
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < m; ++i)
                            P(i, j)[v] = M(i, j);
                }
                else { /* padded slot */
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < m; ++i)
                            P(i, j)[v] = (i == j) ? pad_diag : T(0);
                }
            }
        }
    });
    assert(width_ok && "interleave width must be 2, 4, 8 or 16");
    (void)width_ok;
}

template <class T>
void unpack_compact(MatrixBatch<T> &Mk, const T *p, int ldp, int V, bool rowmajor = false)
{
    const int m = Mk.rows(), n = Mk.cols(), nm = Mk.count();
    const int ng = (nm + V - 1) / V;
    const std::size_t gstride = group_stride(rowmajor, ldp, m, n, V);
    const bool width_ok = for_vlen(V, [&](auto vw) {
        constexpr int VV = decltype(vw)::value;
        for (int g = 0; g < ng; ++g) {
            const auto P =
                make_const_view<T, VV>(p + (std::size_t)g * gstride, rowmajor, ldp);
            for (int v = 0; v < VV; ++v) {
                const int idx = g * VV + v;
                if (idx >= nm) continue;
                const auto M = Mk.view(idx);
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i)
                        M(i, j) = P(i, j)[v];
            }
        }
    });
    assert(width_ok && "interleave width must be 2, 4, 8 or 16");
    (void)width_ok;
}

// The suites' usual shape: pack into a freshly sized buffer and return it.
template <class T>
std::vector<T> pack_compact(const MatrixBatch<T> &Mk, int ldp, int V,
                            bool rowmajor = false)
{
    auto p = compact_buffer<T>(Mk.count(), Mk.rows(), Mk.cols(), ldp, V, rowmajor);
    pack_compact(Mk, p.data(), ldp, V, rowmajor);
    return p;
}

// A tau batch (k scalars per matrix) is packed as k x 1 matrices whose padded
// slots are all zero -- pack_compact with a zero pad diagonal.
template <class T> void pack_tau(const MatrixBatch<T> &tau, T *tp, int V)
{
    assert(tau.cols() == 1);
    pack_compact(tau, tp, tau.rows(), V, false, T(0));
}

template <class T> std::vector<T> pack_tau(const MatrixBatch<T> &tau, int V)
{
    auto tp = compact_buffer<T>(tau.count(), tau.rows(), 1, tau.rows(), V);
    pack_tau(tau, tp.data(), V);
    return tp;
}

template <class T> void unpack_tau(MatrixBatch<T> &tau, const T *tp, int V)
{
    assert(tau.cols() == 1);
    unpack_compact(tau, tp, tau.rows(), V);
}

// ----------------------- shared checks and epilogues -----------------

// The end-to-end solve gate every solving suite closes with: the worst
// relative forward error of Xhat against the known X, and the worst relative
// residual ||A Xhat - B|| formed densely -- so a solver bug cannot hide behind
// the factorization that produced Xhat.
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

// One row of a C API validation table: the call made, what it returned, and
// what LAPACK-style argument checking must return.
struct ApiCheck {
    const char *what;
    int got, want;
};

// Verdict over a validation table (each failing row spelled out); the return
// value is the table's contribution to the suite's fail count.
inline int report_api_checks(const ApiCheck *t, std::size_t n)
{
    int bad = 0;
    for (std::size_t i = 0; i < n; ++i)
        bad += (t[i].got != t[i].want);
    std::printf("C API validation: %zu checks | %s\n", n, bad ? "FAIL" : "OK");
    for (std::size_t i = 0; i < n; ++i)
        if (t[i].got != t[i].want)
            std::printf("  %-13s got=%d want=%d\n", t[i].what, t[i].got, t[i].want);
    return bad ? 1 : 0;
}
template <std::size_t N> int report_api_checks(const ApiCheck (&t)[N])
{
    return report_api_checks(t, N);
}

// The suites' shared main() ending: report and turn the fail count into the
// process exit code.
inline int finish(int fails)
{
    if (fails) {
        std::printf("\n%d CHECK(S) FAILED\n", fails);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}

} // namespace cqr::test

#endif // TEST_COMPACT_UTIL_HPP
