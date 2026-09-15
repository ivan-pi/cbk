// test_compact_util.hpp
//
// Shared helpers for the compact-format test suites: a seeded RNG, error
// metrics, input generation, precision overloads of the portable C API, and
// Compact pack/unpack. Header-only and library-free; the dense references
// (LAPACKE / CBLAS through lapack<T>) are in test_lapack_util.hpp, which
// includes this, and the MKL-side dispatch in test_mkl_util.hpp.
//
// Assisted-by: Claude:claude-opus-4-8 Claude:claude-fable-5

#ifndef TEST_COMPACT_UTIL_HPP
#define TEST_COMPACT_UTIL_HPP

#include "cbk.h"
#include "cbk_common.hpp"
#include "cbk_matrix_batch.hpp"
#include "cbk_matrix_view.hpp"

#include <cstdio>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <algorithm>

namespace cbk::test {

// The dense strided view every suite addresses its host-side matrices through
// (src/cbk_matrix_view.hpp), and the compact analogue the kernels use, which
// the pack/unpack helpers below address the interleaved side through
// (src/cbk_common.hpp).
using cbk::detail::ConstMatrixView;
using cbk::detail::mat_view;
using cbk::detail::MatrixBatch;
using cbk::detail::MatrixView;

using cbk::detail::for_vlen;
using cbk::detail::group_stride;
using cbk::detail::make_const_view;
using cbk::detail::make_view;

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

// ----------------------- test ratios and the threshold ----------------
// Every numerical check is a dimensionless test ratio in the convention of
// LAPACK's TESTING suites: the error divided by what backward stability
// allows it to be, (order) * (operand norm) * eps, and one threshold decides
// -- THRESH = 30, the value of LAPACK's dtest.in. A correct kernel's ratio is
// O(1) in either precision; a case prints its ratios next to its verdict, so
// a near miss is visible, and a ratio >= THRESH fails (so does NaN: the
// comparison is written to fail it). One knob tunes every suite; there are
// no per-check tolerances. Only structural contracts -- a bit-for-bit
// reproduction, storage the routine must not touch, the identity in the
// padding lanes -- stay exact.
//
// The three forms are LAPACK's own checkers', norms included:
//   test_ratio<T>(err, dim, norm)     err / (max(1, dim) * norm * eps)
//     the factorization form (dqrt01, dpot01, dsyt01, dqrt16): err the
//     1-norm of the residual matrix R - Q^T A, L L^T - A, ... (diff_norm1,
//     orth_norm1); dim the order it accumulates over; norm the operand(s) it
//     is relative to (1 for an orthogonal Q)
//   residual_ratio(anorm, R, X)       max_j ||r_j||_1 / (anorm ||x_j||_1 eps)
//     the solve form (dget02, dpot02, dtrt02): per right-hand side, R the
//     residual B - op(A) X, anorm = ||op(A)||_1 -- no order factor
//   forward_ratio(Xhat, X, rcond)     max_j (||xhat_j - x_j||_inf / ||x_j||_inf) rcond / eps
//     the forward-error form (dget04): per right-hand side, discounted by
//     rcond = 1 / (||A|| ||A^-1||), the amplification a solve is allowed
// A zero solution column with a nonzero error is 1 / eps (a failure), as
// in LAPACK; an empty system is 0.
constexpr double THRESH = 30.0;

template <class T> double test_ratio(double err, int dim, double norm = 1.0)
{
    return err / (std::max(1, dim) * std::max(norm, norm_floor) *
                  (double)std::numeric_limits<T>::epsilon());
}

template <class Rv, class Xv> double residual_ratio(double anorm, Rv R, Xv X)
{
    using T = elem_t<Xv>;
    assert(R.cols == X.cols);
    const double eps = std::numeric_limits<T>::epsilon();
    if (R.rows == 0 || R.cols == 0) return 0;
    if (anorm <= 0) return 1.0 / eps;
    double r = 0;
    for (int j = 0; j < R.cols; ++j) {
        double bnorm = 0, xnorm = 0;
        for (int i = 0; i < R.rows; ++i)
            bnorm += std::abs((double)R(i, j));
        for (int i = 0; i < X.rows; ++i)
            xnorm += std::abs((double)X(i, j));
        r = std::max(r, xnorm <= 0 ? 1.0 : (bnorm / anorm) / xnorm);
    }
    return r / eps;
}

template <class Xv, class Xa> double forward_ratio(Xv Xhat, Xa X, double rcond)
{
    using T = elem_t<Xv>;
    assert(Xhat.rows == X.rows && Xhat.cols == X.cols);
    const double eps = std::numeric_limits<T>::epsilon();
    double r = 0;
    for (int j = 0; j < X.cols; ++j) {
        double diff = 0, xnorm = 0;
        for (int i = 0; i < X.rows; ++i) {
            diff = std::max(diff, std::abs((double)Xhat(i, j) - (double)X(i, j)));
            xnorm = std::max(xnorm, std::abs((double)X(i, j)));
        }
        if (xnorm <= 0) {
            if (diff > 0) r = std::max(r, 1.0);
        }
        else
            r = std::max(r, (diff / xnorm) * rcond);
    }
    return r / eps;
}

inline bool passes(double ratio)
{
    return ratio < THRESH; /* false for NaN */
}

inline const char *verdict(double ratio)
{
    return passes(ratio) ? "OK" : "FAIL";
}

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

// ||A - B||_1 of two same-shaped views, in double: the residual norm LAPACK's
// checkers take of a reconstruction (dqrt01, dpot01, dsyt01) or a comparison.
template <class Av, class Bv> double diff_norm1(Av A, Bv B)
{
    assert(A.rows == B.rows && A.cols == B.cols);
    double mx = 0;
    for (int j = 0; j < A.cols; ++j) {
        double s = 0;
        for (int i = 0; i < A.rows; ++i)
            s += std::abs((double)A(i, j) - (double)B(i, j));
        mx = std::max(mx, s);
    }
    return mx;
}

// ||Q^T Q - I||_1, formed in double: dqrt02's orthogonality residual.
template <class Qv> double orth_norm1(Qv Q)
{
    double mx = 0;
    for (int j = 0; j < Q.cols; ++j) {
        double s = 0;
        for (int i = 0; i < Q.cols; ++i) {
            double d = 0;
            for (int l = 0; l < Q.rows; ++l)
                d += (double)Q(l, i) * Q(l, j);
            s += std::abs(d - (i == j ? 1.0 : 0.0));
        }
        mx = std::max(mx, s);
    }
    return mx;
}

// Deviation of Q's columns from orthonormality, max |(Q^T Q - I)(i,j)| over
// the upper triangle (Q^T Q is symmetric), accumulated in double so the gate
// does not inherit float rounding. The MKL orgqr suite's gate.
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

// ----------------------- portable C API, by scalar type ------------
// compact<T>::geqrf / ormqr / orgqr / potrf / potrs / posv / sytrfnp / sytrsnp
// / sysvnp / trsm / gels forward to the d/s entry points of cbk.h, so
// the templated suites call one name for both precisions; compact<T>::name
// labels their output.

template <class T> struct compact;

// clang-format off
// NOLINTBEGIN(bugprone-macro-parentheses): T is a type name, p a token to paste
#define CBK_TEST_COMPACT_DISPATCH(T, p, label)                                             \
template <> struct compact<T> {                                                            \
    static constexpr const char *name = label;                                             \
    static int geqrf(char lay, int m, int n, T *a, int ld, T *tau, int V, int nm)          \
    { return p##geqrf_compact(lay, m, n, a, ld, tau, V, nm); }                             \
    static int ormqr(char lay, char si, char tr, int m, int n, int k, const T *a,        \
                     int lda, const T *tau, T *c, int ldc, int V, int nm)                  \
    { return p##ormqr_compact(lay, si, tr, m, n, k, a, lda, tau, c, ldc, V, nm); }         \
    static int orgqr(char lay, int m, int n, int k, T *a, int lda, const T *tau,           \
                     int V, int nm)                                                        \
    { return p##orgqr_compact(lay, m, n, k, a, lda, tau, V, nm); }                         \
    static int potrf(char lay, char up, int n, T *a, int ld, int V, int nm)                \
    { return p##potrf_compact(lay, up, n, a, ld, V, nm); }                                 \
    static int potrs(char lay, char up, int n, int nrhs, const T *a, int lda, T *b,        \
                     int ldb, int V, int nm)                                               \
    { return p##potrs_compact(lay, up, n, nrhs, a, lda, b, ldb, V, nm); }                  \
    static int posv(char lay, char up, int n, int nrhs, T *a, int lda, T *b, int ldb,      \
                    int V, int nm)                                                         \
    { return p##posv_compact(lay, up, n, nrhs, a, lda, b, ldb, V, nm); }                   \
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

CBK_TEST_COMPACT_DISPATCH(double, d, "double")
CBK_TEST_COMPACT_DISPATCH(float, s, "float")
#undef CBK_TEST_COMPACT_DISPATCH

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
// stored factor. Shared by the reconstruction checks of both ?sytrfnp suites
// (LAPACK has no unpivoted LDL^T to hand this to; the portable suite checks
// it against a ?trmm-formed reconstruction once).
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

// ----------------------- dense batches and compact packing -----------
// The dense batches are MatrixBatch (src/cbk_matrix_batch.hpp), shared with
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
//
// Never empty: an operand with a zero extent is still an argument the routine
// may require present (Fortran semantics -- ?posv's bp at nrhs = 0, ?gels's A
// at m = 0 -- asserted in cbk.cpp), so the buffer keeps one element, the
// dummy a Fortran caller would pass.
template <class T>
std::vector<T> compact_buffer(int nm, int rows, int cols, int ldp, int V,
                              bool rowmajor = false)
{
    const int ng = (nm + V - 1) / V;
    return std::vector<T>(std::max<std::size_t>(
        1, (std::size_t)ng * group_stride(rowmajor, ldp, rows, cols, V)));
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

// ----------------------- the small-dimension sweep -------------------
// LAPACK's linear-equation suites take M and N from a list that starts
// 0 1 2 3 5 (TESTING/dtest.in) and run every path over the full M x N cross
// product, so the empty operand, the single row or column, and the orders
// just below and above a register block are exercised in every routine, not
// only where someone thought of them. The suites' sweeps take their extents
// from this list and hand each shape to their own run_case: a zero extent is
// the routine's quick return, and the case checks exactly that (info = 0 and
// every error 0). The leading dimension of an empty operand is 1, as LAPACK's
// ld >= max(1, m) requires; the sweeps spell it max(1, extent).
constexpr int small_dims[] = {0, 1, 2, 3, 5};

// ----------------------- shared checks and epilogues -----------------

// One row of a C API validation table: the call made, what it returned, and
// what LAPACK-style argument checking must return.
struct ApiCheck {
    std::string what;
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
            std::printf("  %-16s got=%d want=%d\n", t[i].what.c_str(), t[i].got,
                        t[i].want);
    return bad ? 1 : 0;
}

// The suites table-test two shared entry-point signatures, each validated by
// one helper in src/cbk.cpp, so each is one contract with one table:
//   factor  (layout, uplo, n, ap, ldap, V, nm)            -- ?potrf, ?sytrfnp
//   solve   (layout, uplo, n, nrhs, ap, ldap, bp, ldbp, V, nm)
//                                     -- ?potrs, ?posv, ?sytrsnp, ?sysvnp
// Each append_* runs that contract's rows against one entry point and appends
// them to `t`, labels prefixed with `tag`; pass the entry point itself, e.g.
// append_solve_api_checks(t, "trs", dpotrs_compact). The buffers are owned
// here, seeded with unit diagonals (at the compact, interleaved offsets --
// not a dense 2-D layout) so the valid calls factor/solve sanely.
using factor_api_fn = std::function<int(char, char, int, double *, int, int, int)>;
using solve_api_fn =
    std::function<int(char, char, int, int, double *, int, double *, int, int, int)>;

inline std::vector<double> seeded_compact_diag(int n, int ld, int V)
{
    std::vector<double> ap((std::size_t)ld * n * V, 0);
    for (int v = 0; v < V; ++v)
        for (int i = 0; i < n; ++i)
            ap[((std::size_t)i * ld + i) * V + v] = 1.0;
    return ap;
}

inline void append_factor_api_checks(std::vector<ApiCheck> &t, const std::string &tag,
                                     const factor_api_fn &f)
{
    const int n = 8, V = 4, nm = 4, ld = 8;
    std::vector<double> ap = seeded_compact_diag(n, ld, V);
    auto call = [&](char lay, char up, int n_, int ldap_, int V_, int nm_) {
        return f(lay, up, n_, ap.data(), ldap_, V_, nm_);
    };
    auto name = [&](const char *what) { return tag + " " + what; };
    // clang-format off
    t.insert(t.end(), {
        {name("valid col L"), call('C', 'L', n,  ld,  V, nm),   0},
        {name("valid row U"), call('R', 'U', n,  ld,  V, nm),   0},
        {name("bad layout"),  call('X', 'L', n,  ld,  V, nm),  -1},
        {name("bad uplo"),    call('C', 'X', n,  ld,  V, nm),  -2},
        {name("n<0"),         call('C', 'L', -1, ld,  V, nm),  -3},
        {name("ldap<n"),      call('C', 'L', n,  n-1, V, nm),  -5},
        {name("bad V"),       call('C', 'L', n,  ld,  3, nm),  -6},
        {name("nm<0"),        call('C', 'L', n,  ld,  V, -1),  -7},
        {name("empty n=0"),   call('C', 'L', 0,  1,   V, nm),   0},
        {name("empty nm=0"),  call('C', 'L', n,  ld,  V, 0),    0},
    });
    // clang-format on
}

inline void append_solve_api_checks(std::vector<ApiCheck> &t, const std::string &tag,
                                    const solve_api_fn &f)
{
    const int n = 8, nrhs = 3, V = 4, nm = 4, ld = 8;
    std::vector<double> ap = seeded_compact_diag(n, ld, V);
    std::vector<double> bp((std::size_t)ld * nrhs * V, 0);
    auto call = [&](char lay, char up, int n_, int nrhs_, int ldap_, int ldbp_, int V_,
                    int nm_) {
        return f(lay, up, n_, nrhs_, ap.data(), ldap_, bp.data(), ldbp_, V_, nm_);
    };
    auto name = [&](const char *what) { return tag + " " + what; };
    // clang-format off
    t.insert(t.end(), {
        {name("valid col L"), call('C', 'L', n, nrhs, ld, n,    V, nm),   0},
        {name("valid row U"), call('R', 'U', n, nrhs, ld, nrhs, V, nm),   0},
        {name("bad layout"),  call('X', 'L', n, nrhs, ld, n,    V, nm),  -1},
        {name("bad uplo"),    call('C', 'X', n, nrhs, ld, n,    V, nm),  -2},
        {name("n<0"),         call('C', 'L', -1, nrhs, ld, n,   V, nm),  -3},
        {name("nrhs<0"),      call('C', 'L', n, -1,  ld, n,     V, nm),  -4},
        {name("ldap<n"),      call('C', 'L', n, nrhs, n-1, n,   V, nm),  -6},
        {name("ldbp<n"),      call('C', 'L', n, nrhs, ld, n-1,  V, nm),  -8},
        {name("ldbp<nrhs R"), call('R', 'L', n, nrhs, ld, nrhs-1, V, nm), -8},
        {name("bad V"),       call('C', 'L', n, nrhs, ld, n,    3, nm),  -9},
        {name("nm<0"),        call('C', 'L', n, nrhs, ld, n,    V, -1), -10},
        {name("empty nrhs"),  call('C', 'L', n, 0,   ld, n,     V, nm),   0},
    });
    // clang-format on
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

} // namespace cbk::test

#endif // TEST_COMPACT_UTIL_HPP
