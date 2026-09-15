/* cbk_trsm_compact.hpp
 *
 * Compact (interleaved-batch) triangular solve with multiple right-hand sides,
 * templated on scalar type T and interleave width V:
 *
 *     op(A) X = alpha B   (side='L')   or   X op(A) = alpha B   (side='R'),
 *     op(A) = A ('N') or A^T ('T'),
 *
 * A is upper/lower, unit/non-unit triangular; B is overwritten by X. The compact
 * analogue of BLAS ?trsm and a portable alternative to mkl_?trsm_compact -- the
 * solve that closes the batched QR:
 *
 *     cbk_dgeqrf_compact(A -> H, tau);          // A = Q R
 *     cbk_dormqr_compact('L','T', H, tau, B);   // B := Q^T B
 *     cbk_dtrsm_compact ('L','U','N','N', R, B) // B := R^{-1} Q^T B = X
 *
 * Algorithm: the scalar BLAS ?trsm substitution run V matrices at a time. Compact
 * format stores element (i,j) of all V contiguously, so it lifts verbatim with
 * double -> V-wide vector, one lane per matrix (no data-dependent branch). The
 * tuned side='L', column-major path is templated on the RHS block width and on
 * uplo/trans/diag; from trsm_block_min up it sweeps the pivots in blocks, with
 * each block's effect on the rows still to come applied as one register-tiled
 * rank-NB update instead of a chain of rank-1 passes. The other side/layout
 * combinations go through one strided kernel over BatchViews
 * (docs/cbk_dtrsm_compact_design.md has the details).
 *
 * Compact storage (matches mkl_?gepack_compact); group g = idx/V, slot v = idx%V,
 * A the order-s (s = m left / n right) triangular batch, B the m x n batch:
 *     A_v(i,j) = ap[g*ldap*s*V + (j*ldap+i)*V + v]   (column-major)
 *     B_v(i,j) = bp[g*ldbp*n*V + (j*ldbp+i)*V + v]
 * Row-major swaps the in-matrix roles to i*ld + j.
 *
 * Assisted-by: Claude:claude-opus-4-8 Claude:claude-fable-5
 */

#ifndef CBK_TRSM_COMPACT_HPP
#define CBK_TRSM_COMPACT_HPP

#include "cbk_common.hpp"

#include <cstddef>
#include <cassert>
#include <type_traits>

CBK_KERNEL_BEGIN

namespace cbk::detail {

/* Single-column column-oriented (gaxpy) solve, side='L', column-major, op(A)=A:
 * reads A down columns (contiguous), for the 1-column tail where a strided
 * single-column row-dot would prefetch poorly. bj is the column, scaled first. */
template <bool UPPER, bool UNIT, typename T, int V, typename Int>
inline void trsm_axpy_col(Int m, const typename pack<T, V>::type *A, Int ldap,
                          typename pack<T, V>::type *bj,
                          const typename pack<T, V>::type &va)
{
    using VT = typename pack<T, V>::type;
    for (Int i = 0; i < m; ++i)
        bj[i] = bj[i] * va;
    for (Int t = 0; t < m; ++t) {
        const Int kk = UPPER ? m - 1 - t : t;
        const VT *ak = A + kk * ldap;
        /* the reciprocal of trsm_dot_block, so a column's answer does not
         * depend on which block of the 4/2/1 split it landed in */
        const VT xk = UNIT ? bj[kk] : bj[kk] * (T(1) / ak[kk]);
        if (!UNIT) bj[kk] = xk;
        const Int lo = UPPER ? 0 : kk + 1;
        const Int hi = UPPER ? kk : m;
        for (Int i = lo; i < hi; ++i)
            bj[i] -= ak[i] * xk;
    }
}

/* Row-dot solve of a fixed JB-column block, side='L', column-major. JB and the
 * uplo/trans/diag config are compile-time, so the JB accumulators land in
 * registers and the c-loops unroll. B points at the first of the JB adjacent RHS
 * columns (column c at B + c*ldbp); alpha is folded into the first load. */
template <int JB, bool UPPER, bool TRAN, bool UNIT, typename T, int V, typename Int>
inline void trsm_dot_block(Int m, const typename pack<T, V>::type *A, Int ldap,
                           typename pack<T, V>::type *B, Int ldbp,
                           const typename pack<T, V>::type &va)
{
    using VT = typename pack<T, V>::type;
    /* back-substitute (sweep rows high -> low) when op(A) is upper-triangular:
     * A upper & no-trans, or A lower & trans (its transpose is upper). */
    constexpr bool back = (UPPER != TRAN);
    for (Int t = 0; t < m; ++t) {
        const Int i = back ? m - 1 - t : t;
        VT w[JB];
        for (int c = 0; c < JB; ++c)
            w[c] = B[c * ldbp + i] * va;
        const Int lo = back ? i + 1 : 0;
        const Int hi = back ? m : i;
        for (Int l = lo; l < hi; ++l) {
            /* TRAN reads A(l,i) = A[i*ldap+l] (down column i, contiguous);
             * !TRAN reads A(i,l) = A[l*ldap+i] (across row i, stride ldap). */
            const VT av = TRAN ? A[i * ldap + l] : A[l * ldap + i];
            for (int c = 0; c < JB; ++c)
                w[c] -= av * B[c * ldbp + l];
        }
        if (!UNIT) {
            /* One divide per pivot row, not one per (row, column): the divider
             * port is not pipelined, so a divide per RHS column dominated the
             * small orders outright (measured: at n = 8 the divides were the
             * whole kernel at 16 right-hand sides). It costs one extra
             * rounding per solved entry; the suites' test ratios cover it. */
            const VT r = T(1) / A[i * ldap + i];
            for (int c = 0; c < JB; ++c)
                w[c] = w[c] * r;
        }
        for (int c = 0; c < JB; ++c)
            B[c * ldbp + i] = w[c];
    }
}

/* Pivot-block width of the blocked sweep below, and the order from which that
 * sweep is used at all. Under the block width the substitution is the plain
 * row-dot above; above it, a block of pivots is solved and its effect on the
 * rows still to come is applied as one register-tiled update, which is what
 * turns the sweep from a chain of rank-1 passes into a rank-NB one. Below
 * trsm_block_min the blocking costs more in per-block overhead than the
 * tiling saves (measured under both gcc and clang, which agree on where the
 * crossing is). */
#ifndef CBK_TRSM_NB
#define CBK_TRSM_NB 32
#endif
constexpr int trsm_nb = CBK_TRSM_NB;
#ifndef CBK_TRSM_BLOCK_MIN
#define CBK_TRSM_BLOCK_MIN 40
#endif
constexpr int trsm_block_min = CBK_TRSM_BLOCK_MIN;

/* One IB x JB tile of B -- rows i0.., RHS columns j0.. -- updated by the
 * pivots [k0,k1) already solved:  B(i,j) -= sum_l op(A)(i,l) B(l,j).
 * The IB*JB accumulators live in registers, so the IB values of op(A) and JB
 * of B loaded per l feed IB*JB fused multiply-adds -- the register tiling the
 * blocked potrf uses for its trailing update (sytrfnp_syrk_tile), with the
 * same JB-loop shape the compiler unrolls rather than hand-expanded copies. */
template <int IB, int JB, bool TRAN, typename T, int V, typename Int>
inline void trsm_update_tile(const typename pack<T, V>::type *A, Int ldap,
                             typename pack<T, V>::type *B, Int ldbp, Int i0, Int j0,
                             Int k0, Int k1)
{
    using VT = typename pack<T, V>::type;
    VT acc[IB][JB];
    for (int r = 0; r < IB; ++r)
        for (int c = 0; c < JB; ++c)
            acc[r][c] = B[(j0 + c) * ldbp + i0 + r];
    for (Int l = k0; l < k1; ++l) {
        /* TRAN reads op(A)(i,l) = A(l,i) = A[i*ldap+l], one per tile row down
         * column i; !TRAN reads A(i,l) = A[l*ldap+i], IB contiguous packs of
         * column l. */
        VT u[IB], w[JB];
        for (int r = 0; r < IB; ++r)
            u[r] = TRAN ? A[(i0 + r) * ldap + l] : A[l * ldap + i0 + r];
        for (int c = 0; c < JB; ++c)
            w[c] = B[(j0 + c) * ldbp + l];
        for (int r = 0; r < IB; ++r)
            for (int c = 0; c < JB; ++c)
                acc[r][c] -= u[r] * w[c];
    }
    for (int r = 0; r < IB; ++r)
        for (int c = 0; c < JB; ++c)
            B[(j0 + c) * ldbp + i0 + r] = acc[r][c];
}

/* The pivots [k0,k1) applied to every row in [r0,r1) and every RHS column, in
 * 2x4 tiles with 2x1 / 1x4 / 1x1 edges.
 *
 * 2x4 holds 2*4 accumulators plus 2 + 4 operands = 14 vectors live, which fits
 * the *16* architectural vector registers of SSE and AVX as well as AVX-512's
 * 32 -- and the kernel is instantiated at all three widths. A 4x4 tile needs
 * 24 and is the faster shape where 32 registers exist, but it spills where
 * only 16 do: built for an AVX2 target it lost about 40% at V = 4, 16
 * right-hand sides, order 128, against this shape. One tile for every
 * generation is worth more than the last few percent on one of them. */
template <bool TRAN, typename T, int V, typename Int>
void trsm_update(Int r0, Int r1, Int n, const typename pack<T, V>::type *A, Int ldap,
                 typename pack<T, V>::type *B, Int ldbp, Int k0, Int k1)
{
    Int i = r0;
    for (; i + 2 <= r1; i += 2) {
        Int j = 0;
        for (; j + 4 <= n; j += 4)
            trsm_update_tile<2, 4, TRAN, T, V, Int>(A, ldap, B, ldbp, i, j, k0, k1);
        for (; j < n; ++j)
            trsm_update_tile<2, 1, TRAN, T, V, Int>(A, ldap, B, ldbp, i, j, k0, k1);
    }
    for (; i < r1; ++i) {
        Int j = 0;
        for (; j + 4 <= n; j += 4)
            trsm_update_tile<1, 4, TRAN, T, V, Int>(A, ldap, B, ldbp, i, j, k0, k1);
        for (; j < n; ++j)
            trsm_update_tile<1, 1, TRAN, T, V, Int>(A, ldap, B, ldbp, i, j, k0, k1);
    }
}

/* The blocked sweep: pivots in blocks of trsm_nb, each block's own triangle
 * solved by the row-dot above (every RHS column, in the same 4/2/1 split) and
 * its effect on the rows still to come applied by one trsm_update.
 *
 * The sweep runs in *pivot* order p = 0 .. m-1; the row it touches is
 * i = m-1-p when op(A) is upper-triangular (back-substitution) and p when it
 * is lower. Either way a block of pivots is a contiguous run of rows, so one
 * index (d0) and a length place both the diagonal block and the rows left.
 *
 * alpha is already folded into B by the caller, so the sweep runs at alpha = 1:
 * each entry of B is read as the right-hand side exactly once before its
 * solution overwrites it, which makes pre-scaling bit-identical to scaling at
 * the point of use. */
template <bool UPPER, bool TRAN, bool UNIT, typename T, int V, typename Int>
void trsm_left_blocked(Int m, Int n, const typename pack<T, V>::type *A, Int ldap,
                       typename pack<T, V>::type *B, Int ldbp)
{
    using VT = typename pack<T, V>::type;
    constexpr bool back = (UPPER != TRAN);
    VT one;
    broadcast<T, V>(one, T(1));
    for (Int p0 = 0; p0 < m; p0 += trsm_nb) {
        const Int p1 = (p0 + trsm_nb < m) ? p0 + trsm_nb : m;
        const Int db = p1 - p0;
        const Int d0 = back ? m - p1 : p0;
        const VT *Ad = A + d0 * ldap + d0;
        VT *Bd = B + d0;
        Int j = 0;
        for (; j + 4 <= n; j += 4)
            trsm_dot_block<4, UPPER, TRAN, UNIT, T, V, Int>(db, Ad, ldap, Bd + j * ldbp,
                                                            ldbp, one);
        if (n - j >= 2) {
            trsm_dot_block<2, UPPER, TRAN, UNIT, T, V, Int>(db, Ad, ldap, Bd + j * ldbp,
                                                            ldbp, one);
            j += 2;
        }
        if (n - j >= 1)
            trsm_dot_block<1, UPPER, TRAN, UNIT, T, V, Int>(db, Ad, ldap, Bd + j * ldbp,
                                                            ldbp, one);
        const Int r0 = back ? Int(0) : d0 + db;
        const Int r1 = back ? d0 : m;
        trsm_update<TRAN, T, V, Int>(r0, r1, n, A, ldap, B, ldbp, d0, d0 + db);
    }
}

/* One group, side='L', column-major, fully specialized on uplo/trans/diag:
 * the RHS columns are swept in 4/2/1 blocks so the 1-3 leftover columns still
 * reuse each A load (2- and 1-wide tails), instead of a one-column-at-a-time
 * remainder. The driver handles alpha = 0 (B := 0); alpha is nonzero here. */
template <bool UPPER, bool TRAN, bool UNIT, typename T, int V, typename Int>
void trsm_left_dot_tb(Int m, Int n, T alpha, const T *a_, Int ldap, T *b_, Int ldbp)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point_v<T>,
                  "trsm_compact is defined for real float/double");
    /* n = 1 never steps to a second RHS column, so ldbp is unused then: the
     * routing in trsm_compact_group reaches here with ldbp = 1 for a one-column
     * row-major B, whose unit row stride is a valid column-major column. */
    assert(ldap >= m && (n <= 1 || ldbp >= m));

    const VT *A = reinterpret_cast<const VT *>(a_);
    VT *B = reinterpret_cast<VT *>(b_);

    VT va;
    broadcast<T, V>(va, alpha);

    /* The blocked sweep pays when the tile has something to amortize over: a
     * second RHS column, or the transposed sweep, whose unblocked form reduces
     * each row into a single accumulator and runs at FMA latency. A single
     * column of op(A) = A is already the contiguous column-axpy below, and
     * blocking it only adds passes (measured, both compilers). */
    if (m >= trsm_block_min && (TRAN || n >= 2)) {
        if (alpha != T(1))
            for (Int j = 0; j < n; ++j)
                for (Int i = 0; i < m; ++i)
                    B[j * ldbp + i] = B[j * ldbp + i] * va;
        trsm_left_blocked<UPPER, TRAN, UNIT, T, V, Int>(m, n, A, ldap, B, ldbp);
        return;
    }

    Int j = 0;
    for (; j + 4 <= n; j += 4)
        trsm_dot_block<4, UPPER, TRAN, UNIT, T, V, Int>(m, A, ldap, B + j * ldbp, ldbp,
                                                        va);
    if (n - j >= 2) {
        trsm_dot_block<2, UPPER, TRAN, UNIT, T, V, Int>(m, A, ldap, B + j * ldbp, ldbp,
                                                        va);
        j += 2;
    }
    if (n - j >= 1) {
        /* The single leftover column: for op(A)=A the row-dot would stream A
         * strided with no reuse, so use the contiguous column-axpy instead;
         * op(A)=A^T reads A down a column already, so the dot 1-block is fine. */
        if constexpr (!TRAN)
            trsm_axpy_col<UPPER, UNIT, T, V, Int>(m, A, ldap, B + j * ldbp, va);
        else
            trsm_dot_block<1, UPPER, TRAN, UNIT, T, V, Int>(m, A, ldap, B + j * ldbp,
                                                            ldbp, va);
    }
}

/* Runtime (uplo, trans, diag) -> the compile-time-specialized driver. */
template <bool UPPER, bool TRAN, typename T, int V, typename Int>
inline void trsm_left_dot_u(bool unit, Int m, Int n, T alpha, const T *a, Int ldap, T *b,
                            Int ldbp)
{
    if (unit)
        trsm_left_dot_tb<UPPER, TRAN, true, T, V, Int>(m, n, alpha, a, ldap, b, ldbp);
    else
        trsm_left_dot_tb<UPPER, TRAN, false, T, V, Int>(m, n, alpha, a, ldap, b, ldbp);
}
template <bool UPPER, typename T, int V, typename Int>
inline void trsm_left_dot_t(bool tran, bool unit, Int m, Int n, T alpha, const T *a,
                            Int ldap, T *b, Int ldbp)
{
    if (tran)
        trsm_left_dot_u<UPPER, true, T, V, Int>(unit, m, n, alpha, a, ldap, b, ldbp);
    else
        trsm_left_dot_u<UPPER, false, T, V, Int>(unit, m, n, alpha, a, ldap, b, ldbp);
}
template <typename T, int V, typename Int>
inline void trsm_left_dot(bool upper, bool tran, bool unit, Int m, Int n, T alpha,
                          const T *a, Int ldap, T *b, Int ldbp)
{
    if (upper)
        trsm_left_dot_t<true, T, V, Int>(tran, unit, m, n, alpha, a, ldap, b, ldbp);
    else
        trsm_left_dot_t<false, T, V, Int>(tran, unit, m, n, alpha, a, ldap, b, ldbp);
}

/* One group, fully general: any side / layout via BatchView strides. Same
 * substitution as the tuned path; only the addressing changes (it lives in the
 * two BatchViews). side='L' sweeps a row of X at a time, side='R' a column. */
template <typename T, int V, typename Int = int>
void trsm_compact_group_strided(bool left, bool upper, bool tran, bool unit, Int m, Int n,
                                T alpha, ConstBatchView<T, V, Int> A,
                                BatchView<T, V, Int> B)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point_v<T>,
                  "trsm_compact is defined for real float/double");

    assert(A.si && A.sj && B.si && B.sj);

    VT va;
    broadcast<T, V>(va, alpha);

    if (left) {
        /* solve op(A) X = alpha B column by column; A is m x m */
        const bool back = (upper != tran);
        for (Int j = 0; j < n; ++j)
            for (Int t = 0; t < m; ++t) {
                const Int i = back ? m - 1 - t : t;
                VT w = B(i, j) * va;
                const Int lo = back ? i + 1 : 0;
                const Int hi = back ? m : i;
                for (Int l = lo; l < hi; ++l)
                    w -= (tran ? A(l, i) : A(i, l)) * B(l, j);
                B(i, j) = unit ? w : w / A(i, i);
            }
    }
    else {
        /* solve X op(A) = alpha B, one column of X at a time; A is n x n */
        const bool fwd = (upper != tran);
        for (Int t = 0; t < n; ++t) {
            const Int j = fwd ? t : n - 1 - t;
            for (Int i = 0; i < m; ++i)
                B(i, j) = B(i, j) * va; /* scale this column of X */
            const Int lo = fwd ? 0 : j + 1;
            const Int hi = fwd ? j : n;
            for (Int l = lo; l < hi; ++l) {
                const VT c = tran ? A(j, l) : A(l, j);
                for (Int i = 0; i < m; ++i)
                    B(i, j) -= c * B(i, l);
            }
            if (!unit) {
                const VT d = A(j, j);
                for (Int i = 0; i < m; ++i)
                    B(i, j) = B(i, j) / d;
            }
        }
    }
}

/* One group, any side / layout, alpha != 0, through the views. Routes to the
 * tuned trsm_left_dot when it applies -- side='L' with B's rows contiguous
 * (unit row stride) and A either column-major (unit row stride) or stored as
 * its transpose (unit column stride: A is then M^T for the column-major M of
 * leading dimension A.si, so uplo and transa flip -- the case of a QR factor
 * reached through a transposed view, as gels's LQ path does) -- and to the
 * strided kernel otherwise. One branch per group, outside every kernel loop.
 * The per-group entry point the fused solves (gels, and through the pointer
 * form below, the Cholesky and LDL^T solves) compose their sweeps from. */
template <typename T, int V, typename Int = int>
inline void trsm_compact_group(bool left, bool upper, bool tran, bool unit, Int m, Int n,
                               T alpha, ConstBatchView<T, V, Int> A,
                               BatchView<T, V, Int> B)
{
    if (left && B.si == 1 && A.si == 1)
        trsm_left_dot<T, V, Int>(upper, tran, unit, m, n, alpha,
                                 reinterpret_cast<const T *>(A.data), A.sj,
                                 reinterpret_cast<T *>(B.data), B.sj);
    else if (left && B.si == 1 && A.sj == 1)
        trsm_left_dot<T, V, Int>(!upper, !tran, unit, m, n, alpha,
                                 reinterpret_cast<const T *>(A.data), A.si,
                                 reinterpret_cast<T *>(B.data), B.sj);
    else
        trsm_compact_group_strided<T, V, Int>(left, upper, tran, unit, m, n, alpha, A, B);
}

/* The same on packed pointers in a layout: side='L' column-major reaches the
 * tuned path, the other three side/layout combinations the strided kernel. */
template <typename T, int V, typename Int = int>
inline void trsm_compact_group(bool left, bool upper, bool rowmajor, bool tran, bool unit,
                               Int m, Int n, T alpha, const T *a, Int ldap, T *b,
                               Int ldbp)
{
    trsm_compact_group<T, V, Int>(left, upper, tran, unit, m, n, alpha,
                                  make_const_view<T, V, Int>(a, rowmajor, ldap),
                                  make_view<T, V, Int>(b, rowmajor, ldbp));
}

/* All groups (nm matrices). A padded partial last group is processed too, which
 * is harmless: padded slots are identity triangular factors (unit diagonal), so
 * the diagonal divide never hits zero and their X = alpha B is never read back. */
template <typename T, int V, typename Int = int>
void trsm_compact(bool left, bool upper, bool rowmajor, bool tran, bool unit, Int m,
                  Int n, T alpha, const T *ap, Int ldap, T *bp, Int ldbp, Int nm)
{
    assert(nm >= 1 && m >= 0 && n >= 0);

    /* A is the order-s triangular factor: s = m (left) or n (right). */
    const Int s = left ? m : n;

    /* A is s x s, B is m x n. */
    const std::size_t str_a = group_stride(rowmajor, ldap, s, s, V);
    const std::size_t str_b = group_stride(rowmajor, ldbp, m, n, V);

    /* alpha == 0 is the BLAS ?trsm fast path: B := 0 with A untouched. Handle it
     * once here -- both group kernels then assume alpha != 0. */
    if (alpha == T(0)) {
        zero_compact<T, V, Int>(rowmajor, m, n, bp, ldbp, nm);
        return;
    }

    for_each_group<V>(
        nm,
        [&](Int g) {
            trsm_compact_group<T, V, Int>(left, upper, rowmajor, tran, unit, m, n, alpha,
                                          ap + (std::size_t)g * str_a, ldap,
                                          bp + (std::size_t)g * str_b, ldbp);
        },
        (double)s * s * (left ? n : m) * V /* ~substitution flops per group */);
}

} /* namespace cbk::detail */

CBK_KERNEL_END

#endif /* CBK_TRSM_COMPACT_HPP */
