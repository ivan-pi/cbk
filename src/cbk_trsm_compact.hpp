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
 * Algorithm: the BLAS ?trsm substitution run V matrices at a time. Compact
 * format stores element (i,j) of all V contiguously, so it lifts verbatim with
 * double -> V-wide vector, one lane per matrix (no data-dependent branch).
 * There is one kernel, the register-tiled forward substitution L X = alpha B
 * on a lower-triangular view; every side / uplo / trans / layout reaches it by
 * transposing or reversing the views of A and B
 * (docs/cbk_dtrsm_compact_design.md has the details). A non-unit diagonal is
 * solved by multiplying with its reciprocal, not by dividing.
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

/* What a sweep does with the diagonal of its triangular operand: nothing
 * (unit), solve with it (nonunit), or -- the LDL^T solve's second sweep, whose
 * factor carries D on the diagonal of a unit triangle -- scale the right-hand
 * side by its inverse first and then sweep as unit: L X = alpha D^{-1} B. */
enum class TrsmDiag { unit, nonunit, scale };

/* One IB x JB tile of the forward substitution L X = alpha B, L lower
 * triangular: rows i .. i+IB-1 of the columns j .. j+JB-1 of X, the rows above
 * i already solved. The tile is loaded once, accumulates its dot products with
 * the solved rows in registers -- each pack of L feeds JB FMAs, each pack of X
 * feeds IB -- is finished against its own IB x IB corner of L, and is stored
 * once. A diagonal entry is applied as a multiply by its reciprocal, taken
 * first so the divider runs under the accumulation instead of after it; one
 * divide per row of a tile, not one per element. */
template <int IB, int JB, typename T, int V, typename Int>
inline void trsm_tile(TrsmDiag diag, T alpha, ConstBatchView<T, V, Int> L,
                      BatchView<T, V, Int> B, Int i, Int j)
{
    using VT = typename pack<T, V>::type;
    VT inv[IB], x[IB][JB];
    CBK_UNROLL
    for (int r = 0; r < IB; ++r) {
        inv[r] = diag == TrsmDiag::unit ? VT{} : T(1) / L(i + r, i + r);
        CBK_UNROLL
        for (int c = 0; c < JB; ++c) {
            x[r][c] = alpha == T(1) ? B(i + r, j + c) : B(i + r, j + c) * alpha;
            if (diag == TrsmDiag::scale) x[r][c] = x[r][c] * inv[r];
        }
    }
    for (Int k = 0; k < i; ++k) {
        VT l[IB];
        CBK_UNROLL
        for (int r = 0; r < IB; ++r)
            l[r] = L(i + r, k);
        CBK_UNROLL
        for (int c = 0; c < JB; ++c) {
            const VT b = B(k, j + c);
            CBK_UNROLL
            for (int r = 0; r < IB; ++r)
                x[r][c] -= l[r] * b;
        }
    }
    CBK_UNROLL
    for (int r = 0; r < IB; ++r) {
        CBK_UNROLL
        for (int q = 0; q < r; ++q) {
            const VT l = L(i + r, i + q);
            CBK_UNROLL
            for (int c = 0; c < JB; ++c)
                x[r][c] -= l * x[q][c];
        }
        CBK_UNROLL
        for (int c = 0; c < JB; ++c) {
            if (diag == TrsmDiag::nonunit) x[r][c] = x[r][c] * inv[r];
            B(i + r, j + c) = x[r][c];
        }
    }
}

/* The tile: IB * JB accumulators, IB packs of L and one of X live in the
 * inner loop, so 4 x 6 with 32 registers and 3 x 4 with 16 (measured: a tile
 * that does not fit runs slower than a smaller one that does). */
constexpr int trsm_ib = simd_registers >= 32 ? 4 : 3;
constexpr int trsm_jb = simd_registers >= 32 ? 6 : 4;

/* The JB columns j .. j+JB-1 of X, top to bottom in trsm_ib-row tiles, the
 * leftover rows as one shorter tile. */
template <int JB, typename T, int V, typename Int>
inline void trsm_cols(TrsmDiag diag, Int m, T alpha, ConstBatchView<T, V, Int> L,
                      BatchView<T, V, Int> B, Int j)
{
    Int i = 0;
    for (; i + trsm_ib <= m; i += trsm_ib)
        trsm_tile<trsm_ib, JB, T, V, Int>(diag, alpha, L, B, i, j);
    for_remainder<trsm_ib>(m - i, [&](auto ib) {
        trsm_tile<ib.value, JB, T, V, Int>(diag, alpha, L, B, i, j);
    });
}

/* One group, forward substitution L X = alpha B through the views: L lower
 * triangular of order m, B m x n, alpha nonzero. Column blocks outermost, so
 * a block's columns stay cache-resident while L streams past them once:
 * trsm_jb-column blocks, then the leftover columns as one narrower block. */
template <typename T, int V, typename Int>
void trsm_lower_group(TrsmDiag diag, Int m, Int n, T alpha, ConstBatchView<T, V, Int> L,
                      BatchView<T, V, Int> B)
{
    static_assert(std::is_floating_point_v<T>,
                  "trsm_compact is defined for real float/double");
    assert(L.si && L.sj && B.si && B.sj);

    Int j = 0;
    for (; j + trsm_jb <= n; j += trsm_jb)
        trsm_cols<trsm_jb, T, V, Int>(diag, m, alpha, L, B, j);
    for_remainder<trsm_jb>(
        n - j, [&](auto jb) { trsm_cols<jb.value, T, V, Int>(diag, m, alpha, L, B, j); });
}

/* One group, any side / uplo / trans / layout, alpha != 0: every case is the
 * forward substitution above on the right views. side='R' is the left solve
 * of the transposed system, op(A)^T X^T = alpha B^T; the operator M of that
 * left solve is A or A^T by a stride swap; and an upper-triangular M is lower
 * triangular with its rows and columns reversed, which turns the back
 * substitution into the same forward sweep (X's columns come out reversed
 * too, harmlessly: they are independent). The per-group entry point the
 * fused solves (gels, and through the pointer form below, the Cholesky and
 * LDL^T solves) compose their sweeps from. */
template <typename T, int V, typename Int = int>
inline void trsm_compact_group(bool left, bool upper, bool tran, TrsmDiag diag, Int m,
                               Int n, T alpha, ConstBatchView<T, V, Int> A,
                               BatchView<T, V, Int> B)
{
    if (m == 0 || n == 0) return;
    if (!left) {
        B = B.transposed();
        const Int t = m;
        m = n;
        n = t;
    }
    const bool flip = (tran == left); /* M = A^T */
    if (flip) A = A.transposed();
    if (upper != flip) { /* M upper */
        A = A.reversed(m, m);
        B = B.reversed(m, n);
    }
    trsm_lower_group<T, V, Int>(diag, m, n, alpha, A, B);
}

/* The same on packed pointers in a layout. */
template <typename T, int V, typename Int = int>
inline void trsm_compact_group(bool left, bool upper, bool rowmajor, bool tran,
                               TrsmDiag diag, Int m, Int n, T alpha, const T *a, Int ldap,
                               T *b, Int ldbp)
{
    trsm_compact_group<T, V, Int>(left, upper, tran, diag, m, n, alpha,
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
    assert(ldap >= (left ? m : n) && ldbp >= (rowmajor ? n : m));

    /* A is the order-s triangular factor: s = m (left) or n (right). */
    const Int s = left ? m : n;

    /* A is s x s, B is m x n. */
    const std::size_t str_a = group_stride(rowmajor, ldap, s, s, V);
    const std::size_t str_b = group_stride(rowmajor, ldbp, m, n, V);

    /* alpha == 0 is the BLAS ?trsm fast path: B := 0 with A untouched. Handle it
     * once here -- the group kernel then assumes alpha != 0. */
    if (alpha == T(0)) {
        zero_compact<T, V, Int>(rowmajor, m, n, bp, ldbp, nm);
        return;
    }

    const TrsmDiag diag = unit ? TrsmDiag::unit : TrsmDiag::nonunit;
    for_each_group<V>(
        nm,
        [&](Int g) {
            trsm_compact_group<T, V, Int>(left, upper, rowmajor, tran, diag, m, n, alpha,
                                          ap + (std::size_t)g * str_a, ldap,
                                          bp + (std::size_t)g * str_b, ldbp);
        },
        (double)s * s * (left ? n : m) * V /* ~substitution flops per group */);
}

} /* namespace cbk::detail */

CBK_KERNEL_END

#endif /* CBK_TRSM_COMPACT_HPP */
