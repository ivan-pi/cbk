/* cbk_sytrfnp_compact.hpp
 *
 * Compact (interleaved-batch) LDL^T factorization of symmetric matrices,
 * without pivoting ("np"), templated on scalar type T and interleave width V:
 *
 *     A = L D L^T  (uplo lower),   A = U^T D U  (uplo upper),
 *
 * with L (U) unit lower (upper) triangular and D diagonal, run V matrices at a
 * time. On exit D sits on the diagonal and the strict off-diagonal of the named
 * triangle holds L (U); the unit diagonal is implied, not stored. MKL ships no
 * compact sytrf of any kind; the "np" suffix follows its unpivoted compact LU,
 * mkl_?getrfnp_compact.
 *
 * The square-root-free sibling of cbk_potrf_compact.hpp: symmetric *indefinite*
 * matrices factor too (negative pivots are fine -- there is no sqrt), which is
 * the point of LDL^T over Cholesky. cbk_sytrsnp_compact.hpp solves from the
 * factor and cbk_sysvnp_compact.hpp fuses the two per group.
 *
 * Algorithm: the right-looking sweep, potf2 with the sqrt pivot replaced by a
 * reciprocal, lifted double -> V-wide vector, one lane per matrix, blocked
 * recursively exactly as cbk_potrf_compact.hpp: the columns are split in
 * halves at NB-column panel boundaries down to panels of at most NB pivots
 * (lower):
 *
 *     factor(j0, j1):
 *         if j1 - j0 <= NB: the sweep on the panel, each pivot's rank-1 update
 *                           confined to the panel's remaining columns
 *         else: factor(j0, mid); A(mid:n, mid:j1) -= L(mid:n, j0:mid) D
 *               L(mid:j1, j0:mid)^T (register-tiled); factor(mid, j1)
 *
 * and the panel's sweep, per pivot column j:
 *
 *     d = A(j,j);  invd = 1/d                      (pivot stays as D(j,j))
 *     A(i,jj)-= A(i,j)*(A(jj,j)*invd) for j < jj < j1, i >= jj  (rank-1, on the
 *                                                     unscaled column, in-panel)
 *     A(i,j) *= invd                for i > j            (scale -> L(i,j))
 *
 * Like Cholesky the math has no data-dependent branch, so no lane mask is
 * needed. The tiled rank-K update is the rank-1 update's arithmetic with the
 * trailing element accumulated in registers over the K pivots, in pivot order;
 * the tile, the column block and the trailing update are shared with potrf
 * (WithD = false there: the weights are the scaled column, D = I).
 *
 * Only the lower trapezoid of the *view* is touched: the kernel gets A for uplo
 * lower and A^T for upper (A symmetric, so the lower factorization of A^T lands
 * U = L(A^T)^T in the upper storage with the same D), and the strictly-opposite
 * triangle passes through untouched. Note the upper convention is therefore the
 * transpose dual A = U^T D U (matching potrf's A = U^T U), NOT LAPACK ?sytrf's
 * A = U D U^T (design section 6.3). Column-major lower and row-major upper are
 * the contiguous cases (unit row stride); the other two are strided.
 *
 * Pivots are the *updated* (Schur-complement) diagonal entries, not the input
 * ones: a zero on the input diagonal is harmless unless a pivot itself is zero.
 * A zero pivot -- a singular leading principal minor -- gets Inf/NaN in its
 * lane's factor; there is no info = j early exit and no pivoting (both are
 * per-lane branching that does not vectorize; design section 6.2).
 *
 * Compact storage (matches mkl_?gepack_compact); group g = idx/V, slot v = idx%V:
 *     A_v(i,j) = ap[ g*ldap*n*V + (j*ldap + i)*V + v ]   (column-major)
 * Row-major swaps the in-matrix roles (i*ldap + j); the group stride is the
 * same either way (A is n x n).
 *
 * Assisted-by: Claude:claude-fable-5
 */

#ifndef CBK_SYTRFNP_COMPACT_HPP
#define CBK_SYTRFNP_COMPACT_HPP

#include "cbk_common.hpp"

#include <cstddef>
#include <cassert>
#include <type_traits>

CBK_KERNEL_BEGIN

namespace cbk::detail {

/* Leaf panel width: the pivot columns factored by the unblocked sweep between
 * tiled updates (potrf's potrf_nb, the same trade-off). */
#ifndef CBK_SYTRFNP_NB
#define CBK_SYTRFNP_NB 8
#endif
constexpr int sytrfnp_nb = CBK_SYTRFNP_NB;

/* Symmetric rank-1 trailing update of the JB columns jj .. jj+JB-1 by pivot
 * column j: A(i,c) -= A(i,j) * (A(c,j) * w) for i >= c, with the weight w
 * still in place at A(j,j) (re-read here rather than passed, so no pack
 * crosses a call boundary by reference -- see the alignment note in
 * cbk_common.hpp) folded into the per-column weight w[c] = A(c,j) * w, one
 * extra multiply per block column. In the panel sweep the column is unscaled
 * and w = 1/d, the pivot's reciprocal; potrf's panel calls it the same way
 * with 1/a. The near-diagonal corner fills in triangularly (column c touches
 * rows i >= c); below it all JB columns take the same A(i,j), loaded once. JB
 * is compile-time so w[] stays in registers. */
template <int JB, typename T, int V, typename Int>
inline void sytrfnp_update_block(Int j, Int n, const BatchView<T, V, Int> &A, Int jj)
{
    using VT = typename pack<T, V>::type;
    const VT d = A(j, j);
    VT w[JB];
    for (int c = 0; c < JB; ++c)
        w[c] = A(jj + c, j) * d;
    for (int c = 0; c < JB; ++c)
        for (int r = c; r < JB; ++r)
            A(jj + r, jj + c) -= A(jj + r, j) * w[c];
    for (Int i = jj + JB; i < n; ++i) {
        const VT av = A(i, j);
        for (int c = 0; c < JB; ++c)
            A(i, jj + c) -= av * w[c];
    }
}

/* Rank-K update of one IB x JB tile of the trailing matrix, rows i .. i+IB-1
 * and columns jj .. jj+JB-1, by the columns k0 .. k1-1:
 *     A(i+r, jj+c) -= sum_k A(i+r,k) * D(k) * A(jj+c,k)
 * accumulated in registers over k, in pivot order, so the tile is loaded and
 * stored once and each pack of the k columns feeds IB (or JB) FMAs. WithD:
 * the pivot D(k) = A(k,k) is loaded once per k and folded into the JB column
 * weights (potrf's columns are already scaled: D = I, no load). Diag: the tile
 * sits on the diagonal (i == jj) and only its lower triangle r >= c exists --
 * the loop bounds are compile-time, so the guard folds away -- and the row
 * packs are the unscaled column packs, already loaded. */
/* CBK_UNROLL before each loop over the tile asks for it to be fully unrolled
 * (GCC and clang both take the GCC spelling; 16 covers the 4 x 4 tile), which
 * keeps acc[][] in registers under GCC, whose scalar replacement runs before
 * it unrolls these loops and would otherwise stage the tile through the stack
 * on entry and exit. Undefined after the tile: a kernel needing another
 * factor defines its own. */
#define CBK_UNROLL _Pragma("GCC unroll 16")

template <int IB, int JB, bool Diag, bool WithD, typename T, int V, typename Int>
inline void sytrfnp_syrk_tile(BatchView<T, V, Int> A, Int i, Int jj, Int k0, Int k1)
{
    using VT = typename pack<T, V>::type;
    VT acc[IB][JB];
    CBK_UNROLL
    for (int r = 0; r < IB; ++r) {
        CBK_UNROLL
        for (int c = 0; c < JB; ++c)
            if (!Diag || r >= c) acc[r][c] = A(i + r, jj + c);
    }
    for (Int k = k0; k < k1; ++k) {
        VT u[JB], w[JB];
        CBK_UNROLL
        for (int c = 0; c < JB; ++c)
            u[c] = A(jj + c, k);
        if constexpr (WithD) {
            const VT d = A(k, k);
            CBK_UNROLL
            for (int c = 0; c < JB; ++c)
                w[c] = u[c] * d;
        }
        else {
            CBK_UNROLL
            for (int c = 0; c < JB; ++c)
                w[c] = u[c];
        }
        CBK_UNROLL
        for (int r = 0; r < IB; ++r) {
            const VT av = Diag ? u[r] : A(i + r, k);
            CBK_UNROLL
            for (int c = 0; c < JB; ++c)
                if (!Diag || r >= c) acc[r][c] -= av * w[c];
        }
    }
    CBK_UNROLL
    for (int r = 0; r < IB; ++r) {
        CBK_UNROLL
        for (int c = 0; c < JB; ++c)
            if (!Diag || r >= c) A(i + r, jj + c) = acc[r][c];
    }
}

#undef CBK_UNROLL

/* The rank-K update of the JB columns jj .. jj+JB-1, rows jj .. n-1, by the
 * columns k0 .. k1-1: the diagonal tile, then 4-row tiles down the column
 * block (the 4 x 4 tile is 16 accumulators, 4 + 4 weights, the pivot and a
 * load: within the 32 registers of AVX-512), then the leftover rows one at a
 * time. */
template <int JB, bool WithD, typename T, int V, typename Int>
inline void sytrfnp_syrk_cols(Int n, BatchView<T, V, Int> A, Int jj, Int k0, Int k1)
{
    sytrfnp_syrk_tile<JB, JB, true, WithD, T, V, Int>(A, jj, jj, k0, k1);
    Int i = jj + JB;
    for (; i + 4 <= n; i += 4)
        sytrfnp_syrk_tile<4, JB, false, WithD, T, V, Int>(A, i, jj, k0, k1);
    for (; i < n; ++i)
        sytrfnp_syrk_tile<1, JB, false, WithD, T, V, Int>(A, i, jj, k0, k1);
}

/* Rank-K update of the lower trapezoid A(jj:n, jj), k1 <= jj < jend, by the
 * columns k0 .. k1-1: four-column blocks, then the leftover columns as one
 * narrower block. */
template <bool WithD, typename T, int V, typename Int>
inline void sytrfnp_trailing_update(Int n, Int jend, BatchView<T, V, Int> A, Int k0,
                                    Int k1)
{
    Int jj = k1;
    for (; jj + 4 <= jend; jj += 4)
        sytrfnp_syrk_cols<4, WithD, T, V, Int>(n, A, jj, k0, k1);
    switch (jend - jj) {
    case 3: sytrfnp_syrk_cols<3, WithD, T, V, Int>(n, A, jj, k0, k1); break;
    case 2: sytrfnp_syrk_cols<2, WithD, T, V, Int>(n, A, jj, k0, k1); break;
    case 1: sytrfnp_syrk_cols<1, WithD, T, V, Int>(n, A, jj, k0, k1); break;
    default: break;
    }
}

/* The unblocked sweep on the panel of columns j0 .. j1-1, rows to n, already
 * updated by every column left of j0: each pivot's rank-1 update is confined
 * to the panel's remaining columns and runs on the still-unscaled column with
 * the weights scaled by 1/d (parked at A(j,j) for the block helper), the
 * column scaled after it; the pivot d is restored in place as D(j,j). */
template <typename T, int V, typename Int>
inline void sytrfnp_compact_panel(Int j0, Int j1, Int n, BatchView<T, V, Int> A)
{
    using VT = typename pack<T, V>::type;
    for (Int j = j0; j < j1; ++j) {
        const VT d = A(j, j);
        const VT invd = T(1) / d;
        A(j, j) = invd;

        Int jj = j + 1;
        for (; jj + 4 <= j1; jj += 4)
            sytrfnp_update_block<4, T, V>(j, n, A, jj);
        for (; jj < j1; ++jj)
            sytrfnp_update_block<1, T, V>(j, n, A, jj);

        A(j, j) = d;
        for (Int i = j + 1; i < n; ++i)
            A(i, j) = A(i, j) * invd;
    }
}

/* Factor the columns j0 .. j1-1 (rows to n, already updated by every column
 * left of j0), recursively: a panel of at most sytrfnp_nb columns is the
 * unblocked sweep; a wider range is split at a panel boundary, the left half
 * factored, the right half updated by it as one rank-(mid-j0) tiled update,
 * then factored. */
template <typename T, int V, typename Int>
void sytrfnp_compact_block(Int j0, Int j1, Int n, BatchView<T, V, Int> A)
{
    if (j1 - j0 <= sytrfnp_nb) {
        sytrfnp_compact_panel<T, V, Int>(j0, j1, n, A);
        return;
    }
    const Int npanels = (j1 - j0 + sytrfnp_nb - 1) / sytrfnp_nb;
    const Int mid = j0 + (npanels / 2) * sytrfnp_nb;
    sytrfnp_compact_block<T, V, Int>(j0, mid, n, A);
    sytrfnp_trailing_update<true, T, V, Int>(n, j1, A, j0, mid);
    sytrfnp_compact_block<T, V, Int>(mid, j1, n, A);
}

/* One group of V interleaved n x n matrices: factor the lower triangle of the
 * view (see the file header for how the view maps uplo/layout onto it). */
template <typename T, int V, typename Int = int>
void sytrfnp_compact_group(Int n, BatchView<T, V, Int> A)
{
    static_assert(std::is_floating_point_v<T>,
                  "sytrfnp_compact is defined for real float/double");
    assert(A.si && A.sj);
    sytrfnp_compact_block<T, V, Int>(Int(0), n, n, A);
}

/* ~flops of the LDL^T of one group: n^3/3 per matrix, times V
 * (shared with the fused sysvnp driver). */
template <typename Int> inline double sytrfnp_flops(Int n, int V)
{
    return (double)n * n * n / 3.0 * V;
}

/* All groups, any layout / uplo. A padded partial last group is processed too,
 * harmlessly: the identity's LDL^T factor is L = I, D = I (no fill). */
template <typename T, int V, typename Int = int>
void sytrfnp_compact(bool rowmajor, bool upper, Int n, T *ap, Int ldap, Int nm)
{
    assert(nm >= 1 && n >= 0);

    const std::size_t str_a = group_stride(rowmajor, ldap, n, n, V);

    for_each_group<V>(
        nm,
        [&](Int g) {
            /* the kernel factors the lower triangle of the view it is given: A
             * itself for uplo lower, A^T for upper (U^T D U = A is L D L^T of A^T) */
            sytrfnp_compact_group<T, V, Int>(
                n, make_lower_view<T, V, Int>(ap + g * str_a, rowmajor, upper, ldap));
        },
        sytrfnp_flops(n, V));
}

} /* namespace cbk::detail */

CBK_KERNEL_END

#endif /* CBK_SYTRFNP_COMPACT_HPP */
