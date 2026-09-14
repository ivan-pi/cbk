/* cbk_potrf_compact.hpp
 *
 * Compact (interleaved-batch) Cholesky factorization of symmetric
 * positive-definite matrices, templated on scalar type T and interleave width V:
 *
 *     A = L L^T  (uplo lower),   A = U^T U  (uplo upper),
 *
 * with a positive diagonal, run V matrices at a time. A portable, vectorized
 * mkl_?potrf_compact.
 *
 * Algorithm: the right-looking LAPACK potf2, lifted double -> V-wide vector,
 * one lane per matrix, blocked recursively. Cholesky has no data-dependent
 * branch, so unlike geqrf's larfg no lane mask is needed; the pivot is an
 * unconditional sqrt. The columns are split in halves at NB-column panel
 * boundaries down to panels of at most NB pivots (lower):
 *
 *     factor(j0, j1):
 *         if j1 - j0 <= NB: potf2 on the panel, each pivot's rank-1 update
 *                           confined to the panel's remaining columns
 *         else: factor(j0, mid); update A(mid:n, mid:j1) by the columns
 *               j0 .. mid-1 (rank-(mid-j0), register-tiled); factor(mid, j1)
 *
 * and the panel's potf2, per pivot column j:
 *
 *     inva = 1/A(j,j)
 *     A(i,jj)-= A(i,j)*(A(jj,j)*inva)  for j < jj < j1, i >= jj  (rank-1, in-panel)
 *     d = sqrt(A(j,j));  A(j,j) = d;  A(i,j) *= d*inva  for i > j (scale column)
 *
 * The tiled rank-K updates make the trailing matrix's traffic (the plain
 * sweep's bound) one pass per split level instead of one per pivot, in the
 * same pivot order; the panel's reordered pivot (divide first, sqrt off the
 * pivot-to-pivot chain; the rank-1 update on the unscaled column is sytrfnp's
 * step with d = a) shortens the serial chain that is the whole cost at small
 * n. Design document sections 6.1 and 6.2 give the rationale and the
 * measurements.
 *
 * Only the lower trapezoid of the *view* is touched: the kernel gets A for uplo
 * lower and A^T for upper (A symmetric, so U = L(A^T)^T lands in the upper
 * storage), and the strictly-opposite triangle passes through untouched as
 * ?potrf requires. Column-major lower and row-major upper are the contiguous
 * cases (unit row stride); the other two are strided.
 *
 * Positive-definiteness is assumed, not enforced (design section 6.2): a
 * non-SPD lane gets NaN/Inf in its factor; there is no info = j early exit.
 *
 * Compact storage (matches mkl_?gepack_compact); group g = idx/V, slot v = idx%V:
 *     A_v(i,j) = ap[ g*ldap*n*V + (j*ldap + i)*V + v ]   (column-major)
 * Row-major swaps the in-matrix roles (i*ldap + j); the group stride is the
 * same either way (A is n x n).
 *
 * Assisted-by: Claude:claude-opus-4-8 Claude:claude-fable-5
 */

#ifndef CBK_POTRF_COMPACT_HPP
#define CBK_POTRF_COMPACT_HPP

#include "cbk_common.hpp"
#include "cbk_sytrfnp_compact.hpp" /* sytrfnp_update_block */

#include <cstddef>
#include <cassert>
#include <type_traits>

CBK_KERNEL_BEGIN

namespace cbk::detail {

/* Leaf panel width: the pivot columns factored by the unblocked sweep between
 * tiled updates. The rank-1 sweep's share of the flops is about 1.5 NB/n; the
 * tiled updates' traffic falls with the number of split levels, log2(n/NB). */
#ifndef CBK_POTRF_NB
#define CBK_POTRF_NB 8
#endif
constexpr int potrf_nb = CBK_POTRF_NB;

/* Rank-K update of one IB x JB tile of the trailing matrix, rows i .. i+IB-1
 * and columns jj .. jj+JB-1, by the columns k0 .. k1-1:
 *     A(i+r, jj+c) -= sum_k A(i+r,k) * A(jj+c,k)
 * accumulated in registers over k, in pivot order, so the tile is loaded and
 * stored once and each pack of the k columns feeds IB (or JB) FMAs. Diag: the
 * tile sits on the diagonal (i == jj) and only its lower triangle r >= c
 * exists -- the loop bounds are compile-time, so the guard folds away. */
/* CBK_UNROLL before each loop over the tile asks for it to be fully unrolled
 * (GCC and clang both take the GCC spelling; 16 covers the 4 x 4 tile), which
 * keeps acc[][] in registers under GCC, whose scalar replacement runs before
 * it unrolls these loops and would otherwise stage the tile through the stack
 * on entry and exit. Undefined after the tile: a kernel needing another
 * factor defines its own. */
#define CBK_UNROLL _Pragma("GCC unroll 16")

template <int IB, int JB, bool Diag, typename T, int V, typename Int>
inline void potrf_syrk_tile(BatchView<T, V, Int> A, Int i, Int jj, Int k0, Int k1)
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
        VT w[JB];
        CBK_UNROLL
        for (int c = 0; c < JB; ++c)
            w[c] = A(jj + c, k);
        CBK_UNROLL
        for (int r = 0; r < IB; ++r) {
            /* on the diagonal the row and column packs are the same loads */
            const VT av = Diag ? w[r] : A(i + r, k);
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
 * block (the 4 x 4 tile is 16 accumulators, 4 weights and a load: the 32
 * registers of AVX-512), then the leftover rows one at a time. */
template <int JB, typename T, int V, typename Int>
inline void potrf_syrk_cols(Int n, BatchView<T, V, Int> A, Int jj, Int k0, Int k1)
{
    potrf_syrk_tile<JB, JB, true, T, V, Int>(A, jj, jj, k0, k1);
    Int i = jj + JB;
    for (; i + 4 <= n; i += 4)
        potrf_syrk_tile<4, JB, false, T, V, Int>(A, i, jj, k0, k1);
    for (; i < n; ++i)
        potrf_syrk_tile<1, JB, false, T, V, Int>(A, i, jj, k0, k1);
}

/* Rank-K update of the lower trapezoid A(jj:n, jj), k1 <= jj < jend, by the
 * columns k0 .. k1-1: four-column blocks, then the leftover columns as one
 * narrower block. */
template <typename T, int V, typename Int>
inline void potrf_trailing_update(Int n, Int jend, BatchView<T, V, Int> A, Int k0, Int k1)
{
    Int jj = k1;
    for (; jj + 4 <= jend; jj += 4)
        potrf_syrk_cols<4, T, V, Int>(n, A, jj, k0, k1);
    switch (jend - jj) {
    case 3: potrf_syrk_cols<3, T, V, Int>(n, A, jj, k0, k1); break;
    case 2: potrf_syrk_cols<2, T, V, Int>(n, A, jj, k0, k1); break;
    case 1: potrf_syrk_cols<1, T, V, Int>(n, A, jj, k0, k1); break;
    default: break;
    }
}

/* The unblocked potf2 on the panel of columns j0 .. j1-1, rows to n, already
 * updated by every column left of j0: each pivot's rank-1 update is confined
 * to the panel's remaining columns. */
template <typename T, int V, typename Int>
inline void potrf_compact_panel(Int j0, Int j1, Int n, BatchView<T, V, Int> A)
{
    using VT = typename pack<T, V>::type;
    for (Int j = j0; j < j1; ++j) {
        /* pivot a = A(j,j): the rank-1 update first, on the unscaled column
         * with 1/a parked at A(j,j) as the block helper's weight (sytrfnp's
         * step with d = 1/a), then the sqrt and the scaling off the chain to
         * the next pivot; 1/sqrt(a) = sqrt(a)/a, no second divide */
        const VT a = A(j, j);
        const VT inva = T(1) / a;
        A(j, j) = inva;

        Int jj = j + 1;
        for (; jj + 4 <= j1; jj += 4)
            sytrfnp_update_block<4, T, V>(j, n, A, jj);
        for (; jj < j1; ++jj)
            sytrfnp_update_block<1, T, V>(j, n, A, jj);

        VT d;
        vsqrt<T, V>(d, a);
        A(j, j) = d;
        const VT invd = d * inva;
        for (Int i = j + 1; i < n; ++i)
            A(i, j) = A(i, j) * invd;
    }
}

/* Factor the columns j0 .. j1-1 (rows to n, already updated by every column
 * left of j0), recursively: a panel of at most potrf_nb columns is the
 * unblocked sweep; a wider range is split at a panel boundary, the left half
 * factored, the right half updated by it as one rank-(mid-j0) tiled update,
 * then factored. */
template <typename T, int V, typename Int>
void potrf_compact_block(Int j0, Int j1, Int n, BatchView<T, V, Int> A)
{
    if (j1 - j0 <= potrf_nb) {
        potrf_compact_panel<T, V, Int>(j0, j1, n, A);
        return;
    }
    const Int npanels = (j1 - j0 + potrf_nb - 1) / potrf_nb;
    const Int mid = j0 + (npanels / 2) * potrf_nb;
    potrf_compact_block<T, V, Int>(j0, mid, n, A);
    potrf_trailing_update<T, V, Int>(n, j1, A, j0, mid);
    potrf_compact_block<T, V, Int>(mid, j1, n, A);
}

/* One group of V interleaved n x n matrices: factor the lower triangle of the
 * view (see the file header for how the view maps uplo/layout onto it). */
template <typename T, int V, typename Int = int>
void potrf_compact_group(Int n, BatchView<T, V, Int> A)
{
    static_assert(std::is_floating_point_v<T>,
                  "potrf_compact is defined for real float/double");
    assert(A.si && A.sj);
    potrf_compact_block<T, V, Int>(Int(0), n, n, A);
}

/* ~flops of the Cholesky of one group: n^3/3 per matrix, times V
 * (shared with the fused posv driver). */
template <typename Int> inline double potrf_flops(Int n, int V)
{
    return (double)n * n * n / 3.0 * V;
}

/* All groups, any layout / uplo. A padded partial last group is processed too,
 * harmlessly: the identity's Cholesky factor is the identity. */
template <typename T, int V, typename Int = int>
void potrf_compact(bool rowmajor, bool upper, Int n, T *ap, Int ldap, Int nm)
{
    assert(nm >= 1 && n >= 0);

    const std::size_t str_a = group_stride(rowmajor, ldap, n, n, V);

    for_each_group<V>(
        nm,
        [&](Int g) {
            /* the kernel factors the lower triangle of the view it is given: A
             * itself for uplo lower, A^T for upper (U^T U = A is L L^T of A^T) */
            potrf_compact_group<T, V, Int>(
                n, make_lower_view<T, V, Int>(ap + g * str_a, rowmajor, upper, ldap));
        },
        potrf_flops(n, V));
}

} /* namespace cbk::detail */

CBK_KERNEL_END

#endif /* CBK_POTRF_COMPACT_HPP */
