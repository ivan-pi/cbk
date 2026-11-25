/* cbk_gepack_compact.hpp
 *
 * Conversion between a batch of conventionally stored (dense) matrices and the
 * compact (interleaved) format, templated on scalar type T and interleave width
 * V -- the portable mkl_?gepack_compact / mkl_?geunpack_compact, and the size
 * of the buffer they convert to and from (cbk_get_size_compact, which unlike
 * mkl_?get_size_compact is one function taking the scalar's size):
 *
 *     gepack    ap  <-  a[0 .. nm-1]      interleave
 *     geunpack  a[0 .. nm-1]  <-  ap      de-interleave
 *
 * a is an array of nm pointers, one per m x n matrix, each stored in `layout`
 * with leading dimension lda; ap is the compact batch in the same layout with
 * leading dimension ldap. Group g = idx/V, slot v = idx%V:
 *     A_v(i,j) = ap[ g*ldap*n*V + (j*ldap + i)*V + v ]   (column-major)
 * and row-major swaps the in-matrix roles (i*ldap + j), the group then holding
 * ldap*m packs.
 *
 * Padding. The compact batch is ceil(nm/V) whole groups; gepack fills the
 * slots past nm in the last group with the identity (ones on the diagonal,
 * zeros elsewhere), which every kernel maps to a finite result, so the kernels
 * run a partial group unmasked. The rows (columns, row-major) between the
 * matrix extent and the leading dimension are padding of another kind and are
 * left untouched on both sides. Both choices are mkl_?gepack_compact's,
 * measured: a buffer packed here is the buffer MKL packs.
 *
 * Each (pack, lane) is one scalar load or store through the matrix pointers --
 * a gather the compiler does not vectorize and need not: the conversion is
 * memory-bound, and the groups run in parallel.
 *
 * Assisted-by: Claude:claude-fable-5
 */

#ifndef CBK_GEPACK_COMPACT_HPP
#define CBK_GEPACK_COMPACT_HPP

#include "cbk_common.hpp"

#include <cstddef>
#include <cassert>

CBK_KERNEL_BEGIN

namespace cbk::detail {

/* Bytes of a compact buffer for nm matrices of leading dimension ld and second
 * dimension sd (the column count column-major, the row count row-major):
 * ld * sd packs for each of the ceil(nm / V) groups. */
template <typename Int>
std::size_t compact_size_bytes(Int ld, Int sd, int V, Int nm,
                               std::size_t elem_size) noexcept
{
    return (std::size_t)ld * (std::size_t)sd * (std::size_t)V *
           (std::size_t)group_count(nm, V) * elem_size;
}

/* Offset of element (i,j) in a dense matrix of leading dimension ld. */
template <typename Int>
inline std::size_t dense_index(bool rowmajor, Int i, Int j, Int ld) noexcept
{
    return rowmajor ? (std::size_t)i * ld + j : (std::size_t)j * ld + i;
}

/* All groups: ap <- a. The slots of the last group past nm get the identity. */
template <typename T, int V, typename Int = int>
void gepack_compact(bool rowmajor, Int m, Int n, const T *const *a, Int lda, T *ap,
                    Int ldap, Int nm)
{
    assert(nm >= 1 && m >= 0 && n >= 0);

    const std::size_t str_a = group_stride(rowmajor, ldap, m, n, V);

    for_each_group<V>(
        nm,
        [&](Int g) {
            const auto P = make_view<T, V, Int>(ap + g * str_a, rowmajor, ldap);
            const Int idx0 = g * V;
            const int live = (nm - idx0 < V) ? (int)(nm - idx0) : V;
            for (Int j = 0; j < n; ++j)
                for (Int i = 0; i < m; ++i) {
                    const std::size_t e = dense_index(rowmajor, i, j, lda);
                    typename pack<T, V>::type x;
                    for (int v = 0; v < live; ++v)
                        x[v] = a[idx0 + v][e];
                    for (int v = live; v < V; ++v)
                        x[v] = (i == j) ? T(1) : T(0);
                    P(i, j) = x;
                }
        },
        (double)m * n * V /* scalars moved per group */);
}

/* All groups: a <- ap. The padded slots of the last group are not read. */
template <typename T, int V, typename Int = int>
void geunpack_compact(bool rowmajor, Int m, Int n, T *const *a, Int lda, const T *ap,
                      Int ldap, Int nm)
{
    assert(nm >= 1 && m >= 0 && n >= 0);

    const std::size_t str_a = group_stride(rowmajor, ldap, m, n, V);

    for_each_group<V>(
        nm,
        [&](Int g) {
            const auto P = make_const_view<T, V, Int>(ap + g * str_a, rowmajor, ldap);
            const Int idx0 = g * V;
            const int live = (nm - idx0 < V) ? (int)(nm - idx0) : V;
            for (Int j = 0; j < n; ++j)
                for (Int i = 0; i < m; ++i) {
                    const std::size_t e = dense_index(rowmajor, i, j, lda);
                    const typename pack<T, V>::type x = P(i, j);
                    for (int v = 0; v < live; ++v)
                        a[idx0 + v][e] = x[v];
                }
        },
        (double)m * n * V /* scalars moved per group */);
}

} /* namespace cbk::detail */

CBK_KERNEL_END

#endif /* CBK_GEPACK_COMPACT_HPP */
