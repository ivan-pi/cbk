/* cbk_compact_pack.hpp
 *
 * Compact pack / unpack with no MKL: pack_compact interleaves a dense
 * MatrixBatch into a compact buffer in the layout of mkl_?gepack_compact,
 * unpack_compact reads one back, and compact_size is the buffer they need.
 * The pair the BLAS-free test suites and the portable benchmark pack with
 * (the MKL-backed suites and benchmarks use MKL's own routines); internal,
 * like the views: src/ is on their include path, the public API stays include/.
 *
 * Group g = idx/V, slot v = idx%V: element (i,j) of every matrix in a group is
 * one V-wide pack, and the packs of a group sit at the leading dimension ldp
 * in the given layout. Both sides are addressed through the project's views:
 * the dense side through MatrixBatch (hence MatrixView), the interleaved side
 * through the kernels' own BatchView, whose element is the pack and whose
 * strides are in packs -- `P(i, j)[v]` is element (i,j) of slot v. The layout
 * is the view's strides, so one body serves column-major and row-major, and
 * the group offset is the library's group_stride. BatchView is templated on
 * the interleave width, so for_vlen turns the runtime V into the compile-time
 * one (2, 4, 8 or 16 -- the widths the C API accepts). Sharing the kernels'
 * views here is deliberate; see "Two views, one idea" in .claude/CLAUDE.md.
 *
 * A padded slot (idx >= nm) carries pad_diag on its diagonal, zero elsewhere:
 * the identity for matrix batches (so kernels run the padding unmasked), zero
 * for tau batches (the identity's reflectors).
 *
 * Assisted-by: Claude:claude-fable-5
 */

#ifndef CBK_COMPACT_PACK_HPP
#define CBK_COMPACT_PACK_HPP

#include "cbk_common.hpp"
#include "cbk_matrix_batch.hpp"

#include <cassert>
#include <cstddef>

namespace cbk::detail {

/* Scalars in the compact buffer of nm rows x cols matrices at leading
 * dimension ldp -- the one place the ng * gstride sizing is written. */
inline std::size_t compact_size(int nm, int rows, int cols, int ldp, int V,
                                bool rowmajor = false)
{
    const int ng = (nm + V - 1) / V;
    return (std::size_t)ng * group_stride(rowmajor, ldp, rows, cols, V);
}

template <class T, class Alloc>
void pack_compact(const MatrixBatch<T, Alloc> &Mk, T *p, int ldp, int V,
                  bool rowmajor = false, T pad_diag = T(1))
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

template <class T, class Alloc>
void unpack_compact(MatrixBatch<T, Alloc> &Mk, const T *p, int ldp, int V,
                    bool rowmajor = false)
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

} /* namespace cbk::detail */

#endif /* CBK_COMPACT_PACK_HPP */
