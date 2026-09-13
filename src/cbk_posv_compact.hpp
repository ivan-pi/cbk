/* cbk_posv_compact.hpp
 *
 * Compact (interleaved-batch) symmetric positive-definite solve A X = B in one
 * pass: the Cholesky factorization of cbk_potrf_compact.hpp fused with the
 * two-sweep solve of cbk_potrs_compact.hpp, per group. The LAPACK ?posv
 * analogue (factor, then solve).
 *
 * Fusing matters for throughput, not arithmetic: the result is bit-identical
 * to potrf followed by potrs (the same group kernels run in the same order on
 * the same data), but each group's factor is solved with while it is still
 * cache-resident, instead of the batch being streamed once for the
 * factorization and once more for the solve -- the penalty the solve benchmark
 * measured for whole-pool pipelines (PLANS.md) -- and the whole solve threads
 * as one group loop. On exit ap holds the factor (L or U in the named
 * triangle, the opposite triangle untouched) and bp holds X. As in LAPACK
 * ?posv -- which calls ?potrf unconditionally; the nrhs = 0 quick return is
 * ?potrs's -- nrhs = 0 still factors ap, and bp is then never referenced.
 *
 * Assisted-by: Claude
 */

#ifndef CBK_POSV_COMPACT_HPP
#define CBK_POSV_COMPACT_HPP

#include "cbk_common.hpp"
#include "cbk_potrf_compact.hpp"
#include "cbk_potrs_compact.hpp"

#include <cstddef>
#include <cassert>

namespace cbk::detail {

/* All groups, any layout / uplo: per group, factor then solve. */
template <typename T, int V, typename Int = int>
void posv_compact(bool rowmajor, bool upper, Int n, Int nrhs, T *ap, Int ldap, T *bp,
                  Int ldbp, Int nm)
{
    assert(nm >= 1 && n >= 0 && nrhs >= 0);

    /* no right-hand sides: factor anyway (LAPACK ?posv), touching only ap */
    if (nrhs == 0) {
        potrf_compact<T, V, Int>(rowmajor, upper, n, ap, ldap, nm);
        return;
    }

    const std::size_t str_a = group_stride(rowmajor, ldap, n, n, V);
    const std::size_t str_b = group_stride(rowmajor, ldbp, n, nrhs, V);
    /* the work estimate for_each_group's threading gate weighs: both fused steps */
    const double flops_per_group = potrf_flops(n, V) + potrs_flops(n, nrhs, V);

    for_each_group<V>(
        nm,
        [&](Int g) {
            T *a = ap + g * str_a;
            potrf_compact_group<T, V, Int>(
                n, make_lower_view<T, V, Int>(a, rowmajor, upper, ldap));
            potrs_compact_group<T, V, Int>(rowmajor, upper, n, nrhs, a, ldap,
                                           bp + g * str_b, ldbp);
        },
        flops_per_group);
}

} /* namespace cbk::detail */

#endif /* CBK_POSV_COMPACT_HPP */
