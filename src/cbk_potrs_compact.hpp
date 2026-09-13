/* cbk_potrs_compact.hpp
 *
 * Compact (interleaved-batch) solve A X = B from the Cholesky factorization of
 * cbk_potrf_compact.hpp, templated on scalar type T and interleave width V.
 * With the factor held in ap (L or U in the named triangle) and B (n x nrhs)
 * in bp, each matrix's solve is two in-place substitution sweeps:
 *
 *     uplo lower, A = L L^T:   L z = B;    L^T X = z
 *     uplo upper, A = U^T U:   U^T z = B;  U   X = z
 *
 * Both sweeps are the compact trsm group kernels with a non-unit diagonal:
 * column-major takes trsm's tuned side='L' row-dot path, row-major the strided
 * kernel (except a single RHS column, contiguous in either layout, which the
 * view routing sends to the tuned path too). No workspace; B is overwritten
 * with X. The LAPACK ?potrs analogue.
 *
 * A NaN/Inf diagonal -- a lane poisoned by a non-SPD input in the factorization
 * (design section 6.2) -- propagates into that lane's solution, matching the
 * factorization's no-check contract. Padded slots of a partial last group hold
 * identity factors, so the whole group runs unmasked, as in the factorization
 * and trsm.
 *
 * Compact storage is as in cbk_potrf_compact.hpp / cbk_trsm_compact.hpp: the
 * factor batch ap is n x n with leading dimension ldap, the RHS batch bp is
 * n x nrhs with leading dimension ldbp (>= n column-major, >= nrhs row-major).
 *
 * Assisted-by: Claude:claude-fable-5
 */

#ifndef CBK_POTRS_COMPACT_HPP
#define CBK_POTRS_COMPACT_HPP

#include "cbk_common.hpp"
#include "cbk_trsm_compact.hpp"

#include <cstddef>
#include <cassert>

namespace cbk::detail {

/* One group: the two sweeps on the factor at `a` and the RHS block at `b`.
 * The stored factor is transposed in the first sweep exactly when it is upper
 * (U^T z = B), and in the second exactly when it is lower (L^T X = z). */
template <typename T, int V, typename Int = int>
void potrs_compact_group(bool rowmajor, bool upper, Int n, Int nrhs, const T *a, Int ldap,
                         T *b, Int ldbp)
{
    /* non-unit triangular sweep with op(F) = F (tran false) or F^T (tran true) */
    const auto sweep = [&](bool tran) {
        trsm_compact_group<T, V, Int>(/*left=*/true, upper, rowmajor, tran,
                                      /*unit=*/false, n, nrhs, T(1), a, ldap, b, ldbp);
    };

    sweep(upper);  /* L z = B   or  U^T z = B */
    sweep(!upper); /* L^T X = z or  U   X = z */
}

/* ~flops of the two sweeps of one group: 2 n^2 nrhs per matrix, times V. */
template <typename Int> inline double potrs_flops(Int n, Int nrhs, int V)
{
    return 2.0 * n * n * nrhs * V;
}

/* All groups, any layout / uplo. */
template <typename T, int V, typename Int = int>
void potrs_compact(bool rowmajor, bool upper, Int n, Int nrhs, const T *ap, Int ldap,
                   T *bp, Int ldbp, Int nm)
{
    assert(nm >= 1 && n >= 0 && nrhs >= 0);

    const std::size_t str_a = group_stride(rowmajor, ldap, n, n, V);
    const std::size_t str_b = group_stride(rowmajor, ldbp, n, nrhs, V);

    for_each_group<V>(
        nm,
        [&](Int g) {
            potrs_compact_group<T, V, Int>(rowmajor, upper, n, nrhs, ap + g * str_a, ldap,
                                           bp + g * str_b, ldbp);
        },
        potrs_flops(n, nrhs, V));
}

} /* namespace cbk::detail */

#endif /* CBK_POTRS_COMPACT_HPP */
