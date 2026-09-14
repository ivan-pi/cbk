/* cbk_gels_compact.hpp
 *
 * Compact (interleaved-batch) least-squares / minimum-norm solve, templated on
 * scalar type T and interleave width V -- the compact form of LAPACK ?gels:
 *
 *     op(A) X = B,   op(A) = A ('N') or A^T ('T'),   A is m x n,
 *
 * assuming op(A) has full rank. When op(A) has more rows than columns the
 * system is overdetermined and X is the least-squares solution
 * (min ||B - op(A) X||_F); when it has more columns than rows it is
 * underdetermined and X is the minimum-norm solution (min ||X||_F subject to
 * op(A) X = B). The solve runs per group of V matrices in one call -- factor,
 * apply Q, back-substitute -- on the group's cache-resident buffers, which is
 * what a geqrf -> ormqr -> trsm chain of whole-batch calls cannot do.
 *
 * Four cases collapse to one kernel. Let F be the *tall* orientation of A:
 *     F = A    (m >= n, LAPACK's QR path)   or   F = A^T  (m < n, LAPACK's LQ path),
 * p x q with p = max(m, n), q = min(m, n). In the compact buffer F is A's
 * BatchView, transposed when m < n (free: the strides swap), and geqrf's kernel
 * over that view IS the LQ factorization in ?gelqf's storage convention (L on
 * and below A's diagonal, reflector rows to its right). With F = Q [R; 0]:
 *
 *   overdetermined  F X = B          B := Q^T B, then R X = B(0:q); rows q..p-1
 *   (m >= n & 'N',                   of B hold the residual (their squared
 *    m <  n & 'T')                   column norms are the residual sums of
 *                                    squares, as for ?gels)
 *   underdetermined F^T X = B        R^T Y = B(0:q), B(q:p) := 0, then B := Q B
 *   (m >= n & 'T',                   (X = Q [Y; 0] is the minimum-norm solution)
 *    m <  n & 'N')
 *
 * Both cases run the three group kernels in sequence -- geqrf, ormqr, trsm --
 * on the group's buffers. (A fused variant that applied each reflector to B as
 * it was built saved no arithmetic and, once F's trailing block and B together
 * outgrew L1, evicted B on every step; it was retired for the plain sequence.)
 *
 * Compact storage (matches mkl_?gepack_compact); group g = idx/V, slot v = idx%V:
 *     A_v(i,j)  = ap  [ g*ldap*n*V    + (j*ldap + i)*V + v ]   (column-major)
 *     B_v(i,j)  = bp  [ g*ldbp*nrhs*V + (j*ldbp + i)*V + v ]
 *     tau_v(kk) = taup[ g*q*V         +  kk*V          + v ]
 * B is max(m, n) x nrhs per matrix, as ?gels declares it (ldb >= max(m, n)):
 * on entry its first (rows of op(A)) rows hold the right-hand sides, on exit
 * its first (columns of op(A)) rows the solution. On exit ap holds the QR (m >= n)
 * or LQ (m < n) factorization of A, and taup the reflector scalars tau, in
 * compact format: together they are the (H, tau) that ?ormqr_compact accepts.
 * (The MKL-style API hands taup in as its work array; the portable C API as an
 * explicit output, like ?geqrf_compact's.)
 *
 * Assisted-by: Claude:claude-fable-5
 */

#ifndef CBK_GELS_COMPACT_HPP
#define CBK_GELS_COMPACT_HPP

#include "cbk_common.hpp"
#include "cbk_geqrf_compact.hpp"
#include "cbk_ormqr_compact.hpp"
#include "cbk_trsm_compact.hpp"

#include <cstddef>
#include <cassert>
#include <type_traits>

CBK_KERNEL_BEGIN

namespace cbk::detail {

/* The tau buffer: q = min(m, n) reflector scalars per matrix, one slot per
 * group so the groups can run in parallel -- the size of a compact tau buffer
 * for the batch (mkl_?get_size_compact(min(m, n), 1, format, nm) in scalars).
 * At least 1, like LAPACK's lwork, for the MKL-style API's workspace query. */
template <typename Int> Int gels_lwork(Int m, Int n, Int nm, int V) noexcept
{
    const Int q = (m < n) ? m : n;
    const Int ngroups = (nm + V - 1) / V;
    const Int need = q * V * ngroups;
    return need < 1 ? Int(1) : need;
}

/* One group: F (p x q, p >= q) is the tall view of A, B its p x nrhs
 * right-hand-side view, tau_ the group's q scalars. overdet selects F X = B
 * (least squares) over F^T X = B (minimum norm). The triangular steps are on
 * the q x q upper triangle R of F, through trsm's group kernel (which reaches
 * its tuned path for column-major storage, transposed view included). */
template <typename T, int V, typename Int = int>
void gels_compact_group(bool overdet, Int p, Int q, Int nrhs, BatchView<T, V, Int> F,
                        T *tau_, BatchView<T, V, Int> B)
{
    static_assert(std::is_floating_point_v<T>,
                  "gels_compact is defined for real float/double");
    assert(p >= q && F.si && F.sj && B.si && B.sj);

    const auto Fc = F.as_const();
    if (overdet) {
        /* F = Q [R; 0]: B := Q^T B, then R X = (Q^T B)(0:q). */
        geqrf_compact_group<T, V, Int>(p, q, F, tau_);
        ormqr_compact_group<T, V, Int>(Direction::Forward, p, nrhs, q, Fc, tau_, B);
        trsm_compact_group<T, V, Int>(true, true, false, false, q, nrhs, T(1), Fc, B);
    }
    else {
        /* F^T = [R^T 0] Q^T: solve R^T Y = B(0:q), then X = Q [Y; 0]. */
        geqrf_compact_group<T, V, Int>(p, q, F, tau_);
        trsm_compact_group<T, V, Int>(true, true, true, false, q, nrhs, T(1), Fc, B);
        zero_block<T, V, Int>(B, q, p, nrhs);
        ormqr_compact_group<T, V, Int>(Direction::Backward, p, nrhs, q, Fc, tau_, B);
    }
}

/* All groups, either layout. trans is 'N', or 'T'/'C' for the transpose. taup
 * holds gels_lwork(m, n, nm, V) scalars. A padded partial last group is
 * processed too, harmlessly: an identity lane (of A, and of B) factors to
 * R = I, tau = 0, and solves to X = B. A rank-deficient op(A) is not detected:
 * a zero diagonal of R divides through to Inf/NaN in that lane, as ?trsm. */
template <typename T, int V, typename Int = int>
void gels_compact(bool rowmajor, char trans, Int m, Int n, Int nrhs, T *ap, Int ldap,
                  T *bp, Int ldbp, T *taup, Int nm)
{
    assert(nm >= 1 && m >= 0 && n >= 0 && nrhs >= 0);

    const bool tran = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');
    const bool tall = (m >= n);
    const Int p = tall ? m : n, q = tall ? n : m;
    const bool overdet = (tall != tran); /* op(A) has p rows and q columns */

    const std::size_t str_a = group_stride(rowmajor, ldap, m, n, V);
    const std::size_t str_b = group_stride(rowmajor, ldbp, p, nrhs, V);
    const std::size_t str_t = (std::size_t)q * V;

    if (q == 0) {
        /* an empty op(A): the (minimum-norm) solution is X = 0, as ?gels sets
         * B(0:max(m,n), :) := 0 when min(m, n) = 0 */
        zero_compact<T, V, Int>(rowmajor, p, nrhs, bp, ldbp, nm);
        return;
    }

    for_each_group<V>(
        nm,
        [&](Int g) {
            const auto A = make_view<T, V, Int>(ap + g * str_a, rowmajor, ldap);
            gels_compact_group<T, V, Int>(
                overdet, p, q, nrhs, tall ? A : A.transposed(), taup + g * str_t,
                make_view<T, V, Int>(bp + g * str_b, rowmajor, ldbp));
        },
        /* ~geqr2 + apply-Q + substitution flops per group */
        (2.0 * p * q * q + 4.0 * p * q * nrhs + 1.0 * q * q * nrhs) * V);
}

} /* namespace cbk::detail */

CBK_KERNEL_END

#endif /* CBK_GELS_COMPACT_HPP */
