/* cqr_orgqr_compact.hpp
 *
 * Compact (interleaved-batch) generation of the explicit orthogonal factor,
 * templated on scalar type T and interleave width V:
 *
 *     Q = H(0) H(1) ... H(k-1),   H(kk) = I - tau(kk) * v(kk) * v(kk)^T,
 *
 * forming the first n columns of the m x m Q (m >= n >= k) in place over the
 * reflectors and scalars a compact QR (?geqrf_compact) left behind. The
 * missing mkl_?orgqr_compact, in portable form -- the form-Q counterpart of
 * ormqr's apply-Q.
 *
 * The algorithm is the unblocked LAPACK org2r, run V matrices at a time.
 * Backward accumulation is what exploits the structure: columns k..n-1 are
 * seeded as unit vectors e_j, then for kk = k-1 down to 0
 *   1. H(kk) is applied to the trailing columns kk+1..n-1 only (larf, the
 *      one-reflector update shared with ormqr and geqrf) -- rows above kk of
 *      those columns are still zero and stay untouched;
 *   2. column kk is formed sweep-free as H(kk) e_kk: the reflector body
 *      scaled by -tau(kk) in place, 1 - tau(kk) on the diagonal, zeros above.
 * A tau(kk) = 0 lane (an already-triangular column, or a padded identity
 * slot) reproduces e_kk exactly, so the whole update is branch-free.
 *
 * Compact storage (matches mkl_?gepack_compact); group g = idx/V, slot v = idx%V:
 *     A_v(i,j)  = ap [ g*ldap*n*V + (j*ldap + i)*V + v ]   (column-major)
 *     tau_v(kk) = taup[ g*k*V      +  kk*V          + v ]
 * A is packed with its n output columns (group stride ldap*n*V); on entry only
 * columns 0..k-1 below the diagonal are read.
 */

#ifndef CQR_ORGQR_COMPACT_HPP
#define CQR_ORGQR_COMPACT_HPP

#include "cqr_compact_common.hpp"
#include "cqr_ormqr_compact.hpp" /* larf */

#include <cstddef>
#include <cassert>
#include <type_traits>

namespace cqr::detail {

/* One group of V interleaved matrices, through the layout-agnostic view:
 * A (m x n, k reflector columns) is overwritten with the first n columns
 * of Q. */
template <typename T, int V, typename Int = int>
void orgqr_compact_group(Int m, Int n, Int k, BatchView<T, V, Int> A, const T *tau_)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point_v<T>,
                  "orgqr_compact is defined for real float/double");
    assert(k <= n && n <= m && A.si && A.sj);

    const VT *tau = reinterpret_cast<const VT *>(tau_);
    const auto Ac = A.as_const();
    VT one;
    broadcast<T, V>(one, T(1));

    /* unit-seed columns k..n-1: every reflector is still to come and reaches
     * them only through the larf below */
    for (Int j = k; j < n; ++j) {
        for (Int i = 0; i < m; ++i)
            A(i, j) = VT{};
        A(j, j) = one;
    }

    for (Int kk = k - 1; kk >= 0; --kk) { /* backward: H(k-1) first, H(0) last */
        const VT t = tau[kk];
        /* apply H(kk) to the trailing columns kk+1 .. n-1 (rows above kk of
         * those columns are still zero and stay untouched: larf reads and
         * writes rows kk..m-1 only) */
        larf<T, V>(kk, m, Ac, t, A, kk + 1, n);
        /* form column kk = H(kk) e_kk, sweep-free: the reflector body scaled
         * by -tau in place, 1 - tau on the diagonal, zeros above. tau = 0 (an
         * already-triangular column, or a padded identity lane) gives e_kk. */
        const VT mt = -t;
        for (Int i = kk + 1; i < m; ++i)
            A(i, kk) = A(i, kk) * mt;
        A(kk, kk) = one - t;
        for (Int i = 0; i < kk; ++i)
            A(i, kk) = VT{};
    }
}

/* All groups, either layout. A padded partial last group is processed too,
 * harmlessly: its identity lanes carry zero reflector bodies and tau = 0, so
 * every step is the identity map and Q = I comes out. */
template <typename T, int V, typename Int = int>
void orgqr_compact(bool rowmajor, Int m, Int n, Int k, T *ap, Int ldap, const T *taup,
                   Int nm)
{
    assert(nm >= 1 && m >= 0 && n >= 0 && k >= 0 && k <= n && n <= m);

    const std::size_t str_a = group_stride(rowmajor, ldap, m, n, V);
    const std::size_t str_t = (std::size_t)k * V;

    /* ~org2r flops per group: sum_j 4(m-j)(n-j) in closed form, plus the
     * unit-seed stores so a k = 0 (or tiny-k) identity fill still gates the
     * threading like zero_compact's does */
    const double flops = (4.0 * m * n * k - 2.0 * (m + n) * (double)k * k +
                          (4.0 / 3.0) * k * k * k + (double)m * (n - k)) *
                         V;

    for_each_group<V>(
        nm,
        [&](Int g) {
            orgqr_compact_group<T, V, Int>(
                m, n, k, make_view<T, V, Int>(ap + g * str_a, rowmajor, ldap),
                taup + g * str_t);
        },
        flops);
}

} /* namespace cqr::detail */

#endif /* CQR_ORGQR_COMPACT_HPP */
