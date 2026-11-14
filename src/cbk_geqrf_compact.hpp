/* cbk_geqrf_compact.hpp
 *
 * Compact (interleaved-batch) QR factorization, templated on scalar type T and
 * interleave width V:
 *
 *     A = Q R,   Q = H(0) H(1) ... H(k-1),   k = min(m, n),
 *     H(kk) = I - tau(kk) * v(kk) * v(kk)^T.
 *
 * A portable, vectorized mkl_?geqrf_compact. The algorithm is LAPACK's blocked
 * geqrf with panels of larfb_nb columns: geqr2 on the panel -- dlarfg to build
 * each reflector, dlarf (the larf() shared with ormqr) to apply it to the
 * panel's remaining columns -- then the panel's reflectors applied to the
 * trailing columns as one block reflector (larfb, also ormqr's), so the
 * trailing matrix is read and written once per panel rather than once per
 * column. All of it runs V matrices at a time: the
 * compact format stores element (i,j) of all V matrices contiguously, so the
 * scalar algorithm lifts with double -> V-wide vector, one lane per matrix. The
 * single data-dependent branch of dlarfg (xnorm == 0 -> tau = 0) becomes a lane
 * mask + select in larfg_pack.
 *
 * Compact storage (matches mkl_?gepack_compact); group g = idx/V, slot v = idx%V:
 *     A_v(i,j)  = ap [ g*ldap*n*V + (j*ldap + i)*V + v ]   (column-major)
 *     tau_v(kk) = taup[ g*k*V      +  kk*V          + v ]
 * On exit each matrix holds, per LAPACK ?geqrf, R (min(m,n) x n upper trapezoid)
 * on and above the diagonal, the Householder vectors v(kk) below it (implicit 1
 * on the diagonal), and taup(0..k-1) the reflector scalars.
 *
 * Assisted-by: Claude:claude-opus-4-8 Claude:claude-fable-5
 */

#ifndef CBK_GEQRF_COMPACT_HPP
#define CBK_GEQRF_COMPACT_HPP

#include "cbk_common.hpp"
#include "cbk_ormqr_compact.hpp" /* larf */

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <type_traits>

CBK_KERNEL_BEGIN

namespace cbk::detail {

/* Same-width signed integer for a floating type: the lane type of the masks the
 * GNU vector relational operators yield, and of the bit-blend below. */
// clang-format off
template <typename T> struct int_bits;
template <> struct int_bits<float>  { using type = std::int32_t; };
template <> struct int_bits<double> { using type = std::int64_t; };
// clang-format on
template <typename T, int V>
using mask_t = typename pack<typename int_bits<T>::type, V>::type;

/* r := mask ? a : b, lane-wise; mask lanes are all-ones or zero, as the vector
 * comparisons produce. The blend runs on the may-alias integer view. */
template <typename T, int V>
inline void vselect(typename pack<T, V>::type &r, const mask_t<T, V> &mask,
                    const typename pack<T, V>::type &a,
                    const typename pack<T, V>::type &b) noexcept
{
    using MT = mask_t<T, V>;
    const MT ai = reinterpret_cast<const MT &>(a);
    const MT bi = reinterpret_cast<const MT &>(b);
    const MT rr = (ai & mask) | (bi & ~mask);
    r = reinterpret_cast<const typename pack<T, V>::type &>(rr);
}

/* Branch-free dlarfg for one pack. Given the diagonal x0 = A(kk,kk) and
 * tail = sum_{i>kk} A(i,kk)^2, write per lane
 *   rdiag -> A(kk,kk) on exit (beta if there is a reflector, else x0),
 *   tau   -> the reflector scalar (0 if the column is already zeroed),
 *   inv   -> 1/(x0 - beta) to scale the reflector body (0 when tau = 0).
 * The mask keys on tail > 0 exactly as dlarfg, so an already-triangular column
 * (and a padded identity lane) yields tau = 0 with x0 unchanged, and no lane
 * ever forms 0/0 or 1/0. */
template <typename T, int V>
inline void larfg_pack(const typename pack<T, V>::type &x0,
                       const typename pack<T, V>::type &tail,
                       typename pack<T, V>::type &rdiag, typename pack<T, V>::type &tau,
                       typename pack<T, V>::type &inv) noexcept
{
    using VT = typename pack<T, V>::type;
    const VT zero = VT{};
    VT norm;
    vsqrt<T, V>(norm, x0 * x0 + tail);
    VT beta;
    vselect<T, V>(beta, x0 >= 0, -norm, norm); /* beta = -copysign(norm, x0) */
    const mask_t<T, V> has = (tail > 0);       /* anything to zero? */
    vselect<T, V>(tau, has, (beta - x0) / beta, zero);
    vselect<T, V>(inv, has, T(1) / (x0 - beta), zero);
    vselect<T, V>(rdiag, has, beta, x0);
}

/* Build H(kk) from column kk of A, rows kk..m-1: the reflector body scaled in
 * place below the diagonal, the R diagonal in A(kk,kk), tau[kk]. One step of
 * geqr2. The sum of squares runs on four accumulators, so it is not one chain
 * of dependent FMAs. */
template <typename T, int V, typename Int>
inline void geqrf_reflector(Int kk, Int m, BatchView<T, V, Int> A,
                            typename pack<T, V>::type *tau) noexcept
{
    using VT = typename pack<T, V>::type;
    VT s[4] = {};
    Int i = kk + 1;
    for (; i + 4 <= m; i += 4) {
        CBK_UNROLL
        for (int u = 0; u < 4; ++u)
            s[u] += A(i + u, kk) * A(i + u, kk);
    }
    for (; i < m; ++i)
        s[0] += A(i, kk) * A(i, kk);
    const VT tail = (s[0] + s[1]) + (s[2] + s[3]);
    VT rdiag, inv, t;
    larfg_pack<T, V>(A(kk, kk), tail, rdiag, t, inv);
    tau[kk] = t;
    for (i = kk + 1; i < m; ++i)
        A(i, kk) = A(i, kk) * inv; /* reflector body */
    A(kk, kk) = rdiag;             /* R diagonal */
}

/* One group of V interleaved m x n matrices, through the layout-agnostic view. */
template <typename T, int V, typename Int = int>
void geqrf_compact_group(Int m, Int n, BatchView<T, V, Int> A, T *tau_)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point_v<T>,
                  "geqrf_compact is defined for real float/double");
    assert(A.si && A.sj);

    VT *tau = reinterpret_cast<VT *>(tau_);
    const auto Ac = A.as_const();
    const Int k = (m < n) ? m : n;

    /* LAPACK's crossover: with two panels or fewer the block reflectors do
     * not pay for their Tf, and the factorization runs unblocked, each
     * reflector applied to every trailing column */
    const bool blocked = k > 2 * larfb_nb;
    const Int nb = blocked ? Int(larfb_nb) : k;
    for (Int k0 = 0; k0 < k; k0 += nb) {
        const Int k1 = k0 + nb < k ? k0 + nb : k;
        const Int pend = blocked ? k1 : n;
        /* the panel k0 .. k1-1 unblocked, each reflector applied to the
         * panel's remaining columns only (to all of them, unblocked) */
        for (Int kk = k0; kk < k1; ++kk) {
            geqrf_reflector<T, V, Int>(kk, m, A, tau);
            larf<T, V>(kk, m, Ac, tau_, A, kk + 1, pend);
        }
        /* the trailing columns k1 .. n-1: the panel's reflectors as one block
         * reflector, H(k1-1) .. H(k0) A = Qb^T A */
        if (blocked && k1 < n)
            larfb_any<true, T, V, Int>(k1 - k0, k0, m, Ac, tau_, A, k1, n);
    }
}

/* All groups, either layout. A padded partial last group is processed too,
 * harmlessly: an identity lane factors to R = I, tau = 0. */
template <typename T, int V, typename Int = int>
void geqrf_compact(bool rowmajor, Int m, Int n, T *ap, Int ldap, T *taup, Int nm)
{
    assert(nm >= 1 && m >= 0 && n >= 0);

    const Int k = (m < n) ? m : n;
    const std::size_t str_a = group_stride(rowmajor, ldap, m, n, V);
    const std::size_t str_t = (std::size_t)k * V;

    for_each_group<V>(
        nm,
        [&](Int g) {
            geqrf_compact_group<T, V, Int>(
                m, n, make_view<T, V, Int>(ap + g * str_a, rowmajor, ldap),
                taup + g * str_t);
        },
        2.0 * m * n * k * V /* ~geqr2 flops per group */);
}

} /* namespace cbk::detail */

CBK_KERNEL_END

#endif /* CBK_GEQRF_COMPACT_HPP */
