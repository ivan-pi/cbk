/* cbk_geqrf_compact.hpp
 *
 * Compact (interleaved-batch) QR factorization, templated on scalar type T and
 * interleave width V:
 *
 *     A = Q R,   Q = H(0) H(1) ... H(k-1),   k = min(m, n),
 *     H(kk) = I - tau(kk) * v(kk) * v(kk)^T.
 *
 * A portable, vectorized mkl_?geqrf_compact. The algorithm is the unblocked
 * LAPACK geqr2 -- dlarfg to build each reflector, dlarf (the larf() shared with
 * ormqr) to apply it to the trailing columns -- run V matrices at a time: the
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
#include <limits>
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
CBK_ALWAYS_INLINE void vselect(typename pack<T, V>::type &r, const mask_t<T, V> &mask,
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
CBK_ALWAYS_INLINE void
larfg_pack(const typename pack<T, V>::type &x0, const typename pack<T, V>::type &tail,
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

/* True when every lane of a comparison mask is set. */
template <typename T, int V>
CBK_ALWAYS_INLINE bool all_lanes(const mask_t<T, V> &mask) noexcept
{
    typename int_bits<T>::type all = -1;
    for (int v = 0; v < V; ++v)
        all &= mask[v];
    return all != 0;
}

/* r := |x|, lane-wise. */
template <typename T, int V>
CBK_ALWAYS_INLINE void vabs(typename pack<T, V>::type &r,
                            const typename pack<T, V>::type &x) noexcept
{
    vselect<T, V>(r, x < 0, -x, x);
}

/* The range in which the plain sum of squares is a faithful norm: the tail
 * sum at least safmin / eps, so whatever its terms lost to underflow is below
 * its rounding error, and the full sum finite. A lane outside it -- a column
 * small or large enough for its squares to leave the floating-point range, or
 * one with nothing below the diagonal -- is one the plain larfg cannot be
 * trusted on: the sum underflowed to zero reads as "already triangular" and
 * returns a finite, wrong R, and the overflowed one returns Inf. */
template <typename T, int V>
CBK_ALWAYS_INLINE void larfg_in_range(mask_t<T, V> &ok,
                                      const typename pack<T, V>::type &x0,
                                      const typename pack<T, V>::type &tail) noexcept
{
    constexpr T tsml = std::numeric_limits<T>::min() / std::numeric_limits<T>::epsilon();
    constexpr T tbig = std::numeric_limits<T>::max();
    ok = (tail >= tsml) & (x0 * x0 + tail <= tbig);
}

/* The reflector of column kk for a pack with a lane out of larfg_in_range:
 * dlarfg's rescaling, lane-wise. Each such lane's column is divided by its
 * largest magnitude (a division, not a reciprocal multiply: the reciprocal of
 * a subnormal overflows), which puts the sum of squares in [1, m]; tau and the
 * reflector body are scale-invariant, and beta is scaled back. A column that
 * is exactly zero below the diagonal -- an already-triangular one, a padded
 * identity lane -- still yields tau = 0 with the diagonal untouched. The lanes
 * that were in range keep the plain path's result, bit for bit, so a lane's
 * factorization never depends on which matrices share its pack.
 *
 * Out of line on purpose (it is the rare path, and the plain one stays small);
 * it takes the view and the indices and loads what it needs, since a pack must
 * not cross a call boundary by reference (cbk_common.hpp). */
template <typename T, int V, typename Int>
__attribute__((noinline)) void geqrf_reflector_scaled(Int kk, Int m,
                                                      BatchView<T, V, Int> A,
                                                      typename pack<T, V>::type *tau)
{
    using VT = typename pack<T, V>::type;
    const VT x0 = A(kk, kk);

    /* the plain path, for the lanes in range */
    VT tail = VT{};
    for (Int i = kk + 1; i < m; ++i)
        tail += A(i, kk) * A(i, kk);
    mask_t<T, V> ok;
    larfg_in_range<T, V>(ok, x0, tail);
    VT rdiag, t, inv;
    larfg_pack<T, V>(x0, tail, rdiag, t, inv);

    /* the scaled path: sc = max |column|, or 1 for an all-zero column */
    VT mx, one;
    broadcast<T, V>(one, T(1));
    vabs<T, V>(mx, x0);
    for (Int i = kk + 1; i < m; ++i) {
        VT a;
        vabs<T, V>(a, A(i, kk));
        vselect<T, V>(mx, a > mx, a, mx);
    }
    VT sc;
    vselect<T, V>(sc, mx > 0, mx, one);
    VT tail_s = VT{};
    for (Int i = kk + 1; i < m; ++i) {
        const VT a = A(i, kk) / sc;
        tail_s += a * a;
    }
    VT rdiag_s, t_s, inv_s;
    larfg_pack<T, V>(x0 / sc, tail_s, rdiag_s, t_s, inv_s);
    /* beta back in the column's scale; with no reflector, x0 itself (exactly) */
    vselect<T, V>(rdiag_s, tail_s > 0, rdiag_s * sc, x0);

    vselect<T, V>(t, ok, t, t_s);
    vselect<T, V>(rdiag, ok, rdiag, rdiag_s);
    tau[kk] = t;
    for (Int i = kk + 1; i < m; ++i) {
        VT body;
        vselect<T, V>(body, ok, A(i, kk) * inv, (A(i, kk) / sc) * inv_s);
        A(i, kk) = body;
    }
    A(kk, kk) = rdiag;
}

/* Build H(kk) from column kk of A, rows kk..m-1: the reflector body scaled in
 * place below the diagonal, the R diagonal in A(kk,kk), the scalar in tau[kk]
 * (the caller reads it back from there for its larf: the result is not handed
 * out through a pack reference, which this function, no longer small enough to
 * be sure of inlining, must not do -- cbk_common.hpp). One step of geqr2. The sum of squares
 * is formed unscaled; a pack with a lane whose sum left the range in which
 * that is safe (larfg_in_range) is redone by geqrf_reflector_scaled. The last
 * column of a square matrix has nothing below the diagonal and no sum to
 * check. */
template <typename T, int V, typename Int>
inline void geqrf_reflector(Int kk, Int m, BatchView<T, V, Int> A,
                            typename pack<T, V>::type *tau) noexcept
{
    using VT = typename pack<T, V>::type;
    VT tail = VT{};
    for (Int i = kk + 1; i < m; ++i)
        tail += A(i, kk) * A(i, kk);
    if (kk + 1 < m) {
        mask_t<T, V> ok;
        larfg_in_range<T, V>(ok, A(kk, kk), tail);
        if (__builtin_expect(!all_lanes<T, V>(ok), 0)) {
            geqrf_reflector_scaled<T, V, Int>(kk, m, A, tau);
            return;
        }
    }
    VT rdiag, t, inv;
    larfg_pack<T, V>(A(kk, kk), tail, rdiag, t, inv);
    tau[kk] = t;
    for (Int i = kk + 1; i < m; ++i)
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

    for (Int kk = 0; kk < k; ++kk) {
        geqrf_reflector<T, V, Int>(kk, m, A, tau);
        /* apply H(kk) to the trailing columns kk+1 .. n-1 */
        /* a local copy, not &tau[kk]: a pointer into the tau buffer may alias
         * the stores to A (the pack is may_alias), and larf reloading it
         * through the trailing update measured markedly slower under gcc */
        const VT t = tau[kk];
        larf<T, V>(kk, m, Ac, &t, A, kk + 1, n);
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
