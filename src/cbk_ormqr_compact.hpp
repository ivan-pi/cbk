/* cbk_ormqr_compact.hpp
 *
 * Compact (interleaved-batch) application of Householder reflectors,
 * templated on scalar type T and interleave width V:
 *
 *     C := op(Q) * C  (side='L')  or  C := C * op(Q)  (side='R'),
 *     op(Q) = Q ('N') or Q^T ('T').
 *
 * The missing mkl_?ormqr_compact, in portable form. Also home of larf(), the
 * one-reflector update that ?geqrf_compact reuses for its trailing columns.
 *
 * Applying the implicit Q:
 *   ?geqrf never forms Q. It returns Q as a product of k Householder reflectors
 *   plus the scalars tau(0..k-1):
 *
 *       Q = H(0) H(1) ... H(k-1),   H(kk) = I - tau(kk) * v(kk) * v(kk)^T,
 *
 *   with v(kk) unit-lower: an implicit 1 in position kk, the sub-diagonal
 *   entries stored in column kk of A, zeros above. Applying one reflector to a
 *   slice c of C (a column for side='L', a row for side='R') is a rank-1 update
 *   in two passes over i = kk+1 .. len-1 (this is LAPACK dorm2r / dlarf):
 *       w  = v^T c = c[kk] + sum_i A(i,kk) c[i]
 *       c := c - tau w v   =>   c[kk] -= tau w;  c[i] -= A(i,kk) tau w.
 *
 *   Each H is symmetric, so Q^T = H(k-1)..H(0), and the reflectors are applied
 *   in the order the product dictates (the LAPACK DORMQR table):
 *
 *       side   op(Q)      expanded                 sweep over kk
 *       'L'    Q  * C     H(0)..H(k-1) * C         descending (Backward)
 *       'L'    Q^T* C     H(k-1)..H(0) * C         ascending  (Forward)
 *       'R'    C * Q      C * H(0)..H(k-1)         ascending  (Forward)
 *       'R'    C * Q^T    C * H(k-1)..H(0)         descending (Backward)
 *
 *   For side='R' the dot/axpy run over the rows of C rather than its columns;
 *   the kernel sees that purely as C^T's BatchView, so the arithmetic is shared.
 *
 * Compact storage (matches mkl_?gepack_compact); group g = idx/V, slot v = idx%V:
 *     A_v(i,j)  = ap [ g*ldap*k*V   + (j*ldap + i)*V + v ]   (column-major)
 *     tau_v(kk) = taup[ g*k*V       +  kk*V          + v ]
 *     C_v(i,j)  = cp [ g*ldcp*n*V   + (j*ldcp + i)*V + v ]
 * A is the (ldap, k) reflector batch, exactly as LAPACK ?ormqr declares it, so
 * its per-matrix column extent -- hence the group stride -- is k.
 *
 * Assisted-by: Claude:claude-fable-5 Claude:claude-opus-4-8
 */

#ifndef CBK_ORMQR_COMPACT_HPP
#define CBK_ORMQR_COMPACT_HPP

#include "cbk_common.hpp"

#include <cstddef>
#include <cassert>
#include <type_traits>

CBK_KERNEL_BEGIN

namespace cbk::detail {

/* Reflector sweep order: Forward applies kk = 0..k-1, Backward k-1..0. */
enum class Direction { Forward, Backward };

/* Apply H(kk) to the JB adjacent slices p0 .. p0+JB-1 of C (slices run along
 * C's second index; the reflector along its first, i = kk+1 .. len-1). JB is a
 * compile-time constant so w[] lives in registers and the c-loops unroll; every
 * reflector load A(i,kk) is then reused JB times, which is the point of the
 * blocking. tau is read here from its buffer, not passed as a pack (the
 * alignment note in cbk_common.hpp). Splitting the dot products over several
 * rows' accumulators measured slower, not faster: the loop is load-bound. */
template <int JB, typename T, int V, typename Int>
inline void larf_block(Int kk, Int len, ConstBatchView<T, V, Int> A, const T *tau,
                       BatchView<T, V, Int> C, Int p0)
{
    using VT = typename pack<T, V>::type;
    VT w[JB];
    CBK_UNROLL
    for (int c = 0; c < JB; ++c)
        w[c] = C(kk, p0 + c);
    for (Int i = kk + 1; i < len; ++i) {
        const VT av = A(i, kk);
        CBK_UNROLL
        for (int c = 0; c < JB; ++c)
            w[c] += av * C(i, p0 + c);
    }
    const VT t = reinterpret_cast<const VT *>(tau)[kk];
    CBK_UNROLL
    for (int c = 0; c < JB; ++c) {
        w[c] *= t; /* fold tau into w */
        C(kk, p0 + c) -= w[c];
    }
    for (Int i = kk + 1; i < len; ++i) {
        const VT av = A(i, kk);
        CBK_UNROLL
        for (int c = 0; c < JB; ++c)
            C(i, p0 + c) -= av * w[c];
    }
}

/* Apply H(kk) (column kk of A, scalars tau) to slices [p0, p1) of C, four at
 * a time, the leftover slices as one narrower block. */
template <typename T, int V, typename Int>
inline void larf(Int kk, Int len, ConstBatchView<T, V, Int> A, const T *tau,
                 BatchView<T, V, Int> C, Int p0, Int p1)
{
    Int p = p0;
    for (; p + 4 <= p1; p += 4)
        larf_block<4, T, V>(kk, len, A, tau, C, p);
    for_remainder<4>(p1 - p,
                     [&](auto jb) { larf_block<jb.value, T, V>(kk, len, A, tau, C, p); });
}

/* Block reflectors (compact WY). NB consecutive reflectors H(k0) ..
 * H(k0+NB-1) are applied together as I - Y Tf Y^T (LAPACK dlarft/dlarfb), Y
 * the unit-lower columns k0 .. k0+NB-1 of A from row k0 and Tf the NB x NB
 * upper-triangular factor. C is then read and written once per NB reflectors,
 * in register tiles of NB reflectors x JB slices, instead of once per
 * reflector. The tile, NB * JB accumulators plus NB packs of Y, is sized to
 * the register file like trsm's (4 x 6 with 32 registers, 4 x 2 with 16);
 * 4 x 4 and 8 x 2 measured within a few percent of 4 x 6, 2 x 8 well
 * behind. */
#ifndef CBK_LARFB_NB
#define CBK_LARFB_NB 4
#endif
#ifndef CBK_LARFB_JB
#define CBK_LARFB_JB (simd_registers >= 32 ? 6 : 2)
#endif
constexpr int larfb_nb = CBK_LARFB_NB;
constexpr int larfb_jb = CBK_LARFB_JB;

/* Tf of the block of NB reflectors from column k0 (dlarft 'F', 'C'):
 *     Tf(j,j) = tau(k0+j),  Tf(0:j, j) = -tau(k0+j) Tf(0:j, 0:j) Y(:,0:j)^T y_j
 * The Gram products y_i^T y_j (i < j) are accumulated in one pass down the
 * rows; the top NB rows carry the implicit unit diagonal and zeros above it. */
template <int NB, typename T, int V, typename Int>
inline void larft(Int k0, Int len, ConstBatchView<T, V, Int> A, const T *tau_,
                  BatchView<T, V, Int> Tf)
{
    using VT = typename pack<T, V>::type;
    const VT *tau = reinterpret_cast<const VT *>(tau_);
    VT g[NB][NB] = {};
    CBK_UNROLL
    for (int t = 1; t < NB; ++t) { /* row k0+t: y_t = 1, y_q = A(.,k0+q) for q < t */
        VT a[NB];
        CBK_UNROLL
        for (int q = 0; q < t; ++q)
            a[q] = A(k0 + t, k0 + q);
        CBK_UNROLL
        for (int j = 1; j <= t; ++j) {
            CBK_UNROLL
            for (int i = 0; i < j; ++i)
                g[i][j] += j == t ? a[i] : a[i] * a[j];
        }
    }
    for (Int r = k0 + NB; r < len; ++r) {
        VT a[NB];
        CBK_UNROLL
        for (int q = 0; q < NB; ++q)
            a[q] = A(r, k0 + q);
        CBK_UNROLL
        for (int j = 1; j < NB; ++j) {
            CBK_UNROLL
            for (int i = 0; i < j; ++i)
                g[i][j] += a[i] * a[j];
        }
    }
    VT tm[NB][NB] = {};
    CBK_UNROLL
    for (int j = 0; j < NB; ++j) {
        const VT tj = tau[k0 + j];
        tm[j][j] = tj;
        CBK_UNROLL
        for (int i = 0; i < j; ++i) {
            VT z = VT{};
            CBK_UNROLL
            for (int l = i; l < j; ++l)
                z += tm[i][l] * g[l][j];
            tm[i][j] = -(tj * z);
        }
    }
    CBK_UNROLL
    for (int j = 0; j < NB; ++j) {
        CBK_UNROLL
        for (int i = 0; i <= j; ++i)
            Tf(i, j) = tm[i][j];
    }
}

/* Apply the block reflector of the NB reflectors from column k0 to the JB
 * slices p0 .. p0+JB-1 of C: W = Y^T C, W := op(Tf) W, C -= Y W. Forward
 * (the sweep H(k0) first, i.e. Qb^T C) takes op = transpose, Backward (Qb C)
 * Tf itself. W stays in registers; each row of Y is loaded once per tile and
 * feeds NB * JB FMAs in each pass. */
template <int NB, int JB, bool Fwd, typename T, int V, typename Int>
inline void larfb_tile(Int k0, Int len, ConstBatchView<T, V, Int> A,
                       ConstBatchView<T, V, Int> Tf, BatchView<T, V, Int> C, Int p0)
{
    using VT = typename pack<T, V>::type;
    VT w[NB][JB];
    CBK_UNROLL
    for (int t = 0; t < NB; ++t) { /* the top rows: unit diagonal, zeros above */
        VT cv[JB];
        CBK_UNROLL
        for (int c = 0; c < JB; ++c)
            w[t][c] = cv[c] = C(k0 + t, p0 + c);
        CBK_UNROLL
        for (int q = 0; q < t; ++q) {
            const VT a = A(k0 + t, k0 + q);
            CBK_UNROLL
            for (int c = 0; c < JB; ++c)
                w[q][c] += a * cv[c];
        }
    }
    for (Int r = k0 + NB; r < len; ++r) {
        VT a[NB];
        CBK_UNROLL
        for (int q = 0; q < NB; ++q)
            a[q] = A(r, k0 + q);
        CBK_UNROLL
        for (int c = 0; c < JB; ++c) {
            const VT cv = C(r, p0 + c);
            CBK_UNROLL
            for (int q = 0; q < NB; ++q)
                w[q][c] += a[q] * cv;
        }
    }
    /* W := Tf^T W (descending, in place) or Tf W (ascending, in place) */
    CBK_UNROLL
    for (int s = 0; s < NB; ++s) {
        const int q = Fwd ? NB - 1 - s : s;
        const VT d = Tf(q, q);
        CBK_UNROLL
        for (int c = 0; c < JB; ++c)
            w[q][c] *= d;
        CBK_UNROLL
        for (int l = 0; l < NB; ++l) {
            if (Fwd ? l >= q : l <= q) continue;
            const VT tq = Fwd ? Tf(l, q) : Tf(q, l);
            CBK_UNROLL
            for (int c = 0; c < JB; ++c)
                w[q][c] += tq * w[l][c];
        }
    }
    CBK_UNROLL
    for (int t = 0; t < NB; ++t) {
        VT a[NB];
        CBK_UNROLL
        for (int q = 0; q < t; ++q)
            a[q] = A(k0 + t, k0 + q);
        CBK_UNROLL
        for (int c = 0; c < JB; ++c) {
            VT cv = C(k0 + t, p0 + c) - w[t][c];
            CBK_UNROLL
            for (int q = 0; q < t; ++q)
                cv -= a[q] * w[q][c];
            C(k0 + t, p0 + c) = cv;
        }
    }
    for (Int r = k0 + NB; r < len; ++r) {
        VT a[NB];
        CBK_UNROLL
        for (int q = 0; q < NB; ++q)
            a[q] = A(r, k0 + q);
        CBK_UNROLL
        for (int c = 0; c < JB; ++c) {
            VT cv = C(r, p0 + c);
            CBK_UNROLL
            for (int q = 0; q < NB; ++q)
                cv -= a[q] * w[q][c];
            C(r, p0 + c) = cv;
        }
    }
}

/* Apply the NB reflectors from column k0 to the slices [p0, p1) of C as one
 * block reflector: Tf, then larfb_jb-slice tiles, the leftover slices as one
 * narrower tile. Tf lives in a local buffer of plain T, read through a view. */
template <int NB, bool Fwd, typename T, int V, typename Int>
inline void larfb(Int k0, Int len, ConstBatchView<T, V, Int> A, const T *tau,
                  BatchView<T, V, Int> C, Int p0, Int p1)
{
    using VT = typename pack<T, V>::type;
    alignas(64) T tbuf[NB * NB * V];
    const BatchView<T, V, Int> Tf{reinterpret_cast<VT *>(tbuf), Int(1), Int(NB)};
    larft<NB, T, V, Int>(k0, len, A, tau, Tf);
    const auto Tc = Tf.as_const();
    Int p = p0;
    for (; p + larfb_jb <= p1; p += larfb_jb)
        larfb_tile<NB, larfb_jb, Fwd, T, V, Int>(k0, len, A, Tc, C, p);
    for_remainder<larfb_jb>(p1 - p, [&](auto jb) {
        larfb_tile<NB, jb.value, Fwd, T, V, Int>(k0, len, A, Tc, C, p);
    });
}

/* The reflectors k0 .. k0+nb-1, 1 <= nb <= larfb_nb, as one block reflector. */
template <bool Fwd, typename T, int V, typename Int>
inline void larfb_any(Int nb, Int k0, Int len, ConstBatchView<T, V, Int> A, const T *tau,
                      BatchView<T, V, Int> C, Int p0, Int p1)
{
    assert(nb >= 1 && nb <= larfb_nb);
    if (nb == larfb_nb)
        larfb<larfb_nb, Fwd, T, V, Int>(k0, len, A, tau, C, p0, p1);
    else
        for_remainder<larfb_nb>(nb, [&](auto q) {
            larfb<q.value, Fwd, T, V, Int>(k0, len, A, tau, C, p0, p1);
        });
}

/* One group: sweep the k reflectors over all `npanel` slices of C, larfb_nb
 * at a time as block reflectors (the last block narrower). A single slice
 * gains nothing from a block (Tf is not amortized) and takes the reflectors
 * one at a time, through the one-slice block directly (the remainder dispatch
 * of larf measured slower there). A is the (len x k) reflector view; C is viewed with the
 * reflector axis first. */
template <typename T, int V, typename Int = int>
void ormqr_compact_group(Direction dir, Int len, Int npanel, Int k,
                         ConstBatchView<T, V, Int> A, const T *tau_,
                         BatchView<T, V, Int> C)
{
    static_assert(std::is_floating_point_v<T>,
                  "ormqr_compact is defined for real float/double");
    assert(k <= len && A.si && A.sj && C.si && C.sj);

    if (npanel < 2) {
        for (Int s = 0; s < k; ++s) {
            const Int kk = (dir == Direction::Forward) ? s : k - 1 - s;
            if (npanel == 1) larf_block<1, T, V>(kk, len, A, tau_, C, Int(0));
        }
        return;
    }
    const Int nblk = (k + larfb_nb - 1) / larfb_nb;
    for (Int s = 0; s < nblk; ++s) {
        const Int k0 = (dir == Direction::Forward ? s : nblk - 1 - s) * larfb_nb;
        const Int nb = k - k0 < larfb_nb ? k - k0 : Int(larfb_nb);
        if (dir == Direction::Forward)
            larfb_any<true, T, V, Int>(nb, k0, len, A, tau_, C, Int(0), npanel);
        else
            larfb_any<false, T, V, Int>(nb, k0, len, A, tau_, C, Int(0), npanel);
    }
}

/* All groups: C (m x n) := op(Q) C or C op(Q), either layout. A is the
 * (len x k) reflector batch, len = m (side='L') or n (side='R'), declared
 * (ldap, k) as in LAPACK ?ormqr. trans is 'N', or 'T'/'C' for the transpose.
 * A padded partial last group is processed too, harmlessly: its identity
 * factors have tau = 0. */
template <typename T, int V, typename Int = int>
void ormqr_compact(bool left, bool rowmajor, char trans, Int m, Int n, Int k, const T *ap,
                   Int ldap, const T *taup, T *cp, Int ldcp, Int nm)
{
    assert(nm >= 1 && m >= 0 && n >= 0 && k >= 0);
    assert(ldap >= (rowmajor ? k : (left ? m : n)) && ldcp >= (rowmajor ? n : m));

    const bool tran = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');
    const Direction dir = (left == tran) ? Direction::Forward : Direction::Backward;
    const Int len = left ? m : n;
    const Int npanel = left ? n : m;

    const std::size_t str_a = group_stride(rowmajor, ldap, len, k, V);
    const std::size_t str_t = (std::size_t)k * V;
    const std::size_t str_c = group_stride(rowmajor, ldcp, m, n, V);

    for_each_group<V>(nm, [&](Int g) {
        /* A is swept down its columns. For side='R' the reflectors act on the
         * rows of C, so the kernel is handed C^T. */
        auto C = make_view<T, V, Int>(cp + g * str_c, rowmajor, ldcp);
        if (!left) C = C.transposed();
        ormqr_compact_group<T, V, Int>(
            dir, len, npanel, k,
            make_const_view<T, V, Int>(ap + g * str_a, rowmajor, ldap), taup + g * str_t,
            C);
    });
}

} /* namespace cbk::detail */

CBK_KERNEL_END

#endif /* CBK_ORMQR_COMPACT_HPP */
