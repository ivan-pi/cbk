/* cbk_trsm_compact.hpp
 *
 * Compact (interleaved-batch) triangular solve with multiple right-hand sides,
 * templated on scalar type T and interleave width V:
 *
 *     op(A) X = alpha B   (side='L')   or   X op(A) = alpha B   (side='R'),
 *     op(A) = A ('N') or A^T ('T'),
 *
 * A is upper/lower, unit/non-unit triangular; B is overwritten by X. The compact
 * analogue of BLAS ?trsm and a portable alternative to mkl_?trsm_compact -- the
 * solve that closes the batched QR:
 *
 *     cbk_dgeqrf_compact(A -> H, tau);          // A = Q R
 *     cbk_dormqr_compact('L','T', H, tau, B);   // B := Q^T B
 *     cbk_dtrsm_compact ('L','U','N','N', R, B) // B := R^{-1} Q^T B = X
 *
 * Algorithm: the scalar BLAS ?trsm substitution run V matrices at a time. Compact
 * format stores element (i,j) of all V contiguously, so it lifts verbatim with
 * double -> V-wide vector, one lane per matrix (no data-dependent branch). The
 * tuned side='L' path is templated on the RHS block width and on
 * layout/uplo/trans/diag; from trsm_block_min up it is left-looking -- a block
 * of rows pulls in everything already solved in one long register-tiled
 * reduction and solves its own diagonal block without leaving registers --
 * falling back above trsm_lazy_max_bytes to a right-looking sweep that splits
 * the pivots recursively. Layout is a
 * template parameter of that one body rather than a second kernel, and
 * side='R' reduces to side='L' on the transposed views, so every side/layout
 * combination reaches it; the strided kernel over BatchViews is the fallback
 * for a view with no unit stride (docs/cbk_dtrsm_compact_design.md has the
 * details).
 *
 * Compact storage (matches mkl_?gepack_compact); group g = idx/V, slot v = idx%V,
 * A the order-s (s = m left / n right) triangular batch, B the m x n batch:
 *     A_v(i,j) = ap[g*ldap*s*V + (j*ldap+i)*V + v]   (column-major)
 *     B_v(i,j) = bp[g*ldbp*n*V + (j*ldbp+i)*V + v]
 * Row-major swaps the in-matrix roles to i*ld + j.
 *
 * Assisted-by: Claude:claude-opus-4-8 Claude:claude-fable-5
 */

#ifndef CBK_TRSM_COMPACT_HPP
#define CBK_TRSM_COMPACT_HPP

#include "cbk_common.hpp"

#include <cstddef>
#include <cassert>
#include <type_traits>

CBK_KERNEL_BEGIN

namespace cbk::detail {

/* Element (i,j) of one group's matrix, for the two contiguous layouts: the
 * first index is the contiguous one column-major, the second row-major. It is
 * the whole difference between the two -- the substitution, the blocking, the
 * tile and the reciprocal below are written once and instantiated for both,
 * the same way BatchView carries the layout as strides for the rest of the
 * library. Both indices are compile-time-selected, so neither layout pays for
 * the other. */
template <bool ROW, typename Int> constexpr Int trsm_ix(Int i, Int j, Int ld) noexcept
{
    return ROW ? i * ld + j : j * ld + i;
}

/* Single-column column-oriented (gaxpy) solve, side='L', column-major, op(A)=A:
 * reads A down columns (contiguous), for the 1-column tail where a strided
 * single-column row-dot would prefetch poorly. bj is the column, scaled first. */
template <bool UPPER, bool UNIT, typename T, int V, typename Int>
inline void trsm_axpy_col(Int m, const typename pack<T, V>::type *A, Int ldap,
                          typename pack<T, V>::type *bj,
                          const typename pack<T, V>::type &va)
{
    using VT = typename pack<T, V>::type;
    for (Int i = 0; i < m; ++i)
        bj[i] = bj[i] * va;
    for (Int t = 0; t < m; ++t) {
        const Int kk = UPPER ? m - 1 - t : t;
        const VT *ak = A + kk * ldap;
        /* the reciprocal of trsm_dot_block, so a column's answer does not
         * depend on which block of the 4/2/1 split it landed in */
        const VT xk = UNIT ? bj[kk] : bj[kk] * (T(1) / ak[kk]);
        if (!UNIT) bj[kk] = xk;
        const Int lo = UPPER ? 0 : kk + 1;
        const Int hi = UPPER ? kk : m;
        for (Int i = lo; i < hi; ++i)
            bj[i] -= ak[i] * xk;
    }
}

/* Row-dot solve of a fixed JB-column block, side='L'. JB and the
 * layout/uplo/trans/diag config are compile-time, so the JB accumulators land
 * in registers and the c-loops unroll. j0 is the first of the JB adjacent RHS
 * columns; alpha is folded into the first load. Column-major walks the JB
 * columns with stride ldbp and the pivot rows contiguously, row-major the other
 * way round -- one body, trsm_ix decides. */
template <int JB, bool ROW, bool UPPER, bool TRAN, bool UNIT, typename T, int V,
          typename Int>
inline void trsm_dot_block(Int m, const typename pack<T, V>::type *A, Int ldap,
                           typename pack<T, V>::type *B, Int ldbp, Int j0,
                           const typename pack<T, V>::type &va)
{
    using VT = typename pack<T, V>::type;
    /* back-substitute (sweep rows high -> low) when op(A) is upper-triangular:
     * A upper & no-trans, or A lower & trans (its transpose is upper). */
    constexpr bool back = (UPPER != TRAN);
    for (Int t = 0; t < m; ++t) {
        const Int i = back ? m - 1 - t : t;
        VT w[JB];
        for (int c = 0; c < JB; ++c)
            w[c] = B[trsm_ix<ROW, Int>(i, j0 + c, ldbp)] * va;
        const Int lo = back ? i + 1 : 0;
        const Int hi = back ? m : i;
        for (Int l = lo; l < hi; ++l) {
            /* op(A)(i,l): A(l,i) transposed, A(i,l) not. Column-major reads
             * the transposed one down a column and the plain one across a row;
             * row-major swaps which of the two is the contiguous walk. */
            const VT av = TRAN ? A[trsm_ix<ROW, Int>(l, i, ldap)]
                               : A[trsm_ix<ROW, Int>(i, l, ldap)];
            for (int c = 0; c < JB; ++c)
                w[c] -= av * B[trsm_ix<ROW, Int>(l, j0 + c, ldbp)];
        }
        if (!UNIT) {
            /* One divide per pivot row, not one per (row, column): the divider
             * port is not pipelined, so a divide per RHS column dominated the
             * small orders outright (measured: at n = 8 the divides were the
             * whole kernel at 16 right-hand sides). It costs one extra
             * rounding per solved entry; the suites' test ratios cover it. */
            const VT r = T(1) / A[trsm_ix<ROW, Int>(i, i, ldap)];
            for (int c = 0; c < JB; ++c)
                w[c] = w[c] * r;
        }
        for (int c = 0; c < JB; ++c)
            B[trsm_ix<ROW, Int>(i, j0 + c, ldbp)] = w[c];
    }
}

/* Pivot-block width of the blocked sweep below, and the order from which that
 * sweep is used at all. Under the block width the substitution is the plain
 * row-dot above; above it, a block of pivots is solved and its effect on the
 * rows still to come is applied as one register-tiled update, which is what
 * turns the sweep from a chain of rank-1 passes into a rank-NB one. Below
 * trsm_block_min the blocking costs more in per-block overhead than the
 * tiling saves (measured under both gcc and clang, which agree on where the
 * crossing is). */
#ifndef CBK_TRSM_LEAF
#define CBK_TRSM_LEAF 8
#endif
constexpr int trsm_leaf = CBK_TRSM_LEAF;

/* Above trsm_block_min the sweep is left-looking (trsm_left_lazy) while the
 * panel of already-solved right-hand sides each block re-reads -- m rows by the
 * 4 columns of a block, one pack each -- still fits a first-level cache; past
 * that it goes back to the right-looking sweep, whose working set is bounded by
 * the recursion instead of by m. The crossover is stated in bytes and compared
 * against the pack width, so it lands at the right order for every format
 * rather than at a fixed n: 32 KiB is the conservative first-level size x86 has
 * had for two decades (this machine has 48), which puts it at m = 128 for a
 * 64-byte pack, and measurement bears that out -- left-looking led through 128,
 * tied at 170-192 and lost at 256. Override for an unusual cache. */
#ifndef CBK_TRSM_LAZY_MAX_BYTES
#define CBK_TRSM_LAZY_MAX_BYTES 32768
#endif
constexpr std::size_t trsm_lazy_max_bytes = CBK_TRSM_LAZY_MAX_BYTES;
#ifndef CBK_TRSM_BLOCK_MIN
#define CBK_TRSM_BLOCK_MIN 40
#endif
constexpr int trsm_block_min = CBK_TRSM_BLOCK_MIN;

/* One IB x JB tile of B -- rows i0.., RHS columns j0.. -- updated by the
 * pivots [k0,k1) already solved:  B(i,j) -= sum_l op(A)(i,l) B(l,j).
 * The IB*JB accumulators live in registers, so the IB values of op(A) and JB
 * of B loaded per l feed IB*JB fused multiply-adds -- the register tiling the
 * blocked potrf uses for its trailing update (sytrfnp_syrk_tile), with the
 * same JB-loop shape the compiler unrolls rather than hand-expanded copies. */
template <int IB, int JB, bool ROW, bool TRAN, typename T, int V, typename Int>
inline void trsm_update_tile(const typename pack<T, V>::type *A, Int ldap,
                             typename pack<T, V>::type *B, Int ldbp, Int i0, Int j0,
                             Int k0, Int k1)
{
    using VT = typename pack<T, V>::type;
    VT acc[IB][JB];
    for (int r = 0; r < IB; ++r)
        for (int c = 0; c < JB; ++c)
            acc[r][c] = B[trsm_ix<ROW, Int>(i0 + r, j0 + c, ldbp)];
    for (Int l = k0; l < k1; ++l) {
        VT u[IB], w[JB];
        for (int r = 0; r < IB; ++r)
            u[r] = TRAN ? A[trsm_ix<ROW, Int>(l, i0 + r, ldap)]
                        : A[trsm_ix<ROW, Int>(i0 + r, l, ldap)];
        for (int c = 0; c < JB; ++c)
            w[c] = B[trsm_ix<ROW, Int>(l, j0 + c, ldbp)];
        for (int r = 0; r < IB; ++r)
            for (int c = 0; c < JB; ++c)
                acc[r][c] -= u[r] * w[c];
    }
    for (int r = 0; r < IB; ++r)
        for (int c = 0; c < JB; ++c)
            B[trsm_ix<ROW, Int>(i0 + r, j0 + c, ldbp)] = acc[r][c];
}

/* Rows per update tile. The tile holds IB*JB accumulators plus IB + JB
 * operands live, so it has to fit the target's *architectural* vector
 * registers: 4x4 needs 24, 2x4 needs 14.
 *
 * Which of those is available follows from the pack width, not from any
 * particular CPU. A pack of 64 bytes or more is a zmm, and the only
 * instruction set that has one also has 32 vector registers; a narrower pack
 * may well be running on a machine with 16 (SSE, AVX2), where a 4x4 tile
 * spills -- built for an AVX2 target it lost about 40% at V = 4, 16
 * right-hand sides, order 128. So the tile is chosen by the register file the
 * pack width implies, which is a compile-time property of the template
 * arguments; nothing here is tuned to a micro-architecture, and a machine with
 * 32 registers running a narrow pack simply gets the conservative tile. */
template <typename T, int V> constexpr int trsm_tile_rows = (sizeof(T) * V >= 64) ? 4 : 2;

/* The pivots [k0,k1) applied to every row in [r0,r1) and every RHS column, in
 * IB x 4 tiles with IB x 1 / 1 x 4 / 1 x 1 edges. */
template <bool ROW, bool TRAN, typename T, int V, typename Int>
void trsm_update(Int r0, Int r1, Int n, const typename pack<T, V>::type *A, Int ldap,
                 typename pack<T, V>::type *B, Int ldbp, Int k0, Int k1)
{
    /* RHS columns outermost. The tile reads the solved pivot rows of its own
     * JB columns -- a k1-k0 by JB panel of B -- once per row tile, so that
     * panel wants to stay in L1 while the rows stream past it. Sweeping the
     * rows outermost instead makes the resident set the *whole* k by n panel,
     * which stops fitting as soon as there are many right-hand sides: measured
     * on this update alone, row-outermost collapsed from about 2.5 to 1.4
     * G vector-FMA/s at 16 columns and a 64-pivot block, and to 1.35 at 64
     * columns, while column-outermost held its rate across every shape. */
    constexpr int IB = trsm_tile_rows<T, V>;
    Int j = 0;
    for (; j + 4 <= n; j += 4) {
        Int i = r0;
        for (; i + IB <= r1; i += IB)
            trsm_update_tile<IB, 4, ROW, TRAN, T, V, Int>(A, ldap, B, ldbp, i, j, k0, k1);
        for (; i < r1; ++i)
            trsm_update_tile<1, 4, ROW, TRAN, T, V, Int>(A, ldap, B, ldbp, i, j, k0, k1);
    }
    for (; j < n; ++j) {
        Int i = r0;
        for (; i + IB <= r1; i += IB)
            trsm_update_tile<IB, 1, ROW, TRAN, T, V, Int>(A, ldap, B, ldbp, i, j, k0, k1);
        for (; i < r1; ++i)
            trsm_update_tile<1, 1, ROW, TRAN, T, V, Int>(A, ldap, B, ldbp, i, j, k0, k1);
    }
}

/* Rows per left-looking block: measured the fastest of 2, 3, 4, 5 and 6 at
 * every order and right-hand-side count tried. Not for the reason one would
 * guess -- IB = 4 and 5 hold more accumulators (24 and 29 vectors live against
 * 19) and have the better arithmetic intensity (2.00 and 2.22 fused
 * multiply-adds per load against 1.71), and the disassembly shows they spill
 * nothing, their reduction loops being a clean 30 and 35 instructions. They
 * are simply slower, so what binds is neither register pressure nor the
 * load-to-arithmetic ratio. Where the pack width does not imply AVX-512's 32
 * registers, 2 is the only width that fits 16. */
template <typename T, int V> constexpr int trsm_ll_rows = (sizeof(T) * V >= 64) ? 3 : 2;

/* One left-looking block: the IB x JB corner of X at pivot positions
 * [p0, p0+IB) and RHS columns [j0, j0+JB).
 *
 * The right-looking sweep above walks the pivots forward and pushes each
 * block's contribution onto every row still to come, so a row's accumulator is
 * loaded and stored once per pivot block that reaches it, and each update
 * reduces over only that block. Left-looking inverts it: a row block pulls in
 * everything already solved, in one reduction as long as the rows behind it,
 * and its accumulators never leave registers -- including through the diagonal
 * block, which is solved in place with no memory traffic at all and so needs no
 * separate leaf pass. One accumulator round trip for the whole solve instead of
 * one per pivot block, and the longest reduction the problem allows: the tile's
 * rate rises steeply with reduction length (measured 1.7 G vector-FMA/s at 8,
 * 3.9 at 64), which is what this buys.
 *
 * It is the same arithmetic in the same order as the right-looking sweep, so
 * the two agree bit for bit; only the loop structure differs.
 *
 * Pivot p touches row i = m-1-p when op(A) is upper-triangular and p when it is
 * lower, so a run of pivots is a contiguous run of rows either way -- d0 below
 * is its lowest row index, and the diagonal block is solved in pivot order,
 * which runs down the block when back-substituting. */
template <int IB, int JB, bool ROW, bool UPPER, bool TRAN, bool UNIT, typename T, int V,
          typename Int>
inline void trsm_ll_block(Int p0, Int m, Int j0, const typename pack<T, V>::type *A,
                          Int ldap, typename pack<T, V>::type *B, Int ldbp)
{
    using VT = typename pack<T, V>::type;
    constexpr bool back = (UPPER != TRAN);
    /* row of pivot p0 + r, and the contiguous run the block occupies */
    const Int d0 = back ? m - p0 - IB : p0;
    const auto row = [&](int r) { return back ? d0 + (IB - 1 - r) : d0 + r; };
    /* the pivots already solved: rows [m-p0, m) back-substituting, [0, p0) not */
    const Int q0 = back ? m - p0 : Int(0), q1 = back ? m : p0;

    VT acc[IB][JB];
    for (int r = 0; r < IB; ++r)
        for (int c = 0; c < JB; ++c)
            acc[r][c] = B[trsm_ix<ROW, Int>(row(r), j0 + c, ldbp)];
    for (Int l = q0; l < q1; ++l) {
        VT u[IB], w[JB];
        for (int r = 0; r < IB; ++r)
            u[r] = TRAN ? A[trsm_ix<ROW, Int>(l, row(r), ldap)]
                        : A[trsm_ix<ROW, Int>(row(r), l, ldap)];
        for (int c = 0; c < JB; ++c)
            w[c] = B[trsm_ix<ROW, Int>(l, j0 + c, ldbp)];
        for (int r = 0; r < IB; ++r)
            for (int c = 0; c < JB; ++c)
                acc[r][c] -= u[r] * w[c];
    }
    /* the IB x IB diagonal block, entirely in registers */
    for (int k = 0; k < IB; ++k) {
        const Int ik = row(k);
        if (!UNIT) {
            const VT rcp = T(1) / A[trsm_ix<ROW, Int>(ik, ik, ldap)];
            for (int c = 0; c < JB; ++c)
                acc[k][c] = acc[k][c] * rcp;
        }
        for (int r = k + 1; r < IB; ++r) {
            const Int ir = row(r);
            const VT av = TRAN ? A[trsm_ix<ROW, Int>(ik, ir, ldap)]
                               : A[trsm_ix<ROW, Int>(ir, ik, ldap)];
            for (int c = 0; c < JB; ++c)
                acc[r][c] -= av * acc[k][c];
        }
    }
    for (int r = 0; r < IB; ++r)
        for (int c = 0; c < JB; ++c)
            B[trsm_ix<ROW, Int>(row(r), j0 + c, ldbp)] = acc[r][c];
}

/* The left-looking sweep: RHS columns outermost (as in trsm_update, so the
 * solved panel each block reads stays resident), pivots in blocks of IB with a
 * 1-wide tail. */
template <bool ROW, bool UPPER, bool TRAN, bool UNIT, typename T, int V, typename Int>
void trsm_left_lazy(Int m, Int n, const typename pack<T, V>::type *A, Int ldap,
                    typename pack<T, V>::type *B, Int ldbp)
{
    constexpr int IB = trsm_ll_rows<T, V>;
    Int j = 0;
    for (; j + 4 <= n; j += 4) {
        Int p = 0;
        for (; p + IB <= m; p += IB)
            trsm_ll_block<IB, 4, ROW, UPPER, TRAN, UNIT, T, V, Int>(p, m, j, A, ldap, B,
                                                                    ldbp);
        for (; p < m; ++p)
            trsm_ll_block<1, 4, ROW, UPPER, TRAN, UNIT, T, V, Int>(p, m, j, A, ldap, B,
                                                                   ldbp);
    }
    for (; j < n; ++j) {
        Int p = 0;
        for (; p + IB <= m; p += IB)
            trsm_ll_block<IB, 1, ROW, UPPER, TRAN, UNIT, T, V, Int>(p, m, j, A, ldap, B,
                                                                    ldbp);
        for (; p < m; ++p)
            trsm_ll_block<1, 1, ROW, UPPER, TRAN, UNIT, T, V, Int>(p, m, j, A, ldap, B,
                                                                   ldbp);
    }
}

/* The sweep, split recursively in pivot space.
 *
 * A block of pivots whose own triangle is solved by the row-dot costs
 * m * NB / 2 * n multiply-adds of unblocked work -- NB/m of the total -- while
 * the rest is the register-tiled update. Widening NB makes the updates' running
 * reduction longer (and the tile faster) but the unblocked share bigger; the
 * two pull against each other, and no single width wins (measured: 4, 8, 16 and
 * 32 all trail the recursion at order 128). Halving the range instead gets
 * both: the top split's update reduces over m/2 pivots, and the leaves that
 * are solved unblocked are trsm_leaf wide, a few percent of the work.
 *
 * The sweep runs in *pivot* order p = 0 .. m-1; the row it touches is
 * i = m-1-p when op(A) is upper-triangular (back-substitution) and p when it
 * is lower. Either way a range of pivots is a contiguous run of rows, so one
 * index and a length place the diagonal block, the rows it updates, and the
 * pivots doing the updating.
 *
 * alpha is already folded into B by the caller, so the sweep runs at alpha = 1:
 * each entry of B is read as the right-hand side exactly once before its
 * solution overwrites it, which makes pre-scaling bit-identical to scaling at
 * the point of use. */
template <bool ROW, bool UPPER, bool TRAN, bool UNIT, typename T, int V, typename Int>
void trsm_left_blocked(Int p0, Int p1, Int m, Int n, const typename pack<T, V>::type *A,
                       Int ldap, typename pack<T, V>::type *B, Int ldbp)
{
    using VT = typename pack<T, V>::type;
    constexpr bool back = (UPPER != TRAN);
    const Int len = p1 - p0;
    if (len <= trsm_leaf) {
        /* broadcast here rather than carry a pack across this recursive call:
         * a pack parameter taken by reference is loaded at the natural vector
         * alignment once the call is not inlined (.claude/CLAUDE.md) */
        VT one;
        broadcast<T, V>(one, T(1));
        const Int d0 = back ? m - p1 : p0;
        const VT *Ad = A + trsm_ix<ROW, Int>(d0, d0, ldap);
        VT *Bd = B + trsm_ix<ROW, Int>(d0, Int(0), ldbp);
        Int j = 0;
        for (; j + 4 <= n; j += 4)
            trsm_dot_block<4, ROW, UPPER, TRAN, UNIT, T, V, Int>(len, Ad, ldap, Bd, ldbp,
                                                                 j, one);
        if (n - j >= 2) {
            trsm_dot_block<2, ROW, UPPER, TRAN, UNIT, T, V, Int>(len, Ad, ldap, Bd, ldbp,
                                                                 j, one);
            j += 2;
        }
        if (n - j >= 1)
            trsm_dot_block<1, ROW, UPPER, TRAN, UNIT, T, V, Int>(len, Ad, ldap, Bd, ldbp,
                                                                 j, one);
        return;
    }
    const Int pm = p0 + len / 2;
    trsm_left_blocked<ROW, UPPER, TRAN, UNIT, T, V, Int>(p0, pm, m, n, A, ldap, B, ldbp);
    /* the rows of pivots [pm, p1), updated by the pivots [p0, pm) just solved */
    const Int r0 = back ? m - p1 : pm, r1 = back ? m - pm : p1;
    const Int k0 = back ? m - pm : p0, k1 = back ? m - p0 : pm;
    trsm_update<ROW, TRAN, T, V, Int>(r0, r1, n, A, ldap, B, ldbp, k0, k1);
    trsm_left_blocked<ROW, UPPER, TRAN, UNIT, T, V, Int>(pm, p1, m, n, A, ldap, B, ldbp);
}

/* One group, side='L', column-major, fully specialized on uplo/trans/diag:
 * the RHS columns are swept in 4/2/1 blocks so the 1-3 leftover columns still
 * reuse each A load (2- and 1-wide tails), instead of a one-column-at-a-time
 * remainder. The driver handles alpha = 0 (B := 0); alpha is nonzero here. */
template <bool ROW, bool UPPER, bool TRAN, bool UNIT, typename T, int V, typename Int>
void trsm_left_dot_tb(Int m, Int n, T alpha, const T *a_, Int ldap, T *b_, Int ldbp)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point_v<T>,
                  "trsm_compact is defined for real float/double");
    /* Column-major: A's columns and B's columns are contiguous, so both leading
     * dimensions span the other extent. Row-major: they span the rows. n = 1
     * never steps to a second RHS column, so ldbp is unused then -- the routing
     * in trsm_compact_group reaches here with ldbp = 1 for a one-column
     * row-major B, whose unit row stride is a valid column-major column. */
    assert((ROW ? ldap >= m : ldap >= m) && (n <= 1 || (ROW ? ldbp >= n : ldbp >= m)));

    const VT *A = reinterpret_cast<const VT *>(a_);
    VT *B = reinterpret_cast<VT *>(b_);

    VT va;
    broadcast<T, V>(va, alpha);

    /* The blocked sweep pays when the tile has something to amortize over: a
     * second RHS column, or the transposed sweep, whose unblocked form reduces
     * each row into a single accumulator and runs at FMA latency. A single
     * column of op(A) = A is already the contiguous column-axpy below, and
     * blocking it only adds passes (measured, both compilers). */
    if (m >= trsm_block_min && (TRAN || n >= 2)) {
        if (alpha != T(1))
            for (Int j = 0; j < n; ++j)
                for (Int i = 0; i < m; ++i)
                    B[trsm_ix<ROW, Int>(i, j, ldbp)] =
                        B[trsm_ix<ROW, Int>(i, j, ldbp)] * va;
        if ((std::size_t)m * 4 * V * sizeof(T) <= trsm_lazy_max_bytes)
            trsm_left_lazy<ROW, UPPER, TRAN, UNIT, T, V, Int>(m, n, A, ldap, B, ldbp);
        else
            trsm_left_blocked<ROW, UPPER, TRAN, UNIT, T, V, Int>(Int(0), m, m, n, A, ldap,
                                                                 B, ldbp);
        return;
    }

    Int j = 0;
    for (; j + 4 <= n; j += 4)
        trsm_dot_block<4, ROW, UPPER, TRAN, UNIT, T, V, Int>(m, A, ldap, B, ldbp, j, va);
    if (n - j >= 2) {
        trsm_dot_block<2, ROW, UPPER, TRAN, UNIT, T, V, Int>(m, A, ldap, B, ldbp, j, va);
        j += 2;
    }
    if (n - j >= 1) {
        /* The single leftover column: column-major with op(A)=A would stream A
         * strided with no reuse, so use the contiguous column-axpy instead.
         * op(A)=A^T reads A down a column already, and row-major has no
         * contiguous single column to walk, so both take the dot 1-block. */
        if constexpr (!TRAN && !ROW)
            trsm_axpy_col<UPPER, UNIT, T, V, Int>(m, A, ldap, B + j * ldbp, va);
        else
            trsm_dot_block<1, ROW, UPPER, TRAN, UNIT, T, V, Int>(m, A, ldap, B, ldbp, j,
                                                                 va);
    }
}

/* Runtime (layout, uplo, trans, diag) -> the compile-time-specialized driver. */
template <bool ROW, bool UPPER, bool TRAN, typename T, int V, typename Int>
inline void trsm_left_dot_u(bool unit, Int m, Int n, T alpha, const T *a, Int ldap, T *b,
                            Int ldbp)
{
    if (unit)
        trsm_left_dot_tb<ROW, UPPER, TRAN, true, T, V, Int>(m, n, alpha, a, ldap, b,
                                                            ldbp);
    else
        trsm_left_dot_tb<ROW, UPPER, TRAN, false, T, V, Int>(m, n, alpha, a, ldap, b,
                                                             ldbp);
}
template <bool ROW, bool UPPER, typename T, int V, typename Int>
inline void trsm_left_dot_t(bool tran, bool unit, Int m, Int n, T alpha, const T *a,
                            Int ldap, T *b, Int ldbp)
{
    if (tran)
        trsm_left_dot_u<ROW, UPPER, true, T, V, Int>(unit, m, n, alpha, a, ldap, b, ldbp);
    else
        trsm_left_dot_u<ROW, UPPER, false, T, V, Int>(unit, m, n, alpha, a, ldap, b,
                                                      ldbp);
}
template <bool ROW, typename T, int V, typename Int>
inline void trsm_left_dot(bool upper, bool tran, bool unit, Int m, Int n, T alpha,
                          const T *a, Int ldap, T *b, Int ldbp)
{
    if (upper)
        trsm_left_dot_t<ROW, true, T, V, Int>(tran, unit, m, n, alpha, a, ldap, b, ldbp);
    else
        trsm_left_dot_t<ROW, false, T, V, Int>(tran, unit, m, n, alpha, a, ldap, b, ldbp);
}

/* One group, fully general: any side / layout via BatchView strides. Same
 * substitution as the tuned path; only the addressing changes (it lives in the
 * two BatchViews). side='L' sweeps a row of X at a time, side='R' a column. */
template <typename T, int V, typename Int = int>
void trsm_compact_group_strided(bool left, bool upper, bool tran, bool unit, Int m, Int n,
                                T alpha, ConstBatchView<T, V, Int> A,
                                BatchView<T, V, Int> B)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point_v<T>,
                  "trsm_compact is defined for real float/double");

    assert(A.si && A.sj && B.si && B.sj);

    VT va;
    broadcast<T, V>(va, alpha);

    if (left) {
        /* solve op(A) X = alpha B column by column; A is m x m */
        const bool back = (upper != tran);
        for (Int j = 0; j < n; ++j)
            for (Int t = 0; t < m; ++t) {
                const Int i = back ? m - 1 - t : t;
                VT w = B(i, j) * va;
                const Int lo = back ? i + 1 : 0;
                const Int hi = back ? m : i;
                for (Int l = lo; l < hi; ++l)
                    w -= (tran ? A(l, i) : A(i, l)) * B(l, j);
                B(i, j) = unit ? w : w / A(i, i);
            }
    }
    else {
        /* solve X op(A) = alpha B, one column of X at a time; A is n x n */
        const bool fwd = (upper != tran);
        for (Int t = 0; t < n; ++t) {
            const Int j = fwd ? t : n - 1 - t;
            for (Int i = 0; i < m; ++i)
                B(i, j) = B(i, j) * va; /* scale this column of X */
            const Int lo = fwd ? 0 : j + 1;
            const Int hi = fwd ? j : n;
            for (Int l = lo; l < hi; ++l) {
                const VT c = tran ? A(j, l) : A(l, j);
                for (Int i = 0; i < m; ++i)
                    B(i, j) -= c * B(i, l);
            }
            if (!unit) {
                const VT d = A(j, j);
                for (Int i = 0; i < m; ++i)
                    B(i, j) = B(i, j) / d;
            }
        }
    }
}

/* One group, any side / layout, alpha != 0, through the views. Every operand
 * a view can describe reaches the tuned kernel; the strided kernel stays as the
 * fallback for a view with no unit stride at all.
 *
 * side='R' first: X op(A) = alpha B transposes to op(A)^T X^T = alpha B^T, a
 * *left* solve on the transposed views with the extents swapped. op(A)^T is
 * A^T when transa = 'N' and A when it is 'T', which is exactly op(A^T) for the
 * same transa, and A^T is upper where A is lower -- so the reduction is
 * "transpose both views, swap m and n, flip uplo, keep transa". One identity
 * instead of a second kernel, the same trick that makes ormqr's side='R' and
 * gels's LQ case the kernels they already had.
 *
 * Then the layout: B's rows contiguous is column-major, B's columns contiguous
 * is row-major, and A is either stored the same way or as its transpose (the
 * other unit stride: A is then M^T for the M of leading dimension A's other
 * stride, so uplo and transa flip -- the case of a QR factor reached through a
 * transposed view, as gels's LQ path does). Four branches, one per (layout of
 * B, layout of A); all of them land in the same body, which carries the layout
 * as a template parameter. One branch per group, outside every kernel loop.
 * The per-group entry point the fused solves (gels, and through the pointer
 * form below, the Cholesky and LDL^T solves) compose their sweeps from. */
template <typename T, int V, typename Int = int>
inline void trsm_compact_group(bool left, bool upper, bool tran, bool unit, Int m, Int n,
                               T alpha, ConstBatchView<T, V, Int> A,
                               BatchView<T, V, Int> B)
{
    if (!left) {
        trsm_compact_group<T, V, Int>(true, !upper, tran, unit, n, m, alpha,
                                      A.transposed(), B.transposed());
        return;
    }
    if (B.si == 1 && A.si == 1)
        trsm_left_dot<false, T, V, Int>(upper, tran, unit, m, n, alpha,
                                        reinterpret_cast<const T *>(A.data), A.sj,
                                        reinterpret_cast<T *>(B.data), B.sj);
    else if (B.si == 1 && A.sj == 1)
        trsm_left_dot<false, T, V, Int>(!upper, !tran, unit, m, n, alpha,
                                        reinterpret_cast<const T *>(A.data), A.si,
                                        reinterpret_cast<T *>(B.data), B.sj);
    else if (B.sj == 1 && A.sj == 1)
        trsm_left_dot<true, T, V, Int>(upper, tran, unit, m, n, alpha,
                                       reinterpret_cast<const T *>(A.data), A.si,
                                       reinterpret_cast<T *>(B.data), B.si);
    else if (B.sj == 1 && A.si == 1)
        trsm_left_dot<true, T, V, Int>(!upper, !tran, unit, m, n, alpha,
                                       reinterpret_cast<const T *>(A.data), A.sj,
                                       reinterpret_cast<T *>(B.data), B.si);
    else
        trsm_compact_group_strided<T, V, Int>(left, upper, tran, unit, m, n, alpha, A, B);
}

/* The same on packed pointers in a layout: side='L' column-major reaches the
 * tuned path, the other three side/layout combinations the strided kernel. */
template <typename T, int V, typename Int = int>
inline void trsm_compact_group(bool left, bool upper, bool rowmajor, bool tran, bool unit,
                               Int m, Int n, T alpha, const T *a, Int ldap, T *b,
                               Int ldbp)
{
    trsm_compact_group<T, V, Int>(left, upper, tran, unit, m, n, alpha,
                                  make_const_view<T, V, Int>(a, rowmajor, ldap),
                                  make_view<T, V, Int>(b, rowmajor, ldbp));
}

/* All groups (nm matrices). A padded partial last group is processed too, which
 * is harmless: padded slots are identity triangular factors (unit diagonal), so
 * the diagonal divide never hits zero and their X = alpha B is never read back. */
template <typename T, int V, typename Int = int>
void trsm_compact(bool left, bool upper, bool rowmajor, bool tran, bool unit, Int m,
                  Int n, T alpha, const T *ap, Int ldap, T *bp, Int ldbp, Int nm)
{
    assert(nm >= 1 && m >= 0 && n >= 0);

    /* A is the order-s triangular factor: s = m (left) or n (right). */
    const Int s = left ? m : n;

    /* A is s x s, B is m x n. */
    const std::size_t str_a = group_stride(rowmajor, ldap, s, s, V);
    const std::size_t str_b = group_stride(rowmajor, ldbp, m, n, V);

    /* alpha == 0 is the BLAS ?trsm fast path: B := 0 with A untouched. Handle it
     * once here -- both group kernels then assume alpha != 0. */
    if (alpha == T(0)) {
        zero_compact<T, V, Int>(rowmajor, m, n, bp, ldbp, nm);
        return;
    }

    for_each_group<V>(
        nm,
        [&](Int g) {
            trsm_compact_group<T, V, Int>(left, upper, rowmajor, tran, unit, m, n, alpha,
                                          ap + (std::size_t)g * str_a, ldap,
                                          bp + (std::size_t)g * str_b, ldbp);
        },
        (double)s * s * (left ? n : m) * V /* ~substitution flops per group */);
}

} /* namespace cbk::detail */

CBK_KERNEL_END

#endif /* CBK_TRSM_COMPACT_HPP */
