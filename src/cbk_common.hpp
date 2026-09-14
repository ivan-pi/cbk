/* cbk_common.hpp
 *
 * Shared machinery for the compact (interleaved-batch) kernels:
 *
 *   pack<T,V>             -- the V-wide SIMD element (a GNU vector type).
 *   vsqrt / broadcast     -- lane-wise helpers (geqrf/potrf pivots, trsm's alpha).
 *   BatchView             -- a strided 2-D view of one group of V interleaved
 *                            matrices; every kernel addresses its operands
 *                            through it, so one kernel serves every layout.
 *   make_view / make_const_view -- view a packed T buffer for a layout and ld.
 *   make_lower_view       -- the view whose lower triangle is the named (uplo)
 *                            triangle of a symmetric batch.
 *   group_stride          -- scalars per group of V interleaved matrices.
 *   for_vlen              -- runtime interleave width -> compile-time V.
 *   for_each_group        -- the loop over groups, threaded with OpenMP.
 *   zero_block / zero_compact -- B := 0 for a block of one group / a whole batch.
 *
 * V is the compact-format interleave width (the number of matrices whose
 * element (i,j) is stored contiguously). It does NOT need to match the hardware
 * vector width:
 *   x86:   V*sizeof(T) = 16/32/64 bytes maps exactly to XMM/YMM/ZMM.
 *   NEON:  128-bit registers; V=4 or V=8 doubles lower to short unrolled bursts
 *          of 2/4 independent fmla v*.2d chains, which wide cores execute well.
 *   SVE:   compile fixed-width with -msve-vector-bits=512 on A64FX to map V=8
 *          doubles onto one SVE register.
 *
 * Assisted-by: Claude:claude-fable-5 Claude:claude-opus-4-8
 */

#ifndef CBK_COMMON_HPP
#define CBK_COMMON_HPP

#include <cstddef>
#include <cmath>
#include <limits>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

/* CBK_KERNEL_BEGIN / CBK_KERNEL_END wrap every kernel header. Under clang
 * (icpx included) they mark each function declared between them
 * min_vector_width(512): clang otherwise legalizes vector operations to its
 * preferred width, 256 bits on AVX-512 CPUs, splitting a 512-bit pack into two
 * ymm halves -- half the FMA rate, twice the register pressure. 512 is a floor,
 * not a widening: narrower packs and other targets are unaffected. GCC lowers
 * explicit vector types at their natural width and needs nothing. Details and
 * the measurement: .claude/CLAUDE.md, docs/building.md. */

#if defined(__clang__) && defined(__x86_64__)
// clang-format off
#define CBK_KERNEL_BEGIN _Pragma("clang attribute push (__attribute__((min_vector_width(512))), apply_to = function)")
#define CBK_KERNEL_END _Pragma("clang attribute pop")
// clang-format on
#else
#define CBK_KERNEL_BEGIN
#define CBK_KERNEL_END
#endif

CBK_KERNEL_BEGIN

namespace cbk::detail {

/* ------------------------------------------------------------------ */
/* pack<T,V>::type : the V-wide SIMD element                          */
/* ------------------------------------------------------------------ */

/* The V-wide element is a GNU vector type, available wherever the compiler
 * provides the vector_size and may_alias attributes (GCC, Clang, Intel icpx),
 * under strict -std=c++17 as well as GNU mode. */
#if defined(__has_attribute)
#if __has_attribute(vector_size) && __has_attribute(__may_alias__)
#define CBK_HAS_GNU_VECTORS 1
#endif
#endif

#if defined(CBK_HAS_GNU_VECTORS)

template <typename T, int V> struct pack {
    /* GNU vector_size needs a power-of-two byte width; the supported interleave
     * widths are 2/4/8/16, matching the C API. Checked here, the single
     * chokepoint, so a bad width fails with this message. */
    static_assert(V == 2 || V == 4 || V == 8 || V == 16,
                  "interleave width V must be 2, 4, 8, or 16");
    /* aligned(alignof(T)) makes the type valid on any T-aligned buffer
     * (unaligned vector loads are free on all modern hardware); may_alias
     * exempts it from strict-aliasing violations when viewing a plain T array. */
    using type
        __attribute__((vector_size(V * sizeof(T)), aligned(alignof(T)), may_alias)) = T;
};

#else
#error "cbk compact kernels require the GNU vector extensions " \
       "(__attribute__((vector_size)) with may_alias); compile with GCC, Clang, " \
       "or Intel icpx."
#endif

/* ------------------------------------------------------------------ */
/* Lane-wise vector helpers.                                           */
/*                                                                     */
/* Vectors are passed by reference: passing a GNU vector by value would */
/* commit the base-ISA vector argument ABI without -march, which GCC   */
/* and Clang flag via -Wpsabi. Once inlined the codegen is identical.  */
/* Results are written through an out-parameter (named first). These   */
/* helpers are small enough to always inline; a pack passed by         */
/* reference across a call clang does NOT inline is loaded there with   */
/* the natural vector alignment (the typedef's relaxed alignment is    */
/* lost on the referent) and faults on an under-aligned buffer or      */
/* local -- larger helpers re-read the value from the view instead.    */
/* ------------------------------------------------------------------ */

/* r := sqrt(x), lane-wise. GCC and clang lower the loop to one vsqrt*
 * instruction under -fno-math-errno, which the library's CMakeLists sets
 * (docs/building.md); with math errno on, each lane is a guarded scalar sqrt
 * plus a libm call. The kernels never read errno. */
template <typename T, int V>
inline void vsqrt(typename pack<T, V>::type &r,
                  const typename pack<T, V>::type &x) noexcept
{
    for (int v = 0; v < V; ++v)
        r[v] = std::sqrt(x[v]);
}

/* v := x broadcast to all V lanes. GNU vector types broadcast a scalar in
 * arithmetic but not in assignment; `x - VT{}` subtracts an all-zero vector,
 * leaving x in every lane (and, unlike `VT{} + x`, preserving a zero's sign). */
template <typename T, int V>
inline void broadcast(typename pack<T, V>::type &v, T x) noexcept
{
    v = x - typename pack<T, V>::type{};
}

/* ------------------------------------------------------------------ */
/* BatchView: a strided 2-D view of one group of V interleaved matrices*/
/*                                                                     */
/* A(i,j) = data[i*si + j*sj], addressing element (i,j) of all V       */
/* matrices of the group at once (the element type is the V-wide pack; */
/* strides are in packs). A column-major group has si = 1, sj = ldap;  */
/* row-major swaps them. Because the kernels only ever go through this  */
/* view, one kernel per routine covers every layout -- and ormqr's     */
/* side='R' is just the view of C^T. Offsets are formed in Int (the    */
/* batch and matrix dimensions fit int), which keeps the strided sweep  */
/* vectorizable; the per-group base offset, which can exceed Int, is   */
/* applied to the pointer before the view is built.                    */
/*                                                                     */
/* The view is templated on (T, V), not on the pack type, on purpose:  */
/* the pack's relaxed aligned(alignof(T)) lives on its typedef, and     */
/* clang strips typedef alignment from class template *arguments* --   */
/* a BatchView<pack<T,V>::type> would see a naturally aligned vector   */
/* and emit aligned loads/stores that fault on 16-byte-aligned buffers  */
/* (GCC keeps the typedef alignment either way). Naming the typedef    */
/* inside the class keeps every access unaligned on both compilers.    */
/* ------------------------------------------------------------------ */

template <typename T, int V, bool Const> struct view_elem {
    using type = typename pack<T, V>::type;
};
template <typename T, int V> struct view_elem<T, V, true> {
    using type = const typename pack<T, V>::type;
};

template <typename T, int V, typename Int = int, bool Const = false> struct BatchView {
    using VT = typename view_elem<T, V, Const>::type;

    VT *data;
    Int si; /* stride along the first index  */
    Int sj; /* stride along the second index */

    VT &operator()(Int i, Int j) const noexcept { return data[i * si + j * sj]; }

    /* The same storage read as the transposed matrices: (i,j) -> (j,i). */
    BatchView transposed() const noexcept { return {data, sj, si}; }
    BatchView<T, V, Int, true> as_const() const noexcept { return {data, si, sj}; }
};
template <typename T, int V, typename Int = int>
using ConstBatchView = BatchView<T, V, Int, true>;

/* View one group of matrices stored with leading dimension ld in the given
 * layout: column-major has unit row stride, row-major unit column stride. A
 * const view is for read-only operands such as the reflector batch A. */
template <typename T, int V, typename Int = int>
BatchView<T, V, Int> make_view(T *p, bool rowmajor, Int ld) noexcept
{
    return {reinterpret_cast<typename pack<T, V>::type *>(p), rowmajor ? ld : Int(1),
            rowmajor ? Int(1) : ld};
}
template <typename T, int V, typename Int = int>
ConstBatchView<T, V, Int> make_const_view(const T *p, bool rowmajor, Int ld) noexcept
{
    return {reinterpret_cast<const typename pack<T, V>::type *>(p),
            rowmajor ? ld : Int(1), rowmajor ? Int(1) : ld};
}

/* View one group of a symmetric n x n batch so that its *lower* triangle is the
 * storage of the named triangle: the view of A for uplo lower, its transpose for
 * upper (A is symmetric, so a lower-triangle algorithm on A^T lands its factor
 * in A's upper storage). The potrf and sytrfnp kernels factor the lower
 * triangle of whatever view they are given, so this is what makes the four
 * (layout, uplo) cases one kernel each. */
template <typename T, int V, typename Int = int>
BatchView<T, V, Int> make_lower_view(T *p, bool rowmajor, bool upper, Int ld) noexcept
{
    const auto A = make_view<T, V, Int>(p, rowmajor, ld);
    return upper ? A.transposed() : A;
}

/* Scalars per group of V interleaved rows x cols matrices with leading dimension
 * ld: ld times the complementary extent, times V. */
template <typename Int>
std::size_t group_stride(bool rowmajor, Int ld, Int rows, Int cols, int V) noexcept
{
    return (std::size_t)ld * (rowmajor ? rows : cols) * V;
}

/* Runtime interleave width -> compile-time instantiation: calls f with a
 * std::integral_constant<int, V> for V in {2, 4, 8, 16} and returns true, or
 * returns false (f not called) for any other width. V is the compact interleave
 * width, not necessarily one hardware register: every width is valid for both
 * scalar types (V=16 doubles is a legal 1024-bit vector lowered to two AVX-512
 * ops); MKL's formats only ever select 2/4/8 for double and 4/8/16 for float. */
template <typename F> bool for_vlen(int V, F &&f)
{
    switch (V) {
    case 2: f(std::integral_constant<int, 2>{}); return true;
    case 4: f(std::integral_constant<int, 4>{}); return true;
    case 8: f(std::integral_constant<int, 8>{}); return true;
    case 16: f(std::integral_constant<int, 16>{}); return true;
    default: return false;
    }
}

/* ------------------------------------------------------------------ */
/* for_each_group: the loop over groups, threaded with OpenMP.         */
/*                                                                     */
/* Every all-groups driver is                                          */
/*     for_each_group<V>(nm, [&](Int g) { ... }, flops_per_group);     */
/* which runs body(g) for g = 0 .. ceil(nm/V)-1. Built with OpenMP the */
/* loop is a static-schedule `omp parallel for` on a team of           */
/* min(ngroups, omp_get_max_threads()) threads -- groups are           */
/* independent and equal-sized, so a static schedule balances exactly  */
/* and no thread ever idles -- but only when the call is worth a fork: */
/* more than one thread for at least two groups,                      */
/* available, and total work above parallel_min_flops; the if-clause  */
/* serializes it otherwise (a single group then costs only the gate's  */
/* few ICV reads and an inactive one-thread region), and a single group */
/* returns before touching the runtime at all. The work                */
/* estimate is optional: omit it and the call is assumed worth a fork  */
/* whenever it has two groups and two threads.                         */
/*                                                                     */
/* parallel_min_flops was measured, not guessed: a fork/join costs     */
/* 2-3 us (gcc, libgomp). On the 4-core AVX-512 box it was first tuned */
/* on, dgeqrf calls below ~1e5 flops ran slower in parallel and        */
/* everything above 2e5 gained 1.4-3.6x; on a second 4-core AVX-512    */
/* box parallel broke even near 3e4 and won from 6e4 up, and the 2e5   */
/* gate left up to 1.8x on the table (8x8 batches of 128). 5e4 is the  */
/* compromise: within 10% of serial at 3e4 on the faster-forking box,  */
/* near break-even on the slower one. The constant scales with fork    */
/* cost times single-core flop rate, so override -DCBK_OMP_MIN_FLOPS   */
/* for a different runtime or machine.                                 */
/*                                                                     */
/* Composition: the gate first asks whether one more nesting level may */
/* be active at all, then uses omp_get_max_threads(), the team size at */
/* the *current* nesting level. Called from inside a caller's own       */
/* parallel loop it therefore stays serial by default and no thread    */
/* pools compete. A per-level thread list, e.g. OMP_NUM_THREADS=8,2    */
/* with OMP_MAX_ACTIVE_LEVELS=2, gives each of the 8 outer threads a   */
/* 2-way inner team -- the standard OpenMP knobs, nothing library-     */
/* specific.                                                           */
/* ------------------------------------------------------------------ */

#ifndef CBK_OMP_MIN_FLOPS
#define CBK_OMP_MIN_FLOPS 5e4
#endif
constexpr double parallel_min_flops = CBK_OMP_MIN_FLOPS;

template <int V, typename Int, typename Body>
void for_each_group(
    Int nm, Body &&body,
    [[maybe_unused]] double flops_per_group = std::numeric_limits<double>::infinity())
{
    const Int ngroups = (nm + V - 1) / V;
    if (ngroups == 1) { /* the common single-group call: no OpenMP runtime at all */
        body(0);
        return;
    }
#ifdef _OPENMP
    /* at most one thread per group; nthreads > 1 also implies ngroups >= 2 */
    const Int nthreads =
        ngroups < omp_get_max_threads() ? ngroups : omp_get_max_threads();
    const bool parallel = omp_get_active_level() < omp_get_max_active_levels() &&
                          nthreads > 1 && flops_per_group * ngroups >= parallel_min_flops;
#pragma omp parallel for schedule(static) num_threads(nthreads) if (parallel)
#endif
    for (Int g = 0; g < ngroups; ++g)
        body(g);
}

/* B(i0:i1, 0:ncols) := 0 for one group: the quick-return fills (trsm's
 * alpha = 0, gels's empty op(A)) and gels's B(q:p) := 0 before it applies Q. */
template <typename T, int V, typename Int>
inline void zero_block(const BatchView<T, V, Int> &B, Int i0, Int i1, Int ncols) noexcept
{
    for (Int j = 0; j < ncols; ++j)
        for (Int i = i0; i < i1; ++i)
            B(i, j) = typename pack<T, V>::type{};
}

/* Every group of a rows x cols compact batch := 0, either layout. */
template <typename T, int V, typename Int>
void zero_compact(bool rowmajor, Int rows, Int cols, T *bp, Int ldbp, Int nm)
{
    const std::size_t str = group_stride(rowmajor, ldbp, rows, cols, V);
    for_each_group<V>(
        nm,
        [&](Int g) {
            const auto B = make_view<T, V, Int>(bp + g * str, rowmajor, ldbp);
            zero_block<T, V, Int>(B, Int(0), rows, cols);
        },
        (double)rows * cols * V /* stores per group */);
}

} /* namespace cbk::detail */

CBK_KERNEL_END

#endif /* CBK_COMMON_HPP */
