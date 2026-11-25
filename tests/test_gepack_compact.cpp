// test_gepack_compact.cpp
//
// Validation of the portable conversion routines (?gepack_compact,
// ?geunpack_compact, cbk_get_size_compact) against the layout cbk.h documents,
// written out here by hand on purpose: every other suite packs and unpacks
// through these routines, so this one must not.
//
//   A_v(i,j) = ap[ g*ldap*n*V + (j*ldap + i)*V + v ]   (column-major)
//   A_v(i,j) = ap[ g*ldap*m*V + (i*ldap + j)*V + v ]   (row-major)
//
// with g = idx/V and v = idx%V. All checks are structural, hence exact:
//   1. C API argument validation (info = -j), and cbk_get_size_compact's value.
//   2. ?gepack_compact: every scalar of the buffer is the matrix element the
//      formula names; the slots past nm hold the identity; the scalars between
//      the matrix extent and the leading dimension are untouched.
//   3. ?geunpack_compact: every matrix element comes back; the entries between
//      the extent and lda are untouched; the padded slots are not read (they
//      are poisoned with NaN first).
// over both layouts and precisions, every interleave width, leading
// dimensions equal to and beyond the extent, full and partial last groups,
// several groups (the threaded path), and the small-dimension cross product.
//
// Assisted-by: Claude:claude-fable-5

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#include "test_compact_util.hpp" // compact<T>, frand, small_dims, ApiCheck, finish

using namespace cbk::test;

// Offset of element (i,j) of matrix idx in the compact buffer, from cbk.h.
static std::size_t compact_index(bool row, int m, int n, int ldap, int V, int idx, int i,
                                 int j)
{
    const auto g = (std::size_t)(idx / V), v = (std::size_t)(idx % V);
    return row ? g * ldap * m * V + ((std::size_t)i * ldap + j) * V + v
               : g * ldap * n * V + ((std::size_t)j * ldap + i) * V + v;
}

template <class T>
static int run_case(char layout, int V, int nm, int m, int n, int lda_extra,
                    int ldap_extra)
{
    const bool row = (layout == 'R');
    const int ld0 = std::max(1, row ? n : m); /* the smallest legal leading dim */
    const int sd = row ? m : n;               /* the other extent */
    const int lda = ld0 + lda_extra, ldap = ld0 + ldap_extra;
    const int ng = (nm + V - 1) / V;
    const T sentinel = T(-77);

    /* nm dense matrices, each in its own allocation of lda * sd scalars */
    const std::size_t dsz = std::max<std::size_t>(1, (std::size_t)lda * sd);
    std::vector<std::vector<T>> A(nm, std::vector<T>(dsz, sentinel));
    std::vector<const T *> cptr(nm);
    const auto dense = [&](int i, int j) {
        return row ? (std::size_t)i * lda + j : (std::size_t)j * lda + i;
    };
    for (int idx = 0; idx < nm; ++idx) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                A[idx][dense(i, j)] = frand<T>();
        cptr[idx] = A[idx].data();
    }

    /* 1. the size */
    const std::size_t want = (std::size_t)ldap * sd * V * ng * sizeof(T);
    const std::size_t bytes = compact<T>::get_size(ldap, sd, V, nm);
    int fails = (bytes != want);

    /* 2. pack into a sentinel-filled buffer, with a guard past its end */
    const std::size_t len = want / sizeof(T), guard = 16;
    std::vector<T> ap(len + guard, sentinel);
    fails +=
        (compact<T>::gepack(layout, m, n, cptr.data(), lda, ap.data(), ldap, V, nm) != 0);
    std::vector<char> seen(len + guard, 0);
    int bad_live = 0, bad_pad = 0, bad_gap = 0;
    for (int idx = 0; idx < ng * V; ++idx)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                const std::size_t e = compact_index(row, m, n, ldap, V, idx, i, j);
                seen[e] = 1;
                if (idx < nm)
                    bad_live += (ap[e] != A[idx][dense(i, j)]);
                else
                    bad_pad += (ap[e] != ((i == j) ? T(1) : T(0)));
            }
    for (std::size_t e = 0; e < len + guard; ++e)
        if (!seen[e]) bad_gap += (ap[e] != sentinel);

    /* 3. unpack: poison the padded slots, hand over sentinel-filled matrices */
    for (int idx = nm; idx < ng * V; ++idx)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                ap[compact_index(row, m, n, ldap, V, idx, i, j)] =
                    std::numeric_limits<T>::quiet_NaN();
    std::vector<std::vector<T>> B(nm, std::vector<T>(dsz, sentinel));
    std::vector<T *> ptr(nm);
    for (int idx = 0; idx < nm; ++idx)
        ptr[idx] = B[idx].data();
    fails += (compact<T>::geunpack(layout, m, n, ptr.data(), lda, ap.data(), ldap, V,
                                   nm) != 0);
    int bad_back = 0;
    for (int idx = 0; idx < nm; ++idx)
        bad_back += !std::equal(A[idx].begin(), A[idx].end(), B[idx].begin());

    fails += (bad_live + bad_pad + bad_gap + bad_back != 0);
    std::printf("T=%-6s V=%-2d nm=%-3d %s m=%-2d n=%-2d lda=%-2d ldap=%-2d | size %s | "
                "live %d pad %d gap %d | unpack %d | %s\n",
                compact<T>::name, V, nm, row ? "row" : "col", m, n, lda, ldap,
                bytes == want ? "ok" : "BAD", bad_live, bad_pad, bad_gap, bad_back,
                fails ? "FAIL" : "OK");
    return fails;
}

// --------------------- C API argument validation --------------------

static int test_validation()
{
    const int m = 5, n = 3, V = 4, nm = 6, ng = 2;
    std::vector<double> a((size_t)m * m * nm, 1), ap((size_t)m * m * V * ng, 0);
    std::vector<double *> ptr(nm);
    for (int idx = 0; idx < nm; ++idx)
        ptr[idx] = a.data() + (size_t)idx * m * m;
    auto pack = [&](char lay, int m_, int n_, int lda_, int ldap_, int V_, int nm_) {
        return dgepack_compact(lay, m_, n_, ptr.data(), lda_, ap.data(), ldap_, V_, nm_);
    };
    auto unpack = [&](char lay, int m_, int n_, int lda_, int ldap_, int V_, int nm_) {
        return dgeunpack_compact(lay, m_, n_, ptr.data(), lda_, ap.data(), ldap_, V_,
                                 nm_);
    };
    // clang-format off
    const ApiCheck t[] = {
        {"pack valid col",     pack('C', m, n,  m,   m,   V, nm),   0},
        {"pack valid row",     pack('r', m, n,  n,   n,   V, nm),   0},   // row-major lds >= n
        {"pack bad layout",    pack('X', m, n,  m,   m,   V, nm),  -1},
        {"pack m<0",           pack('C', -1, n, m,   m,   V, nm),  -2},
        {"pack n<0",           pack('C', m, -1, m,   m,   V, nm),  -3},
        {"pack lda<m",         pack('C', m, n,  m-1, m,   V, nm),  -5},
        {"pack lda<n (row)",   pack('R', m, n,  n-1, n,   V, nm),  -5},
        {"pack ldap<m",        pack('C', m, n,  m,   m-1, V, nm),  -7},
        {"pack ldap<n (row)",  pack('R', m, n,  n,   n-1, V, nm),  -7},
        {"pack bad V",         pack('C', m, n,  m,   m,   3, nm),  -8},
        {"pack nm<0",          pack('C', m, n,  m,   m,   V, -1),  -9},
        {"pack empty m=0",     pack('C', 0, n,  1,   1,   V, nm),   0},
        {"pack empty nm=0",    pack('C', m, n,  m,   m,   V, 0),    0},
        {"unpack valid col",   unpack('c', m, n,  m,   m,   V, nm),   0},
        {"unpack valid row",   unpack('R', m, n,  n,   n,   V, nm),   0},
        {"unpack bad layout",  unpack('X', m, n,  m,   m,   V, nm),  -1},
        {"unpack m<0",         unpack('C', -1, n, m,   m,   V, nm),  -2},
        {"unpack n<0",         unpack('C', m, -1, m,   m,   V, nm),  -3},
        {"unpack lda<m",       unpack('C', m, n,  m-1, m,   V, nm),  -5},
        {"unpack ldap<m",      unpack('C', m, n,  m,   m-1, V, nm),  -7},
        {"unpack bad V",       unpack('C', m, n,  m,   m,   5, nm),  -8},
        {"unpack nm<0",        unpack('C', m, n,  m,   m,   V, -1),  -9},
        {"unpack empty n=0",   unpack('C', m, 0,  m,   m,   V, nm),   0},
    };
    // clang-format on
    int fails = report_api_checks(t);

    /* cbk_get_size_compact: bytes of whole groups; 0 for an illegal argument, an
     * empty batch or an empty matrix; no int overflow for a large batch */
    struct {
        const char *what;
        std::size_t got, want;
    } const s[] = {
        {"dsize 5x3 V=4 nm=6", cbk_get_size_compact(5, 3, 4, 6, sizeof(double)),
         (size_t)5 * 3 * 4 * 2 * 8},
        {"ssize 5x3 V=4 nm=6", cbk_get_size_compact(5, 3, 4, 6, sizeof(float)),
         (size_t)5 * 3 * 4 * 2 * 4},
        {"dsize nm=8 (full)", cbk_get_size_compact(5, 3, 8, 8, sizeof(double)),
         (size_t)5 * 3 * 8 * 1 * 8},
        {"dsize nm=9", cbk_get_size_compact(5, 3, 8, 9, sizeof(double)),
         (size_t)5 * 3 * 8 * 2 * 8},
        {"ssize V=16 nm=1", cbk_get_size_compact(2, 2, 16, 1, sizeof(float)),
         (size_t)2 * 2 * 16 * 1 * 4},
        {"dsize nm=0", cbk_get_size_compact(5, 3, 4, 0, sizeof(double)), 0},
        {"dsize sd=0", cbk_get_size_compact(5, 0, 4, 6, sizeof(double)), 0},
        {"dsize ld<0", cbk_get_size_compact(-1, 3, 4, 6, sizeof(double)), 0},
        {"dsize sd<0", cbk_get_size_compact(5, -1, 4, 6, sizeof(double)), 0},
        {"dsize bad V", cbk_get_size_compact(5, 3, 3, 6, sizeof(double)), 0},
        {"dsize nm<0", cbk_get_size_compact(5, 3, 4, -1, sizeof(double)), 0},
        {"elem_size 0", cbk_get_size_compact(5, 3, 4, 6, 0), 0},
        {"elem_size 1 (scalars)", cbk_get_size_compact(5, 3, 4, 6, 1),
         (size_t)5 * 3 * 4 * 2},
        {"dsize beyond 2^31", cbk_get_size_compact(64, 64, 8, 1 << 20, sizeof(double)),
         (size_t)64 * 64 * 8 * (1 << 17) * 8},
    };
    for (const auto &c : s) {
        const bool ok = (c.got == c.want);
        std::printf("  %-22s -> %zu (want %zu) %s\n", c.what, c.got, c.want,
                    ok ? "OK" : "FAIL");
        fails += !ok;
    }
    return fails;
}

// ------------------------------- main --------------------------------

template <class T> static int run_all()
{
    int fails = 0;
    for (char lay : {'C', 'R'})
        for (int V : {2, 4, 8, 16}) {
            fails += run_case<T>(lay, V, V, 7, 5, 0, 0);         /* one full group */
            fails += run_case<T>(lay, V, 2 * V + 1, 7, 5, 0, 0); /* padded last group */
            fails +=
                run_case<T>(lay, V, 3 * V - 1, 5, 7, 2, 3); /* lds beyond the extent */
            fails += run_case<T>(lay, V, 1, 6, 6, 1, 0);    /* a lone matrix */
        }
    /* many groups of a larger matrix: past the threading gate */
    fails += run_case<T>('C', 8, 301, 40, 33, 0, 3);
    fails += run_case<T>('R', 4, 301, 33, 40, 2, 0);
    /* the small-dimension cross product, the empty matrix included */
    for (int m : small_dims)
        for (int n : small_dims) {
            fails += run_case<T>('C', 4, 5, m, n, 0, 1);
            fails += run_case<T>('R', 8, 9, m, n, 1, 0);
        }
    return fails;
}

int main()
{
    int fails = test_validation();
    fails += run_all<double>();
    fails += run_all<float>();
    return finish(fails);
}
