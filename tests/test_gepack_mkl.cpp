/* test_gepack_mkl.cpp
 *
 * The portable ?gepack_compact / ?geunpack_compact / ?get_size_compact against
 * Intel MKL's mkl_?gepack_compact / mkl_?geunpack_compact /
 * mkl_?get_size_compact, which they are documented to reproduce: the same
 * buffer bit for bit -- the identity in the slots past nm and the scalars
 * between the matrix extent and the leading dimension left untouched included
 * -- the same matrices back, and the same size in bytes. Both layouts, both
 * precisions, MKL's three pack formats (hence three interleave widths per
 * precision), leading dimensions at and beyond the extent, full and partial
 * last groups.
 *
 * Assisted-by: Claude:claude-fable-5
 */

#include "test_mkl_util.hpp" /* compact<T>, mkl<T> + the shared helpers */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

using namespace cbk::test;

template <class T>
static int run_case(MKL_LAYOUT layout, MKL_COMPACT_PACK fmt, int nm, int m, int n,
                    int lda_extra, int ldap_extra)
{
    const bool row = (layout == MKL_ROW_MAJOR);
    const char lay = row ? 'R' : 'C';
    const int V = mkl<T>::vlen(fmt);
    const int ld0 = std::max(1, row ? n : m), sd = row ? m : n;
    const int lda = ld0 + lda_extra, ldap = ld0 + ldap_extra;
    const T sentinel = T(-77);

    const std::size_t dsz = (std::size_t)lda * sd;
    std::vector<T> a((std::size_t)nm * dsz);
    for (auto &x : a)
        x = frand<T>();
    std::vector<const T *> cptr(nm);
    for (int idx = 0; idx < nm; ++idx)
        cptr[idx] = a.data() + (std::size_t)idx * dsz;

    /* the size */
    const std::size_t bytes = compact<T>::get_size(ldap, sd, V, nm);
    const auto bytes_mkl = (std::size_t)mkl<T>::get_size(ldap, sd, fmt, nm);
    int fails = (bytes != bytes_mkl);

    /* the packed buffers, from the same sentinel fill */
    const std::size_t len = bytes_mkl / sizeof(T);
    std::vector<T> ap(len, sentinel), ap_mkl(len, sentinel);
    fails +=
        (compact<T>::gepack(lay, m, n, cptr.data(), lda, ap.data(), ldap, V, nm) != 0);
    mkl<T>::gepack(layout, m, n, cptr.data(), lda, ap_mkl.data(), ldap, fmt, nm);
    const bool same_pack = (std::memcmp(ap.data(), ap_mkl.data(), bytes_mkl) == 0);

    /* the unpacked matrices, each library from the other's buffer */
    std::vector<T> b(a.size(), sentinel), b_mkl(a.size(), sentinel);
    std::vector<T *> ptr(nm), ptr_mkl(nm);
    for (int idx = 0; idx < nm; ++idx) {
        ptr[idx] = b.data() + (std::size_t)idx * dsz;
        ptr_mkl[idx] = b_mkl.data() + (std::size_t)idx * dsz;
    }
    fails += (compact<T>::geunpack(lay, m, n, ptr.data(), lda, ap_mkl.data(), ldap, V,
                                   nm) != 0);
    mkl<T>::geunpack(layout, m, n, ptr_mkl.data(), lda, ap.data(), ldap, fmt, nm);
    const bool same_unpack =
        (std::memcmp(b.data(), b_mkl.data(), b.size() * sizeof(T)) == 0);

    fails += !same_pack + !same_unpack;
    std::printf(
        "T=%-6s %-6s V=%-2d nm=%-3d %s m=%-2d n=%-2d lda=%-2d ldap=%-2d | size %s "
        "| pack %s | unpack %s | %s\n",
        compact<T>::name, cbk::detail::compact_format_name(fmt), V, nm,
        row ? "row" : "col", m, n, lda, ldap, bytes == bytes_mkl ? "same" : "DIFF",
        same_pack ? "same" : "DIFF", same_unpack ? "same" : "DIFF",
        fails ? "FAIL" : "OK");
    return fails;
}

template <class T> static int run_all()
{
    int fails = 0;
    for (MKL_LAYOUT layout : {MKL_COL_MAJOR, MKL_ROW_MAJOR})
        for (MKL_COMPACT_PACK fmt :
             {MKL_COMPACT_SSE, MKL_COMPACT_AVX, MKL_COMPACT_AVX512}) {
            const int V = mkl<T>::vlen(fmt);
            fails += run_case<T>(layout, fmt, V, 7, 5, 0, 0);
            fails += run_case<T>(layout, fmt, 2 * V + 1, 7, 5, 0, 0);
            fails += run_case<T>(layout, fmt, 3 * V - 1, 5, 7, 2, 3);
            fails += run_case<T>(layout, fmt, 1, 6, 6, 1, 0);
            fails += run_case<T>(layout, fmt, 5 * V + 3, 1, 4, 0, 0);
            fails += run_case<T>(layout, fmt, 5 * V + 3, 4, 1, 0, 2);
        }
    return fails;
}

int main()
{
    int fails = run_all<double>();
    fails += run_all<float>();
    return finish(fails);
}
