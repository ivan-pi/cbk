/* bench_util.hpp
 *
 * The MKL side of the benchmark harness, on top of bench_common.hpp (the
 * MKL-free part: checks, MatrixPool, timing, bench_sizes, BenchArgs): the MKL
 * compact-format lookups, PackedPool (a pool's pristine compact image in
 * mkl_malloc storage, packed by mkl_dgepack_compact, restored before every
 * timed pass) with PackedSystems (a solve benchmark's matrix + RHS pair of
 * those), the known-solution right-hand sides the Cholesky and LDL^T
 * benchmarks form with cblas_dgemm, and CmdArgs, the shared command line with
 * --simdlen resolved to an MKL_COMPACT_PACK. Needs the MKL headers, and
 * PackedPool calls mkl_malloc / mkl_dgepack_compact, so programs using it
 * link MKL (the benchmarks of the MKL-style API do).
 *
 * Assisted-by: Claude:claude-opus-4-8 Claude:claude-fable-5
 */

#ifndef CBK_BENCH_UTIL_HPP
#define CBK_BENCH_UTIL_HPP

#include "bench_common.hpp"
#include "cbk_compat.h"
#include "cbk_mkl_alloc.h" /* mkl_alloc_bytes / mkl_buffer, for PackedPool */

#include <mkl.h>         /* cblas_dgemm, for the known-solution RHS */
#include <mkl_compact.h> /* mkl_dget_size_compact / mkl_dgepack_compact */

#include <cstring>
#include <vector>

namespace cbk::bench {

using cbk::detail::compact_format_name; /* format -> "SSE"/"AVX"/"AVX512" */
using cbk::detail::format_for_vlen;     /* interleave width -> pack format */
using cbk::detail::vlen_for_format;     /* pack format -> interleave width */

/* B := A X for the known solution X(:,j) = j + 1 (fill_known_x) -- the
 * right-hand sides the solve benchmarks recover, measured against that X by
 * forward_error. */
inline void fill_known_rhs(const MatrixPool &A, MatrixPool &B)
{
    const int n = A.rows(), nrhs = B.cols();
    std::vector<double> xs((size_t)n * nrhs);
    const auto X = mat_view(xs.data(), n, nrhs);
    fill_known_x(X);
    for (int v = 0; v < A.count(); ++v) {
        const auto Av = A.view(v);
        const auto Bv = B.view(v);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nrhs, n, 1.0, Av.data,
                    Av.ld(), X.data, X.ld(), 0.0, Bv.data, Bv.ld());
    }
}

/* A pool packed column-major into a compact buffer it owns: the pristine bytes
 * an in-place compact routine's working copy is restored from before every
 * timed pass (the pack itself stays untimed). */
struct PackedPool {
    MKL_INT bytes; /* mkl_dget_size_compact reports bytes */
    cbk::detail::mkl_buffer<double> p;

    PackedPool(const MatrixPool &P, MKL_COMPACT_PACK fmt)
        : bytes(mkl_dget_size_compact(P.rows(), P.cols(), fmt, P.count())),
          p(cbk::detail::mkl_alloc_bytes<double>(bytes))
    {
        auto ptrs = P.base_ptrs();
        mkl_dgepack_compact(MKL_COL_MAJOR, P.rows(), P.cols(), ptrs.data(), P.rows(),
                            p.get(), P.rows(), fmt, P.count());
    }

    /* An uninitialized working buffer of the same size. */
    cbk::detail::mkl_buffer<double> work() const
    {
        return cbk::detail::mkl_alloc_bytes<double>(bytes);
    }
    void restore_into(double *dst) const { std::memcpy(dst, p.get(), bytes); }
};

/* A solve benchmark's pair of pristine compact images -- the matrices and
 * their right-hand sides -- restored together before every timed pass. */
struct PackedSystems {
    PackedPool a, b;

    PackedSystems(const MatrixPool &A, const MatrixPool &B, MKL_COMPACT_PACK fmt)
        : a(A, fmt), b(B, fmt)
    {
    }

    void restore_into(double *ap, double *bp) const
    {
        a.restore_into(ap);
        b.restore_into(bp);
    }
};

/* Unpack a compact solution batch (n x nrhs per matrix, column-major, ld = n)
 * and measure it against fill_known_rhs's X: the correctness gate of a compact
 * solve path, read from the buffer its last timed pass left behind. */
inline double unpacked_forward_error(const double *bp, int n, int nrhs, int nmat,
                                     MKL_COMPACT_PACK fmt)
{
    std::vector<double> X((size_t)nmat * n * nrhs);
    std::vector<double *> Xp(nmat);
    for (int v = 0; v < nmat; ++v)
        Xp[v] = X.data() + (size_t)v * n * nrhs;
    mkl_dgeunpack_compact(MKL_COL_MAJOR, n, nrhs, Xp.data(), n, bp, n, fmt, nmat);
    return forward_error(X.data(), n, nrhs, nmat);
}

/* The MKL benchmarks' command line: BenchArgs (bench_common.hpp) with the
 * interleave width resolved to MKL's pack format. The default is the host's
 * widest, as mkl_get_format_compact reports it (the same width host_simdlen
 * finds without MKL); --simdlen forces a narrower one. A wider interleave
 * than the host's native SIMD cannot execute on MKL's kernels, so that is
 * refused. Hold the object const. */
struct CmdArgs : BenchArgs {
    MKL_COMPACT_PACK fmt; /* the host's widest, or the --simdlen one */

    CmdArgs(int argc, char **argv, const char *prog) : BenchArgs(argc, argv, prog)
    {
        const MKL_COMPACT_PACK native = mkl_get_format_compact();
        fmt = simdlen ? format_for_vlen<double>(simdlen) : native;
        V = vlen_for_format<double>(fmt);
        check(V > 0 && V <= vlen_for_format<double>(native),
              "requested --simdlen exceeds the host's native SIMD width");
    }
};

} /* namespace cbk::bench */

#endif /* CBK_BENCH_UTIL_HPP */
