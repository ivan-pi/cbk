/* bench_util.hpp
 *
 * The harness shared by the MKL benchmark programs: the MKL-free core of
 * bench_portable_util.hpp (abort-on-failure checks, pack-aligned std::vector
 * storage, MatrixPool, best-of-N timing, the OpenMP thread count) plus the
 * MKL-specific pieces -- the MKL compact-format lookups, PackedPool (the
 * pristine compact image restored before every timed pass), and the
 * factorization / solve benchmarks' command line (--size-sweep, --simdlen,
 * --nrhs, [nmat] [reps]). Needs the MKL headers, and PackedPool calls
 * mkl_malloc / mkl_dgepack_compact, so programs using it link MKL (all
 * benchmarks except bench_geqrf_armpl do).
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#ifndef CQR_BENCH_UTIL_HPP
#define CQR_BENCH_UTIL_HPP

#include "cqr_mkl_ext.h"
#include "cqr_mkl_alloc.h" /* mkl_alloc_bytes / mkl_buffer, for PackedPool */
#include "bench_portable_util.hpp"

#include <mkl_compact.h> /* mkl_dget_size_compact / mkl_dgepack_compact */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace cqr::bench {

using cqr::detail::compact_format_name; /* format -> "SSE"/"AVX"/"AVX512" */
using cqr::detail::format_for_vlen;     /* interleave width -> pack format */
using cqr::detail::vlen_for_format;     /* pack format -> interleave width */

/* A pool packed column-major into a compact buffer it owns: the pristine bytes
 * an in-place compact routine's working copy is restored from before every
 * timed pass (the pack itself stays untimed). */
struct PackedPool {
    MKL_INT bytes; /* mkl_dget_size_compact reports bytes */
    cqr::detail::mkl_buffer<double> p;

    PackedPool(const MatrixPool &P, MKL_COMPACT_PACK fmt)
        : bytes(mkl_dget_size_compact(P.rows(), P.cols(), fmt, P.count())),
          p(cqr::detail::mkl_alloc_bytes<double>(bytes))
    {
        auto ptrs = P.base_ptrs();
        mkl_dgepack_compact(MKL_COL_MAJOR, P.rows(), P.cols(), ptrs.data(), P.rows(),
                            p.get(), P.rows(), fmt, P.count());
    }

    /* An uninitialized working buffer of the same size. */
    cqr::detail::mkl_buffer<double> work() const
    {
        return cqr::detail::mkl_alloc_bytes<double>(bytes);
    }
    void restore_into(double *dst) const { std::memcpy(dst, p.get(), bytes); }
};

/* Command line of the factorization and solve benchmarks: positional [nmat]
 * [reps], plus --size-sweep=nmin:nmax[:stride] (cqr-only scan), --simdlen=2|4|8
 * (force the interleave width instead of the host default) and --nrhs=k (right-
 * hand sides; the factorization benchmarks ignore it). The constructor parses
 * and validates and resolves the pack format; hold the object const. */
struct CmdArgs {
    int nmat = 512;
    int reps = 3;
    int nrhs = 1;
    bool sweep = false;
    int sweep_min = 0, sweep_max = 0, sweep_step = 1;
    MKL_COMPACT_PACK fmt; /* the host's widest, or the --simdlen one */
    int V;                /* its interleave width for double */

    CmdArgs(int argc, char **argv, const char *prog)
    {
        int simdlen = 0;
        std::vector<const char *> pos;
        for (int i = 1; i < argc; ++i) {
            if (std::strncmp(argv[i], "--size-sweep=", 13) == 0) {
                int got = std::sscanf(argv[i] + 13, "%d:%d:%d", &sweep_min, &sweep_max,
                                      &sweep_step);
                check(got >= 2, "usage: --size-sweep=nmin:nmax[:stride]");
                if (got == 2) sweep_step = 1;
                sweep = true;
            }
            else if (std::strncmp(argv[i], "--simdlen=", 10) == 0)
                simdlen = std::atoi(argv[i] + 10);
            else if (std::strncmp(argv[i], "--nrhs=", 7) == 0)
                nrhs = std::atoi(argv[i] + 7);
            else
                pos.push_back(argv[i]);
        }
        if (!pos.empty()) nmat = std::atoi(pos[0]);
        if (pos.size() > 1) reps = std::atoi(pos[1]);
        if (!(nmat > 0 && reps > 0 && nrhs > 0)) {
            std::printf("usage: %s [--size-sweep=nmin:nmax[:stride]] [--simdlen=2|4|8] "
                        "[--nrhs=k>0] [nmat>0] [reps>0]\n",
                        prog);
            std::exit(1);
        }
        check(!sweep || (sweep_min > 0 && sweep_max >= sweep_min && sweep_step > 0),
              "usage: --size-sweep needs 0 < nmin <= nmax and stride > 0");
        /* Double compact widths are 2/4/8 (SSE/AVX/AVX512); 16 is float's AVX512
         * width and has no double format. */
        check(simdlen == 0 || simdlen == 2 || simdlen == 4 || simdlen == 8,
              "usage: --simdlen must be 2, 4, or 8 (16 is float-only; this is double)");

        /* A wider interleave than the host's native SIMD cannot execute
         * (mkl_get_format_compact reports the widest the architecture supports). */
        const MKL_COMPACT_PACK native = mkl_get_format_compact();
        fmt = simdlen ? format_for_vlen<double>(simdlen) : native;
        V = vlen_for_format<double>(fmt);
        check(V > 0 && V <= vlen_for_format<double>(native),
              "requested --simdlen exceeds the host's native SIMD width");
    }
};

} /* namespace cqr::bench */

#endif /* CQR_BENCH_UTIL_HPP */
