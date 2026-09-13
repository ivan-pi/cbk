/* Smoke test of an installed cbk from a downstream project: one call into the
 * portable C API and, when the MKL extension is installed, one into the
 * MKL-style API with the format MKL reports for this CPU. Both factor V
 * interleaved copies of 3 * I, which a QR leaves as they are (every
 * sub-diagonal column is already zero, so tau = 0 and R = A). Exit status 0 on
 * success. */

#include "cbk.h"

#include <cstdio>
#include <vector>

#ifdef CBK_CONSUMER_WITH_MKL
#include <mkl_compact.h>

#include "cbk_compat.h"
#endif

namespace {

constexpr int n = 2, nm = 2;

/* V interleaved 2x2 matrices 3*I: element (i,j) of matrix v at
 * ap[(i + j*n) * V + v]. */
std::vector<double> identity3(int V)
{
    std::vector<double> ap(n * n * V, 0.0);
    for (int v = 0; v < V; ++v) {
        ap[(0 + 0 * n) * V + v] = 3.0;
        ap[(1 + 1 * n) * V + v] = 3.0;
    }
    return ap;
}

bool report(const char *api, int info, const std::vector<double> &ap, int V)
{
    bool ok = info == 0;
    for (int v = 0; v < V; ++v) {
        ok = ok && ap[(0 + 0 * n) * V + v] == 3.0 && ap[(1 + 1 * n) * V + v] == 3.0;
    }
    std::printf("%s: info = %d, %s\n", api, info, ok ? "ok" : "FAILED");
    return ok;
}

} // namespace

int main()
{
    /* First: the header compiled against and the library linked are one version. */
    const int version = cbk_get_version();
    if (version != CBK_VERSION) {
        std::printf("cbk_get_version() = %d, header CBK_VERSION = %d, FAILED\n", version,
                    CBK_VERSION);
        return 1;
    }
    std::printf("cbk_get_version() = %d, ok\n", version);

    constexpr int V = 2;
    std::vector<double> ap = identity3(V), taup(n * V);
    const int info = dgeqrf_compact('C', n, n, ap.data(), n, taup.data(), V, nm);
    const bool ok_compact = report("cbk.h", info, ap, V);

#ifdef CBK_CONSUMER_WITH_MKL
    const MKL_COMPACT_PACK format = mkl_get_format_compact();
    const int VM = (int)cbk::detail::vlen_for_format<double>(format);
    std::vector<double> apm = identity3(VM), taupm(n * VM);
    MKL_INT minfo = -1;
    double work[1];
    cbk_dgeqrf_compact(MKL_COL_MAJOR, n, n, apm.data(), n, taupm.data(), work, 1, &minfo,
                       format, nm);
    const bool ok_mkl = report("cbk_compat.h", (int)minfo, apm, VM);
#else
    const bool ok_mkl = true;
#endif

    return ok_compact && ok_mkl ? 0 : 1;
}
