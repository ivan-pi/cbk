/* qr_reconstruct_compact.cpp
 *
 * Standalone example of the portable C API (cbk.h): QR-factor a batch of
 * small dense matrices in the compact (interleaved) format and rebuild each
 * one from its factors, A = Q R.
 *
 * The program follows the interleave-batch QR example that Arm ships with
 * Arm Performance Libraries, step for step, on this library's routines:
 *
 *     pack       column-major A_v  ->  compact ap          (this file)
 *     factor     dgeqrf_compact     ap <- (R, Householder vectors), taup <- tau
 *     extract R  copy ap, zero below the diagonal          (this file)
 *     multiply   dormqr_compact     rp <- Q R
 *     unpack     compact rp  ->  column-major (QR)_v       (this file)
 *     check      ||A_v - Q_v R_v||_1 <= 5 eps m n ||A_v||_1 for every v
 *
 * The compact layout is the one cbk.h documents: the nm matrices are split
 * into groups of V (the interleave width), and within a group element (i,j)
 * of the V matrices is stored contiguously, so a group of column-major m x n
 * matrices is a (ldap x n) array of V-wide packs:
 *
 *     A_v(i,j) = ap[ g*ldap*n*V + (j*ldap + i)*V + v ],   g = idx/V, v = idx%V
 *
 * A last group that is not full is padded with identity matrices, which the
 * kernels then run unmasked. Arm's `ninter` is V here and `nbatch` the number
 * of groups, nm/V rounded up.
 *
 * There is no LAPACK baseline and no warm-up: this is the round trip, timed
 * per phase so the cost of packing and unpacking can be compared with the
 * kernels' (the two library calls thread their loop over groups; the pack,
 * extract and unpack loops here are serial). Exit status 0 when every
 * matrix passes the check.
 *
 *     qr_reconstruct_compact [nm] [V] [m] [n]     (defaults 32768 4 10 10)
 *
 * V must be 2, 4, 8 or 16; the double-precision pack widths of SSE, AVX and
 * AVX-512 are 2, 4 and 8. The matrices must be square or tall, m >= n:
 * dormqr_compact takes the reflectors as an (ldap, k) batch, k = min(m,n),
 * exactly as LAPACK dormqr's A(LDA,K), so its group stride is ldap*k*V. For
 * m >= n that is the m x n buffer dgeqrf_compact wrote; for m < n the first k
 * columns of every matrix would have to be repacked into an m x k batch
 * first (or the factor-and-solve dgels_compact used, which handles both
 * shapes in one call).
 *
 * Assisted-by: Claude:claude-fable-5
 */

#include "cbk.h"

#include "cbk_matrix_batch.hpp" /* MatrixBatch, the dense batch the examples share */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

namespace {

using cbk::detail::MatrixBatch;

/* Offset of element (i,j) of the matrix idx in a compact column-major batch of
 * m x n matrices with leading dimension ldap: the layout formula of cbk.h. */
std::size_t compact_at(int idx, int i, int j, int ldap, int n, int V)
{
    const std::size_t g = (std::size_t)idx / V, v = (std::size_t)idx % V;
    return g * ldap * n * V + ((std::size_t)j * ldap + i) * V + v;
}

/* A compact buffer for nm matrices of m x n at leading dimension ldap = m:
 * (nm / V rounded up) groups of ldap * n packs, zero-filled. */
std::vector<double> compact_buffer(int nm, int m, int n, int V)
{
    const std::size_t ngroups = ((std::size_t)nm + V - 1) / V;
    return std::vector<double>(ngroups * m * n * V);
}

/* Pack the dense batch into compact layout. The slots of a partial last group
 * (idx >= nm) get the identity, so the kernels can process every group alike. */
void pack(const MatrixBatch<double> &A, double *ap, int V)
{
    const int m = A.rows(), n = A.cols(), nm = A.count();
    const int nslots = (int)(((std::size_t)nm + V - 1) / V * V);
    for (int idx = 0; idx < nslots; ++idx) {
        if (idx < nm) {
            const auto Av = A.view(idx);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    ap[compact_at(idx, i, j, m, n, V)] = Av(i, j);
        }
        else {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    ap[compact_at(idx, i, j, m, n, V)] = (i == j) ? 1.0 : 0.0;
        }
    }
}

/* Unpack the compact buffer into the dense batch (the padded slots are dropped). */
void unpack(MatrixBatch<double> &A, const double *ap, int V)
{
    const int m = A.rows(), n = A.cols(), nm = A.count();
    for (int idx = 0; idx < nm; ++idx) {
        auto Av = A.view(idx);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                Av(i, j) = ap[compact_at(idx, i, j, m, n, V)];
    }
}

/* Seconds since t0. */
using clock_type = std::chrono::steady_clock;
double seconds_since(clock_type::time_point t0)
{
    return std::chrono::duration<double>(clock_type::now() - t0).count();
}

/* The check the Arm example applies: the 1-norm (max column sum) of A - QR
 * against 5 eps m n times the 1-norm of A. */
bool reconstruction_ok(const MatrixBatch<double> &A, const MatrixBatch<double> &QR,
                       int idx)
{
    const int m = A.rows(), n = A.cols();
    const auto Av = A.view(idx), QRv = QR.view(idx);
    double norm_a = 0.0, norm_a_minus_qr = 0.0;
    for (int j = 0; j < n; ++j) {
        double sum_col_a = 0.0, sum_col_diff = 0.0;
        for (int i = 0; i < m; ++i) {
            sum_col_a += std::fabs(Av(i, j));
            sum_col_diff += std::fabs(Av(i, j) - QRv(i, j));
        }
        norm_a = std::max(norm_a, sum_col_a);
        norm_a_minus_qr = std::max(norm_a_minus_qr, sum_col_diff);
    }
    const double eps = std::numeric_limits<double>::epsilon();
    return norm_a_minus_qr <= 5.0 * eps * m * n * norm_a;
}

int usage(const char *prog)
{
    std::fprintf(stderr, "usage: %s [nm] [V] [m] [n]\n", prog);
    std::fprintf(stderr, "  nm  number of matrices (default 32768)\n");
    std::fprintf(stderr, "  V   interleave width, 2, 4, 8 or 16 (default 4)\n");
    std::fprintf(stderr,
                 "  m n rows and columns of each matrix, m >= n (default 10 10)\n");
    return EXIT_FAILURE;
}

} /* namespace */

int main(int argc, char **argv)
{
    if (argc != 1 && argc != 5) return usage(argv[0]);
    const int nm = argc == 5 ? std::atoi(argv[1]) : 32768;
    const int V = argc == 5 ? std::atoi(argv[2]) : 4;
    const int m = argc == 5 ? std::atoi(argv[3]) : 10;
    const int n = argc == 5 ? std::atoi(argv[4]) : 10;
    if (nm < 1 || (V != 2 && V != 4 && V != 8 && V != 16) || n < 1 || m < n)
        return usage(argv[0]);
    const int k = n; /* reflectors per matrix, min(m, n) */
    const std::size_t ngroups = ((std::size_t)nm + V - 1) / V;

    std::printf("QR round trip on the compact batch: nm = %d matrices of %d x %d, V = %d "
                "(%zu groups%s)\n",
                nm, m, n, V, ngroups, (std::size_t)nm % V ? ", the last one padded" : "");

    /* The dense batch: column-major m x n matrices with entries uniform in
     * [0, 1), as in the Arm example (seeded, so runs are reproducible). */
    MatrixBatch<double> A(nm, m, n);
    {
        std::mt19937_64 rng(4733);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        for (int idx = 0; idx < nm; ++idx) {
            auto Av = A.view(idx);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    Av(i, j) = uniform(rng);
        }
    }

    /* The compact buffers: the matrices, their reflector scalars (a compact
     * batch of k x 1 vectors), and R, the copy of the factorization that Q is
     * multiplied into. A std::vector<double> is 16-byte aligned; for the last
     * few percent align the base to the pack width instead (see cbk.h). */
    std::vector<double> ap = compact_buffer(nm, m, n, V);
    std::vector<double> taup = compact_buffer(nm, k, 1, V);
    std::vector<double> rp = compact_buffer(nm, m, n, V);

    /* pack */
    auto t0 = clock_type::now();
    pack(A, ap.data(), V);
    const double t_pack = seconds_since(t0);

    /* factor: ap <- (R on and above the diagonal, the Householder vectors
     * below), taup <- tau; the LAPACK dgeqrf storage, one matrix per slot */
    t0 = clock_type::now();
    int info = dgeqrf_compact('C', m, n, ap.data(), m, taup.data(), V, nm);
    const double t_factor = seconds_since(t0);
    if (info != 0) {
        std::fprintf(stderr, "dgeqrf_compact failed: info = %d\n", info);
        return EXIT_FAILURE;
    }

    /* extract R: a copy of the factorization with the reflectors zeroed. The
     * copy keeps the compact layout, so it is a whole-buffer copy followed by
     * a sweep over the strictly lower triangle of every slot, padding included. */
    t0 = clock_type::now();
    rp = ap;
    for (int idx = 0; idx < (int)(ngroups * V); ++idx)
        for (int j = 0; j < n; ++j)
            for (int i = j + 1; i < m; ++i)
                rp[compact_at(idx, i, j, m, n, V)] = 0.0;
    const double t_extract = seconds_since(t0);

    /* multiply: rp <- Q rp, Q applied from the left as the product of the k
     * reflectors in ap, without ever forming it. ap is read as the (m, k)
     * reflector batch: with m >= n that is exactly the buffer dgeqrf_compact
     * left behind. */
    t0 = clock_type::now();
    info = dormqr_compact('N', m, n, k, ap.data(), m, taup.data(), rp.data(), m, V, nm);
    const double t_multiply = seconds_since(t0);
    if (info != 0) {
        std::fprintf(stderr, "dormqr_compact failed: info = %d\n", info);
        return EXIT_FAILURE;
    }

    /* unpack, into a dense batch allocated (and so page-faulted) beforehand */
    MatrixBatch<double> QR(nm, m, n);
    t0 = clock_type::now();
    unpack(QR, rp.data(), V);
    const double t_unpack = seconds_since(t0);

    /* check */
    int fail = 0;
    for (int idx = 0; idx < nm; ++idx)
        if (!reconstruction_ok(A, QR, idx)) ++fail;

    std::printf("  pack      %9.6f s\n", t_pack);
    std::printf("  factorize %9.6f s   (dgeqrf_compact)\n", t_factor);
    std::printf("  extract R %9.6f s\n", t_extract);
    std::printf("  multiply  %9.6f s   (dormqr_compact)\n", t_multiply);
    std::printf("  unpack    %9.6f s\n", t_unpack);
    std::printf("  total     %9.6f s\n",
                t_pack + t_factor + t_extract + t_multiply + t_unpack);
    if (fail == 0) {
        std::printf("result check passed: ||A - QR||_1 <= 5 eps m n ||A||_1 for all %d "
                    "matrices\n",
                    nm);
        return EXIT_SUCCESS;
    }
    std::printf("result check FAILED for %d of %d matrices\n", fail, nm);
    return EXIT_FAILURE;
}
