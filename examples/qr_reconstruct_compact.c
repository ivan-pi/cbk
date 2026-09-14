/* qr_reconstruct_compact.c
 *
 * Standalone C99 example of the portable C API (cbk.h): QR-factor a batch of
 * small dense matrices in the compact (interleaved) format and rebuild each
 * one from its factors, A = Q R. Nothing but cbk.h, the C library and
 * OpenMP (omp_get_wtime, and the loops over groups).
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
 * kernels'. The two library calls thread their own loop over groups; the
 * pack, extract and unpack loops here are OpenMP loops over the same groups.
 * Exit status 0 when every matrix passes the check.
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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#else
#include <time.h>
static double omp_get_wtime(void) { return (double)clock() / CLOCKS_PER_SEC; }
#endif

/* Offset of element (i,j) of the matrix idx in a compact column-major batch of
 * m x n matrices with leading dimension ldap: the layout formula of cbk.h. */
static size_t compact_at(int idx, int i, int j, int ldap, int n, int V)
{
    const size_t g = (size_t)idx / V, v = (size_t)idx % V;
    return g * ldap * n * V + ((size_t)j * ldap + i) * V + v;
}

/* Column-major dense batch: matrix idx starts at idx*m*n, leading dimension m. */
static size_t dense_at(int idx, int i, int j, int m, int n)
{
    return (size_t)idx * m * n + (size_t)j * m + i;
}

static int usage(const char *prog)
{
    fprintf(stderr, "usage: %s [nm] [V] [m] [n]\n", prog);
    fprintf(stderr, "  nm  number of matrices (default 32768)\n");
    fprintf(stderr, "  V   interleave width, 2, 4, 8 or 16 (default 4)\n");
    fprintf(stderr, "  m n rows and columns of each matrix, m >= n (default 10 10)\n");
    return EXIT_FAILURE;
}

int main(int argc, char **argv)
{
    if (argc != 1 && argc != 5) return usage(argv[0]);
    const int nm = argc == 5 ? atoi(argv[1]) : 32768;
    const int V = argc == 5 ? atoi(argv[2]) : 4;
    const int m = argc == 5 ? atoi(argv[3]) : 10;
    const int n = argc == 5 ? atoi(argv[4]) : 10;
    if (nm < 1 || (V != 2 && V != 4 && V != 8 && V != 16) || n < 1 || m < n)
        return usage(argv[0]);
    const int k = n;                  /* reflectors per matrix, min(m, n) */
    const int ngroups = (nm + V - 1) / V; /* the last one padded if V does not divide nm */

    printf("QR round trip on the compact batch: nm = %d matrices of %d x %d, V = %d "
           "(%d groups%s)\n",
           nm, m, n, V, ngroups, nm % V ? ", the last one padded" : "");

    /* The dense batch: column-major m x n matrices with entries uniform in
     * [0, 1), as in the Arm example (seeded, so runs are reproducible). */
    const size_t dense_size = (size_t)nm * m * n;
    double *A = malloc(dense_size * sizeof(double));
    double *QR = malloc(dense_size * sizeof(double));
    srand(4733);
    for (size_t l = 0; l < dense_size; ++l)
        A[l] = (double)rand() / RAND_MAX;

    /* The compact buffers: the matrices, their reflector scalars (a compact
     * batch of k x 1 vectors), and R, the copy of the factorization that Q is
     * multiplied into. Any double-aligned address is correct; for the last
     * few percent align the base to the pack width instead (see cbk.h). */
    const size_t a_size = (size_t)ngroups * m * n * V;
    const size_t t_size = (size_t)ngroups * k * V;
    double *ap = malloc(a_size * sizeof(double));
    double *taup = malloc(t_size * sizeof(double));
    double *rp = malloc(a_size * sizeof(double));
    if (!A || !QR || !ap || !taup || !rp) {
        fprintf(stderr, "out of memory\n");
        return EXIT_FAILURE;
    }

    /* pack, one group at a time: the slots of a partial last group (idx >= nm)
     * get the identity, so the kernels can process every group alike */
    double t0 = omp_get_wtime();
#pragma omp parallel for
    for (int g = 0; g < ngroups; ++g) {
        for (int v = 0; v < V; ++v) {
            const int idx = g * V + v;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    ap[compact_at(idx, i, j, m, n, V)] =
                        idx < nm ? A[dense_at(idx, i, j, m, n)] : (i == j ? 1.0 : 0.0);
        }
    }
    const double t_pack = omp_get_wtime() - t0;

    /* factor: ap <- (R on and above the diagonal, the Householder vectors
     * below), taup <- tau; the LAPACK dgeqrf storage, one matrix per slot */
    t0 = omp_get_wtime();
    int info = dgeqrf_compact('C', m, n, ap, m, taup, V, nm);
    const double t_factor = omp_get_wtime() - t0;
    if (info != 0) {
        fprintf(stderr, "dgeqrf_compact failed: info = %d\n", info);
        return EXIT_FAILURE;
    }

    /* extract R: a copy of the factorization with the reflectors zeroed. The
     * copy keeps the compact layout, so it is a whole-buffer copy followed by
     * a sweep over the strictly lower triangle of every slot, padding included. */
    t0 = omp_get_wtime();
    memcpy(rp, ap, a_size * sizeof(double));
#pragma omp parallel for
    for (int g = 0; g < ngroups; ++g)
        for (int v = 0; v < V; ++v)
            for (int j = 0; j < n; ++j)
                for (int i = j + 1; i < m; ++i)
                    rp[compact_at(g * V + v, i, j, m, n, V)] = 0.0;
    const double t_extract = omp_get_wtime() - t0;

    /* multiply: rp <- Q rp, Q applied from the left as the product of the k
     * reflectors in ap, without ever forming it. ap is read as the (m, k)
     * reflector batch: with m >= n that is exactly the buffer dgeqrf_compact
     * left behind. */
    t0 = omp_get_wtime();
    info = dormqr_compact('N', m, n, k, ap, m, taup, rp, m, V, nm);
    const double t_multiply = omp_get_wtime() - t0;
    if (info != 0) {
        fprintf(stderr, "dormqr_compact failed: info = %d\n", info);
        return EXIT_FAILURE;
    }

    /* unpack (the padded slots are dropped) */
    t0 = omp_get_wtime();
#pragma omp parallel for
    for (int g = 0; g < ngroups; ++g) {
        for (int v = 0; v < V; ++v) {
            const int idx = g * V + v;
            if (idx >= nm) continue;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    QR[dense_at(idx, i, j, m, n)] = rp[compact_at(idx, i, j, m, n, V)];
        }
    }
    const double t_unpack = omp_get_wtime() - t0;

    /* check, the one the Arm example applies: the 1-norm (max column sum) of
     * A - QR against 5 eps m n times the 1-norm of A */
    int fail = 0;
    for (int idx = 0; idx < nm; ++idx) {
        double norm_a = 0.0, norm_a_minus_qr = 0.0;
        for (int j = 0; j < n; ++j) {
            double sum_col_a = 0.0, sum_col_diff = 0.0;
            for (int i = 0; i < m; ++i) {
                const double a = A[dense_at(idx, i, j, m, n)];
                sum_col_a += fabs(a);
                sum_col_diff += fabs(a - QR[dense_at(idx, i, j, m, n)]);
            }
            if (sum_col_a > norm_a) norm_a = sum_col_a;
            if (sum_col_diff > norm_a_minus_qr) norm_a_minus_qr = sum_col_diff;
        }
        if (norm_a_minus_qr > 5.0 * DBL_EPSILON * m * n * norm_a) ++fail;
    }

    printf("  pack      %9.6f s\n", t_pack);
    printf("  factorize %9.6f s   (dgeqrf_compact)\n", t_factor);
    printf("  extract R %9.6f s\n", t_extract);
    printf("  multiply  %9.6f s   (dormqr_compact)\n", t_multiply);
    printf("  unpack    %9.6f s\n", t_unpack);
    printf("  total     %9.6f s\n", t_pack + t_factor + t_extract + t_multiply + t_unpack);
    if (fail == 0)
        printf("result check passed: ||A - QR||_1 <= 5 eps m n ||A||_1 for all %d "
               "matrices\n",
               nm);
    else
        printf("result check FAILED for %d of %d matrices\n", fail, nm);

    free(A);
    free(QR);
    free(ap);
    free(taup);
    free(rp);
    return fail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
