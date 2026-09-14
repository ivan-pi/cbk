/*
 * qr_workflow_compact.c -- the interleave-batch QR workflow example that Arm
 * ships with Arm Performance Libraries (ib_blog_qr_example), on the portable
 * C API of this library (cbk.h). Plain C99, OpenMP for the timings and the
 * pack/unpack loops, nothing else.
 *
 * Kept as close to the original as the two APIs allow:
 *
 *   armpl_dge_interleave / _deinterleave  ->  dge_interleave / dge_deinterleave
 *                                             (written out below: cbk has no
 *                                             pack helpers in its C API)
 *   armpl_dgeqrfrr_interleave_batch       ->  dgeqrf_compact (no column
 *                                             pivoting: no jpvt, no rank)
 *   armpl_dormqr_interleave_batch         ->  dormqr_compact
 *
 * ArmPL's strides map one to one onto the compact layout of cbk.h with
 * istrd = ninter (= V, the interleave width), jstrd = ninter*m (= ldap*V),
 * bstrd = jstrd*n (the group stride), so the stride setup and the indexing
 * of the original are kept verbatim. ArmPL's nbatch is the number of groups
 * and ninter the interleave width, which cbk restricts to 2, 4, 8 or 16; the
 * total number of matrices nm = nbatch*ninter has no padded last group. The
 * matrices must be square or tall (m >= n): dormqr_compact reads the
 * reflectors as an (ldap, k) batch, k = min(m,n), exactly as LAPACK
 * dormqr's A(LDA,K), which is the m x n buffer dgeqrf_compact left only when
 * n == k (the LAPACK half of the original passes n as K too).
 *
 * The LAPACK comparison of the original (run_lpk_version) is left out: the
 * portable build links no LAPACK.
 *
 * Assisted-by: Claude:claude-fable-5
 */
#include "cbk.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

/*
 * The pack/unpack helpers, with the signatures of armpl_dge_interleave and
 * armpl_dge_deinterleave: matrix ii of an interleaved group of ninter, whose
 * element (i,j) sits at A_ib[i*istrd + j*jstrd + ii], against one dense
 * matrix A whose element (i,j) sits at A[i*row_strd + j*col_strd].
 */
static void dge_interleave(int ninter, int ii, int m, int n, const double *A,
                           int row_strd, int col_strd, double *A_ib, int istrd, int jstrd)
{
    (void)ninter;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            A_ib[i * istrd + j * jstrd + ii] = A[i * row_strd + j * col_strd];
        }
    }
}

static void dge_deinterleave(int ninter, int ii, int m, int n, double *A, int row_strd,
                             int col_strd, const double *A_ib, int istrd, int jstrd)
{
    (void)ninter;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            A[i * row_strd + j * col_strd] = A_ib[i * istrd + j * jstrd + ii];
        }
    }
}

/*
 * This function uses interleave-batch functions. Starting from a batch of
 * matrices laid out in the standard LAPACK column-major format, it packs them
 * into the interleaved format, performs QR factorization, extracts the R
 * factors into a separate array, then multiplies Q by R before unpacking back
 * into LAPACK format and optionally checks the result.
 * The function returns the time taken in seconds, or a negative value if a
 * routine failed or the result check did not pass.
 */
double run_ib_version(int nbatch, int ninter, int m, int n, int check_result)
{
    int min_mn = MIN(m, n);
    int total_matrices = nbatch * ninter;

    /*
       Interleaved-batch setup
       Set strides without any padding

       Use a "column-major" layout for the input matrices
       (i.e. istrd_A matches ninter)
       A is m by n
    */
    int istrd_A = ninter;
    int jstrd_A = istrd_A * m;
    int bstrd_A = jstrd_A * n;
    size_t total_size_A = (size_t)bstrd_A * nbatch;

    int istrd_R = ninter;
    int jstrd_R = istrd_R * m;
    int bstrd_R = jstrd_R * n;
    size_t total_size_R = (size_t)bstrd_R * nbatch;

    /* tau is the array of scalar factors of elementary reflectors */
    int istrd_tau = ninter;
    int bstrd_tau = istrd_tau * min_mn;
    size_t total_size_tau = (size_t)bstrd_tau * nbatch;

    int info;

    /* Interleave-batch arrays */
    double *A_ib_p = (double *)malloc(sizeof(double) * total_size_A);
    double *R_ib_p = (double *)malloc(sizeof(double) * total_size_R);
    double *tau_p = (double *)malloc(sizeof(double) * total_size_tau);

    /* Pack from LAPACK arrays at the start, and unpack back at the end */
    int lda = m;
    double *A_lpk_p = (double *)malloc(sizeof(double) * lda * n * total_matrices);
    int ldqr = m;
    double *QR_lpk_p = (double *)malloc(sizeof(double) * ldqr * n * total_matrices);

    if (!A_ib_p || !R_ib_p || !tau_p || !A_lpk_p || !QR_lpk_p) {
        fprintf(stderr, "Error allocating the batch, exit.\n");
        return -1.0;
    }

    srand(4733);
    /* Populate the LAPACK format matrices with random values in [0,1) */
    for (size_t i = 0; i < (size_t)m * n * total_matrices; i++) {
        A_lpk_p[i] = (double)rand() / RAND_MAX;
    }

    /*
       Now time packing the matrices, doing QR factorization followed by
       multiplying Q by R, and unpack back out
    */
    double t1_ib = omp_get_wtime();

    /* Pack matrices into interleaved-batch format, one by one */
    double t1_ib_pack = omp_get_wtime();
#pragma omp parallel for
    for (int ib = 0; ib < nbatch; ib++) {

        double *A_ib = &A_ib_p[(size_t)ib * bstrd_A];

        for (int ii = 0; ii < ninter; ii++) {

            double *A_lpk = &A_lpk_p[(size_t)(ib * ninter + ii) * lda * n];
            int row_strd = 1; /* LAPACK matrices are always col-major */
            int col_strd = lda;
            dge_interleave(ninter, ii, m, n, A_lpk, row_strd, col_strd, A_ib, istrd_A,
                           jstrd_A);
        }
    }
    double t2_ib_pack = omp_get_wtime();

    double t1_ib_qr = omp_get_wtime();
    /* Perform QR factorizations */
    info = dgeqrf_compact('C', m, n, A_ib_p, m, tau_p, ninter, total_matrices);
    if (info != 0) {
        fprintf(stderr, "Error performing interleave-batch QR factorization, exit.\n");
        return -1.0;
    }
    double t2_ib_qr = omp_get_wtime();

    /* Reconstruct A: extract R, then multiply it by Q (applied as its
       reflectors, never formed). No rank check: cbk's QR does not pivot. */
    double t1_ib_mq = omp_get_wtime();
    if (check_result) {
        printf("Info: reconstructing A.\n");
    }

    /* Make a copy of R */
    memcpy((void *)R_ib_p, (void *)A_ib_p, sizeof(double) * total_size_A);

    /* Zero lower-triangular part of R */
#pragma omp parallel for
    for (int ib = 0; ib < nbatch; ib++) {
        for (int j = 0; j < n; j++) {
            for (int i = j + 1; i < m; i++) {
                for (int ii = 0; ii < ninter; ii++) {
                    R_ib_p[(size_t)ib * bstrd_A + j * jstrd_A + i * istrd_A + ii] = 0.0;
                }
            }
        }
    }

    /* Multiply Q by R */
    char transQ = 'N';
    info = dormqr_compact(transQ, m, n, min_mn, A_ib_p, m, tau_p, R_ib_p, m, ninter,
                          total_matrices);
    if (info != 0) {
        fprintf(stderr, "Error in multiplying by Q matrix, exit.\n");
        return -1.0;
    }
    double t2_ib_mq = omp_get_wtime();

    double t1_ib_unpack = omp_get_wtime();
    const double eps = nextafter(1.0, 2.0) - 1.0;
    int fail = 0;

    /* Unpack matrices from interleaved-batch format, one by one */
#pragma omp parallel for
    for (int ib = 0; ib < nbatch; ib++) {

        double *QR_ib = &R_ib_p[(size_t)ib * bstrd_A];

        for (int ii = 0; ii < ninter; ii++) {

            double *QR_lpk = &QR_lpk_p[(size_t)(ib * ninter + ii) * ldqr * n];
            int row_strd = 1;
            int col_strd = ldqr;
            dge_deinterleave(ninter, ii, m, n, QR_lpk, row_strd, col_strd, QR_ib, istrd_A,
                             jstrd_A);
            if (check_result) {
                /* Compute 1-norms of original matrix A and computed QR */
                double *A_lpk = &A_lpk_p[(size_t)(ib * ninter + ii) * lda * n];
                double norm_a_minus_qr = 0.0;
                double norm_a = 0.0;
                for (int j = 0; j < n; j++) {
                    double sum_diff = 0.0;
                    double sum_col_a = 0.0;
                    for (int i = 0; i < m; i++) {
                        sum_diff += fabs(A_lpk[lda * j + i] - QR_lpk[ldqr * j + i]);
                        sum_col_a += fabs(A_lpk[lda * j + i]);
                    }
                    norm_a_minus_qr = MAX(sum_diff, norm_a_minus_qr);
                    norm_a = MAX(sum_col_a, norm_a);
                }

                /* Check that norm1(A-QR) <= eps*m*n*norm1(A) */
                double tol = 5.0 * eps * m * n * norm_a;
                if (norm_a_minus_qr > tol) {
#pragma omp atomic update
                    fail++;
                }
            }
        }
    }
    double t2_ib_unpack = omp_get_wtime();

    double t2_ib = omp_get_wtime();

    if (check_result) {
        if (fail == 0) {
            printf("Interleave-batch result check passed: ");
            printf("norm1(A-QR) < eps*n*norm1(A) for all cases.\n");
        }
        else {
            printf("Interleave-batch result check failed:\n");
            printf("\tnumber of cases where norm1(A-QR) > eps*n*norm1(A) = ");
            printf("%d.\n", fail);
        }
    }

    if (check_result) {
        printf("Interleave-batch breakdown:\n");
        printf("\tpack: %f\n", t2_ib_pack - t1_ib_pack);
        printf("\tfactorize: %f\n", t2_ib_qr - t1_ib_qr);
        printf("\tmultiply: %f\n", t2_ib_mq - t1_ib_mq);
        printf("\tunpack: %f\n", t2_ib_unpack - t1_ib_unpack);
    }

    free(A_ib_p);
    free(R_ib_p);
    free(tau_p);
    free(A_lpk_p);
    free(QR_lpk_p);

    return fail == 0 ? t2_ib - t1_ib : -1.0;
}

int main(int argc, char **argv)
{

    if (argc != 5) {
        fprintf(stderr,
                "Error: requires 4 command-line arguments, but %d were provided.\n",
                argc - 1);
        fprintf(stderr,
                "Usage: ./qr_workflow_compact <nbatch> <ninter> <nrows> <ncols>.\n");
        return EXIT_FAILURE;
    }

    int nbatch = atoi(argv[1]);
    int ninter = atoi(argv[2]);
    int m = atoi(argv[3]);
    int n = atoi(argv[4]);

    if (ninter != 2 && ninter != 4 && ninter != 8 && ninter != 16) {
        fprintf(stderr,
                "Error: ninter must be 2, 4, 8 or 16 (the cbk interleave widths).\n");
        return EXIT_FAILURE;
    }
    if (nbatch < 1 || n < 1 || m < n) {
        fprintf(stderr, "Error: requires nbatch >= 1 and nrows >= ncols >= 1.\n");
        return EXIT_FAILURE;
    }

    printf("Running example with nbatch = %d, ninter = %d, m = %d, n = %d\n", nbatch,
           ninter, m, n);
    printf("Total number of matrices: %d\n", nbatch * ninter);

    double t_ib;

    /* Warm-up runs */
    for (int nw = 0; nw < 3; nw++) {
        t_ib = run_ib_version(nbatch, ninter, m, n, 0);
        if (t_ib < 0) return EXIT_FAILURE;
    }

    /* Reported run */
    t_ib = run_ib_version(nbatch, ninter, m, n, 1);
    if (t_ib < 0) return EXIT_FAILURE;

    printf("Time for interleave-batch computation: %f\n", t_ib);

    return EXIT_SUCCESS;
}
