/*
 * qr_workflow_compact.c -- the interleave-batch QR workflow example that Arm
 * ships with Arm Performance Libraries (ib_blog_qr_example), on the portable
 * C API of this library (cbk.h). Plain C99, OpenMP for the timings and the
 * loops over groups, nothing else.
 *
 * The same workflow, step for step:
 *
 *   pack        column-major A_v  ->  compact ap             pack_compact
 *   factorize   dgeqrf_compact      ap <- (R, Householder vectors), taup <- tau
 *   extract R   copy ap, zero below the diagonal
 *   multiply    dormqr_compact      rp <- Q R
 *   unpack      compact rp  ->  column-major (QR)_v          unpack_compact
 *   check       norm1(A_v - Q_v R_v) <= 5 eps m n norm1(A_v) for every v
 *
 * with the batch described the way cbk (and Intel MKL's compact API) does:
 * nm, the number of matrices, and V, the interleave width -- 2, 4, 8 or 16,
 * the double-precision pack widths of SSE, AVX and AVX-512 being 2, 4 and 8.
 * The matrices are stored in groups of V; within a group element (i,j) of
 * the V matrices is contiguous, so a group of column-major m x n matrices is
 * an (ldap x n) array of V-wide packs, and cbk.h's layout formula is
 *
 *     A_v(i,j) = ap[ g*ldap*n*V + (j*ldap + i)*V + v ],   g = idx/V, v = idx%V
 *
 * A last group that V does not fill is padded with identity matrices, which
 * the kernels run unmasked. In ArmPL's terms ninter is V and nbatch the
 * number of groups, nm/V rounded up. ArmPL's pack and unpack routines have no
 * counterpart in cbk's C API, so pack_compact and unpack_compact are written
 * out below; ArmPL's rank-revealing QR becomes dgeqrf_compact (no column
 * pivoting, so no jpvt and no rank check) and its ormqr dormqr_compact.
 *
 * The LAPACK comparison of the original (run_lpk_version: the same work as
 * one dgeqrf and one dormqr per matrix from an OpenMP loop) is kept, so the
 * example needs a LAPACK, which the library itself does not: CMake links the
 * one find_package(LAPACK) locates (MKL's under -DCBK_WITH_MKL=ON) and skips
 * the example without one. The Fortran entry points are declared here (LP64:
 * 32-bit integers, and the hidden character-length arguments left off, as
 * the original does).
 *
 * The matrices must be square or tall (m >= n): dormqr_compact reads the
 * reflectors as an (ldap, k) batch, k = min(m,n), exactly as LAPACK dormqr's
 * A(LDA,K), which is the m x n buffer dgeqrf_compact left only when n == k
 * (the LAPACK half of the original passes n as K too).
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

void dgeqrf_(const int *m, const int *n, double *a, const int *lda, double *tau,
             double *work, const int *lwork, int *info);
void dormqr_(const char *side, const char *trans, const int *m, const int *n,
             const int *k, const double *a, const int *lda, const double *tau, double *c,
             const int *ldc, double *work, const int *lwork, int *info);

/*
 * Offset, within one compact column-major group, of element (i,j) of the
 * matrix in slot v (0 <= v < V), at leading dimension ldap: the in-group part
 * of the layout formula of cbk.h. Group g of a batch of m x n matrices starts
 * at ap[g*ldap*n*V]; the callers take that base pointer once per group.
 */
static size_t compact_at(int i, int j, int v, int ldap, int V)
{
    return ((size_t)j * ldap + i) * V + v;
}

/*
 * Pack and unpack between nm dense column-major m x n matrices, matrix idx at
 * A_lpk_p[idx*lda*n] with leading dimension lda, and the compact batch ap
 * with leading dimension ldap. The groups are independent, so the loop over
 * them is an OpenMP loop, as the original's loop over nbatch is. The slots of
 * a partial last group (idx >= nm) are packed as the identity and skipped on
 * unpacking.
 */
static void pack_compact(int nm, int V, int m, int n, const double *A_lpk_p, int lda,
                         double *ap, int ldap)
{
    const int ngroups = (nm + V - 1) / V;
#pragma omp parallel for
    for (int g = 0; g < ngroups; g++) {
        double *A_g = &ap[(size_t)g * ldap * n * V];
        for (int v = 0; v < V; v++) {
            const int idx = g * V + v;
            const double *A_lpk = &A_lpk_p[(size_t)idx * lda * n];
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    A_g[compact_at(i, j, v, ldap, V)] =
                        idx < nm ? A_lpk[lda * j + i] : (i == j ? 1.0 : 0.0);
                }
            }
        }
    }
}

static void unpack_compact(int nm, int V, int m, int n, const double *ap, int ldap,
                           double *A_lpk_p, int lda)
{
    const int ngroups = (nm + V - 1) / V;
#pragma omp parallel for
    for (int g = 0; g < ngroups; g++) {
        const double *A_g = &ap[(size_t)g * ldap * n * V];
        for (int v = 0; v < V; v++) {
            const int idx = g * V + v;
            if (idx >= nm) continue;
            double *A_lpk = &A_lpk_p[(size_t)idx * lda * n];
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    A_lpk[lda * j + i] = A_g[compact_at(i, j, v, ldap, V)];
                }
            }
        }
    }
}

/*
 * This function uses the compact-batch functions. Starting from a batch of
 * matrices laid out in the standard LAPACK column-major format, it packs them
 * into the compact format, performs QR factorization, extracts the R
 * factors into a separate array, then multiplies Q by R before unpacking back
 * into LAPACK format and optionally checks the result.
 * The function returns the time taken in seconds, or a negative value if a
 * routine failed or the result check did not pass.
 */
double run_ib_version(int nm, int V, int m, int n, int check_result)
{
    int min_mn = MIN(m, n);
    int ngroups = (nm + V - 1) / V; /* the last one padded if V does not divide nm */

    /*
       Compact-batch setup: A is m by n, stored with leading dimension m, so a
       group is m*n packs of V; tau is min_mn scalars per matrix, a compact
       batch of min_mn x 1 vectors.
    */
    int ldap = m;
    size_t total_size_A = (size_t)ngroups * ldap * n * V;
    size_t total_size_tau = (size_t)ngroups * min_mn * V;

    int info;

    /* Compact-batch arrays: the matrices, R, and tau */
    double *ap = (double *)malloc(sizeof(double) * total_size_A);
    double *rp = (double *)malloc(sizeof(double) * total_size_A);
    double *taup = (double *)malloc(sizeof(double) * total_size_tau);

    /* Pack from LAPACK arrays at the start, and unpack back at the end */
    int lda = m;
    double *A_lpk_p = (double *)malloc(sizeof(double) * lda * n * nm);
    int ldqr = m;
    double *QR_lpk_p = (double *)malloc(sizeof(double) * ldqr * n * nm);

    if (!ap || !rp || !taup || !A_lpk_p || !QR_lpk_p) {
        fprintf(stderr, "Error allocating the batch, exit.\n");
        return -1.0;
    }

    srand(4733);
    /* Populate the LAPACK format matrices with random values in [0,1) */
    for (size_t i = 0; i < (size_t)m * n * nm; i++) {
        A_lpk_p[i] = (double)rand() / RAND_MAX;
    }

    /*
       Now time packing the matrices, doing QR factorization followed by
       multiplying Q by R, and unpack back out
    */
    double t1_ib = omp_get_wtime();

    /* Pack matrices into compact format */
    double t1_ib_pack = omp_get_wtime();
    pack_compact(nm, V, m, n, A_lpk_p, lda, ap, ldap);
    double t2_ib_pack = omp_get_wtime();

    double t1_ib_qr = omp_get_wtime();
    /* Perform QR factorizations */
    info = dgeqrf_compact('C', m, n, ap, ldap, taup, V, nm);
    if (info != 0) {
        fprintf(stderr, "Error performing compact-batch QR factorization, exit.\n");
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
    memcpy((void *)rp, (void *)ap, sizeof(double) * total_size_A);

    /* Zero lower-triangular part of R, padded slots included */
#pragma omp parallel for
    for (int g = 0; g < ngroups; g++) {
        double *R_g = &rp[(size_t)g * ldap * n * V];
        for (int j = 0; j < n; j++) {
            for (int i = j + 1; i < m; i++) {
                for (int v = 0; v < V; v++) {
                    R_g[compact_at(i, j, v, ldap, V)] = 0.0;
                }
            }
        }
    }

    /* Multiply Q by R */
    char transQ = 'N';
    info = dormqr_compact(transQ, m, n, min_mn, ap, ldap, taup, rp, ldap, V, nm);
    if (info != 0) {
        fprintf(stderr, "Error in multiplying by Q matrix, exit.\n");
        return -1.0;
    }
    double t2_ib_mq = omp_get_wtime();

    /* Unpack matrices from compact format */
    double t1_ib_unpack = omp_get_wtime();
    unpack_compact(nm, V, m, n, rp, ldap, QR_lpk_p, ldqr);
    double t2_ib_unpack = omp_get_wtime();

    /* Check the result, matrix by matrix */
    const double eps = nextafter(1.0, 2.0) - 1.0;
    int fail = 0;
    if (check_result) {
#pragma omp parallel for
        for (int idx = 0; idx < nm; idx++) {
            /* Compute 1-norms of original matrix A and computed QR */
            double *A_lpk = &A_lpk_p[(size_t)idx * lda * n];
            double *QR_lpk = &QR_lpk_p[(size_t)idx * ldqr * n];
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

    double t2_ib = omp_get_wtime();

    if (check_result) {
        if (fail == 0) {
            printf("Compact-batch result check passed: ");
            printf("norm1(A-QR) < eps*n*norm1(A) for all cases.\n");
        }
        else {
            printf("Compact-batch result check failed:\n");
            printf("\tnumber of cases where norm1(A-QR) > eps*n*norm1(A) = ");
            printf("%d.\n", fail);
        }
    }

    if (check_result) {
        printf("Compact-batch breakdown:\n");
        printf("\tpack: %f\n", t2_ib_pack - t1_ib_pack);
        printf("\tfactorize: %f\n", t2_ib_qr - t1_ib_qr);
        printf("\tmultiply: %f\n", t2_ib_mq - t1_ib_mq);
        printf("\tunpack: %f\n", t2_ib_unpack - t1_ib_unpack);
    }

    free(ap);
    free(rp);
    free(taup);
    free(A_lpk_p);
    free(QR_lpk_p);

    return fail == 0 ? t2_ib - t1_ib : -1.0;
}

/*
 * This function does the same as above with standard LAPACK calls, one
 * matrix at a time from a parallel loop, and returns the time taken in
 * seconds (negative if a routine failed or the result check did not pass).
 * V plays no part; it is taken so both versions are called alike.
 */
double run_lpk_version(int nm, int V, int m, int n, int check_result)
{
    (void)V;

    /* LAPACK setup */
    int lda = m;
    double *A_lpk_p = (double *)malloc(sizeof(double) * lda * n * nm);
    double *R_lpk_p = (double *)malloc(sizeof(double) * lda * n * nm);
    double *A_orig_lpk_p = (double *)malloc(sizeof(double) * lda * n * nm);
    int ldqr = m;
    int nthreads = omp_get_max_threads();
    double *tau_lpk_p = (double *)malloc(sizeof(double) * n * nthreads);

    if (!A_lpk_p || !R_lpk_p || !A_orig_lpk_p || !tau_lpk_p) {
        fprintf(stderr, "Error allocating the batch, exit.\n");
        return -1.0;
    }

    srand(4733);
    /* Populate the LAPACK format matrices with random values in [0,1) */
    for (size_t i = 0; i < (size_t)m * n * nm; i++) {
        A_lpk_p[i] = (double)rand() / RAND_MAX;
        A_orig_lpk_p[i] = A_lpk_p[i];
    }

    /* Workspace query, for both routines, one buffer per thread */
    int lwork = -1;
    double dlwork_qr, dlwork_mq;
    int info;
    char side = 'L';
    char transQ = 'N';
    dgeqrf_(&m, &n, A_lpk_p, &lda, tau_lpk_p, &dlwork_qr, &lwork, &info);
    dormqr_(&side, &transQ, &m, &n, &n, A_lpk_p, &lda, tau_lpk_p, R_lpk_p, &lda,
            &dlwork_mq, &lwork, &info);
    lwork = (int)MAX(dlwork_qr, dlwork_mq);
    if (lwork < 1) {
        fprintf(stderr, "Error: LAPACK workspace query failed. Exiting.\n");
        return -1.0;
    }
    double *work_lpk_p = (double *)malloc(sizeof(double) * lwork * nthreads);
    if (!work_lpk_p) {
        fprintf(stderr, "Error allocating the workspace, exit.\n");
        return -1.0;
    }

    const double eps = nextafter(1.0, 2.0) - 1.0;
    int fail = 0;
    int error = 0;

    /* Time a parallelized loop over equivalent LAPACK calls */
    double t1_lpk = omp_get_wtime();
#pragma omp parallel for
    for (int idx = 0; idx < nm; idx++) {
        double *A_lpk = &A_lpk_p[(size_t)idx * lda * n];
        double *work_lpk = &work_lpk_p[(size_t)lwork * omp_get_thread_num()];
        double *tau_lpk = &tau_lpk_p[(size_t)n * omp_get_thread_num()];
        int info_t;

        dgeqrf_(&m, &n, A_lpk, &lda, tau_lpk, work_lpk, &lwork, &info_t);
        if (info_t != 0) {
            fprintf(stderr, "Error performing LAPACK QR factorization, exit.\n");
#pragma omp atomic update
            error++;
        }

        /* Extract R */
        double *R_lpk = &R_lpk_p[(size_t)idx * lda * n];
        memcpy((void *)R_lpk, (void *)A_lpk, sizeof(double) * lda * n);

        for (int j = 0; j < n; j++) {
            for (int i = j + 1; i < m; i++) {
                R_lpk[j * lda + i] = 0.0;
            }
        }

        dormqr_(&side, &transQ, &m, &n, &n, A_lpk, &lda, tau_lpk, R_lpk, &lda, work_lpk,
                &lwork, &info_t);
        if (info_t != 0) {
            fprintf(stderr, "Error multiplying LAPACK QR, exit.\n");
#pragma omp atomic update
            error++;
        }

        if (check_result) {
            double *A_orig = &A_orig_lpk_p[(size_t)idx * lda * n];
            double norm_a_minus_qr = 0.0;
            double norm_a = 0.0;
            for (int j = 0; j < n; j++) {
                double sum_diff = 0.0;
                double sum_col_a = 0.0;
                for (int i = 0; i < m; i++) {
                    sum_diff += fabs(A_orig[lda * j + i] - R_lpk[ldqr * j + i]);
                    sum_col_a += fabs(A_orig[lda * j + i]);
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
    double t2_lpk = omp_get_wtime();

    if (check_result) {
        if (fail == 0) {
            printf("LAPACK result check passed: ");
            printf("norm1(A-QR) < eps*n*norm1(A) for all cases.\n");
        }
        else {
            printf("LAPACK result check failed:\n");
            printf("\tnumber of cases where norm1(A-QR) > eps*n*norm1(A) = ");
            printf("%d.\n", fail);
        }
    }

    free(A_lpk_p);
    free(R_lpk_p);
    free(A_orig_lpk_p);
    free(tau_lpk_p);
    free(work_lpk_p);

    return (error == 0 && fail == 0) ? t2_lpk - t1_lpk : -1.0;
}

int main(int argc, char **argv)
{

    if (argc != 5) {
        fprintf(stderr,
                "Error: requires 4 command-line arguments, but %d were provided.\n",
                argc - 1);
        fprintf(stderr, "Usage: ./qr_workflow_compact <nm> <V> <nrows> <ncols>.\n");
        return EXIT_FAILURE;
    }

    int nm = atoi(argv[1]);
    int V = atoi(argv[2]);
    int m = atoi(argv[3]);
    int n = atoi(argv[4]);

    if (V != 2 && V != 4 && V != 8 && V != 16) {
        fprintf(stderr, "Error: V must be 2, 4, 8 or 16 (the cbk interleave widths).\n");
        return EXIT_FAILURE;
    }
    if (nm < 1 || n < 1 || m < n) {
        fprintf(stderr, "Error: requires nm >= 1 and nrows >= ncols >= 1.\n");
        return EXIT_FAILURE;
    }

    printf("Running example with nm = %d, V = %d, m = %d, n = %d\n", nm, V, m, n);
    printf("Number of groups: %d%s\n", (nm + V - 1) / V,
           nm % V ? " (the last one padded)" : "");

    double t_ib;
    double t_lpk;

    /* Warm-up runs */
    for (int nw = 0; nw < 3; nw++) {
        t_ib = run_ib_version(nm, V, m, n, 0);
        if (t_ib < 0) return EXIT_FAILURE;
        t_lpk = run_lpk_version(nm, V, m, n, 0);
        if (t_lpk < 0) return EXIT_FAILURE;
    }

    /* Reported runs */
    t_ib = run_ib_version(nm, V, m, n, 1);
    if (t_ib < 0) return EXIT_FAILURE;
    t_lpk = run_lpk_version(nm, V, m, n, 1);
    if (t_lpk < 0) return EXIT_FAILURE;

    printf("Time for compact-batch computation: %f\n", t_ib);
    printf("Time for LAPACK computation: %f\n", t_lpk);
    printf("Speedup for compact-batch over LAPACK: %f\n", t_lpk / t_ib);

    return EXIT_SUCCESS;
}
