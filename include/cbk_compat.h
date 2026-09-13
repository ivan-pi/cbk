#ifndef CBK_COMPAT_H
#define CBK_COMPAT_H

/* cbk_compat.h -- batched QR, Cholesky, LDL^T, and triangular solve for
 * matrices in Intel MKL's Compact format.
 *
 *   cbk_?geqrf_compact   -- QR factorization of a Compact-format batch
 *   cbk_?ormqr_compact   -- apply Q (or Q^T) of a Compact-format QR
 *   cbk_?orgqr_compact   -- form the explicit Q of a Compact-format QR
 *   cbk_?potrf_compact   -- Cholesky factorization of an SPD Compact-format batch
 *   cbk_?potrs_compact   -- solve A X = B from that Cholesky factor
 *   cbk_?posv_compact    -- Cholesky factor + solve, fused per group
 *   cbk_?sytrfnp_compact -- unpivoted LDL^T factorization of a symmetric batch
 *   cbk_?sytrsnp_compact -- solve A X = B from that LDL^T factor
 *   cbk_?sysvnp_compact  -- LDL^T factor + solve, fused per group
 *   cbk_?trsm_compact    -- triangular solve op(A) X = alpha B (and variants)
 *   cbk_?gels_compact    -- least-squares / minimum-norm solve op(A) X = B, one call
 *
 * All take MKL's MKL_LAYOUT + MKL_COMPACT_PACK interface, so they drop into the
 * MKL compact ecosystem (pack with mkl_?gepack_compact, take `format` from
 * mkl_get_format_compact(), pass the total batch size `nm`), but are backed by
 * this project's portable SIMD kernels. cbk_?geqrf_compact,
 * cbk_?potrf_compact and cbk_?trsm_compact are signature- and
 * storage-compatible alternatives to the MKL routines of the same name, so they
 * mix freely with them; cbk_?ormqr_compact and cbk_?orgqr_compact are
 * the apply-Q and form-Q steps MKL omits (LAPACK ?ormqr / ?orgqr plus the
 * layout/format/nm arguments), cbk_?potrs_compact
 * and cbk_?posv_compact are the Cholesky solve and fused factor-and-solve
 * MKL likewise has no compact form of (LAPACK ?potrs / ?posv plus
 * layout/format/nm), and the three symmetric
 * "np" routines are the unpivoted LDL^T MKL has no compact form of at all (the
 * suffix follows MKL's own unpivoted mkl_?getrfnp_compact). Together they factor
 * and solve batched systems entirely in the compact format:
 *
 *     cbk_dgeqrf_compact (..., A -> H, tau);       // A = Q R
 *     cbk_dormqr_compact('L','T', ..., H, tau, B); // B := Q^T B
 *     cbk_dtrsm_compact  (..., U, R, B);           // B := R^{-1} Q^T B = X
 *
 *     cbk_dposv_compact  (..., A -> L, B);         // A = L L^T; B := A^{-1} B
 *
 *     cbk_dsysvnp_compact(..., A -> (L, D), B);   // A = L D L^T; B := A^{-1} B
 *
 * and cbk_?gels_compact is the QR chain -- generalized to over- and
 * underdetermined systems, LAPACK ?gels plus layout/format/nm -- as one call
 * that keeps each group of V matrices cache-resident from factorization to
 * solution:
 *
 *     cbk_dgels_compact('N', ..., A, B, work);     // B := X, A := (H, R)
 *
 * Conventions shared by all routines:
 *   - No argument checking (compact routines skip it for vectorization); the
 *     caller passes consistent parameters. Use the portable cbk.h
 *     entry points for LAPACK-style info = -j validation.
 *   - `info` is a single scalar status (MKL leaves the compact info reserved):
 *     0 on success, -1 for an unrecognized `format`.
 *   - Workspace: ?geqrf, ?ormqr and ?orgqr take work/lwork like their LAPACK
 *     namesakes;
 *     with lwork = -1 the call is a query returning the optimal lwork in
 *     work[0]. Today that is 1 -- the current kernels need no scratch -- but
 *     that is an implementation detail, not a contract, so never hard-code
 *     it. Always size work from a
 *     query of the routine you will call, and give each routine its own
 *     buffer: lwork is not checked (no compact routine checks it, MKL's
 *     included), so an undersized work array is a silent overrun. This matters
 *     when mixing cbk and MKL routines: the two libraries' workspace needs
 *     differ, and one routine's lwork says nothing about another's.
 *   - ?gels's work is the exception: it is the routine's tau scratch, one slot
 *     per group so the groups can run in parallel, so its query returns
 *     max(1, min(m,n) * V * ceil(nm/V)) -- the size in scalars of a compact tau
 *     buffer for the batch, mkl_?get_size_compact(min(m,n), 1, format, nm) /
 *     sizeof(scalar). On exit it holds the tau of the factorization left in ap.
 *   - The reflector batch A of ?ormqr is dimensioned (ldap, k) as in LAPACK, so
 *     the compact buffer must be packed with exactly k columns (group stride
 *     ldap*k*V). Square and tall factors from ?geqrf_compact satisfy this as
 *     they are; a wide factor (more columns than reflectors) must be repacked
 *     with its k reflector columns only.
 *
 *   - Threading: built with OpenMP, the loop over groups of V matrices runs in
 *     parallel (static schedule, one thread per group at most) whenever the
 *     call has two or more groups and enough work to pay for the fork; from
 *     inside a caller's own parallel loop it stays serial unless nested
 *     parallelism is enabled (OMP_NUM_THREADS=8,2 with OMP_MAX_ACTIVE_LEVELS=2).
 *     See the note in cbk.h.
 *
 * Per-routine parameter references:
 * docs/cbk_d{geqrf,ormqr,potrf,sytrfnp,trsm,gels}_compact_design.md.
 *
 * Assisted-by: Claude:claude-opus-4-8 Claude:claude-fable-5
 */

#include "mkl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* QR factorization: A -> (R, Householder vectors) in ap, tau in taup. Drop-in
 * for mkl_?geqrf_compact (identical signature). */
void cbk_dgeqrf_compact(MKL_LAYOUT layout, MKL_INT m, MKL_INT n, double *ap, MKL_INT ldap,
                        double *taup, double *work, MKL_INT lwork, MKL_INT *info,
                        MKL_COMPACT_PACK format, MKL_INT nm);

void cbk_sgeqrf_compact(MKL_LAYOUT layout, MKL_INT m, MKL_INT n, float *ap, MKL_INT ldap,
                        float *taup, float *work, MKL_INT lwork, MKL_INT *info,
                        MKL_COMPACT_PACK format, MKL_INT nm);

/* C := op(Q) C (side 'L') or C op(Q) (side 'R'), op(Q) = Q ('N') or Q^T
 * ('T'/'C'), from the (H, tau) of a compact QR: LAPACK ?ormqr plus layout,
 * format and nm. */
void cbk_dormqr_compact(MKL_LAYOUT layout, char side, char trans, MKL_INT m, MKL_INT n,
                        MKL_INT k, const double *ap, MKL_INT ldap, const double *taup,
                        double *cp, MKL_INT ldcp, double *work, MKL_INT lwork,
                        MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

void cbk_sormqr_compact(MKL_LAYOUT layout, char side, char trans, MKL_INT m, MKL_INT n,
                        MKL_INT k, const float *ap, MKL_INT ldap, const float *taup,
                        float *cp, MKL_INT ldcp, float *work, MKL_INT lwork,
                        MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

/* Generate the first n columns of Q = H(0) ... H(k-1) (m >= n >= k) from the
 * (H, tau) of a compact QR, in place over the reflectors: LAPACK ?orgqr plus
 * layout, format and nm. ap is packed m x n; on entry its columns 0..k-1 hold
 * the reflectors (columns k..n-1 need not be set), on exit each matrix holds
 * n orthonormal columns -- the factorization, R included, is overwritten. */
void cbk_dorgqr_compact(MKL_LAYOUT layout, MKL_INT m, MKL_INT n, MKL_INT k, double *ap,
                        MKL_INT ldap, const double *taup, double *work, MKL_INT lwork,
                        MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

void cbk_sorgqr_compact(MKL_LAYOUT layout, MKL_INT m, MKL_INT n, MKL_INT k, float *ap,
                        MKL_INT ldap, const float *taup, float *work, MKL_INT lwork,
                        MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

/* Cholesky factorization of a batch of symmetric positive-definite n x n
 * matrices: A = L L^T (MKL_LOWER) or A = U^T U (MKL_UPPER). Drop-in for
 * mkl_?potrf_compact (identical signature). On exit the named triangle holds
 * the factor; the other is untouched. No workspace and no SPD test: a non-SPD
 * lane poisons itself with NaN/Inf rather than reporting info = j. */
void cbk_dpotrf_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, double *ap,
                        MKL_INT ldap, MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

void cbk_spotrf_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, float *ap,
                        MKL_INT ldap, MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

/* Solve A X = B from the factor of cbk_?potrf_compact (same uplo), in
 * place: L z = B; L^T X = z (MKL_LOWER), or U^T z = B; U X = z (MKL_UPPER).
 * LAPACK ?potrs plus layout/format/nm; MKL has no compact potrs. B is n x nrhs
 * with ldbp >= n (column-major) or >= nrhs (row-major); the factor is not
 * modified. A factor lane poisoned by non-SPD input yields NaN/Inf in that
 * lane's X, not an error. */
void cbk_dpotrs_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, MKL_INT nrhs,
                        const double *ap, MKL_INT ldap, double *bp, MKL_INT ldbp,
                        MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

void cbk_spotrs_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, MKL_INT nrhs,
                        const float *ap, MKL_INT ldap, float *bp, MKL_INT ldbp,
                        MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

/* Symmetric positive-definite solve A X = B in one pass: cbk_?potrf_compact
 * followed by cbk_?potrs_compact, fused per group of V matrices so each
 * factor is solved with while cache-resident and the whole solve is one
 * threaded group loop. LAPACK ?posv plus layout/format/nm. On exit ap holds
 * the Cholesky factor exactly as ?potrf leaves it and bp holds X; the result
 * is bit-identical to the two separate calls. As in LAPACK ?posv, nrhs = 0
 * still factors ap (the nrhs quick return is ?potrs's); bp is then never
 * referenced. */
void cbk_dposv_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, MKL_INT nrhs,
                       double *ap, MKL_INT ldap, double *bp, MKL_INT ldbp, MKL_INT *info,
                       MKL_COMPACT_PACK format, MKL_INT nm);

void cbk_sposv_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, MKL_INT nrhs,
                       float *ap, MKL_INT ldap, float *bp, MKL_INT ldbp, MKL_INT *info,
                       MKL_COMPACT_PACK format, MKL_INT nm);

/* LDL^T factorization without pivoting of a batch of symmetric n x n matrices:
 * A = L D L^T (MKL_LOWER) or A = U^T D U (MKL_UPPER; the transpose dual, as
 * ?potrf's U^T U, not LAPACK ?sytrf's U D U^T). mkl_?potrf_compact's signature
 * (MKL has no compact sytrf). On exit the diagonal holds D and the strict
 * off-diagonal of the named triangle the unit L/U; the other triangle is
 * untouched. No workspace, no ipiv, no singularity test: indefinite matrices
 * factor (no sqrt), but a zero pivot -- a singular leading principal minor --
 * poisons its lane with Inf/NaN rather than reporting info = j. */
void cbk_dsytrfnp_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, double *ap,
                          MKL_INT ldap, MKL_INT *info, MKL_COMPACT_PACK format,
                          MKL_INT nm);

void cbk_ssytrfnp_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, float *ap,
                          MKL_INT ldap, MKL_INT *info, MKL_COMPACT_PACK format,
                          MKL_INT nm);

/* Solve A X = B from the factor of cbk_?sytrfnp_compact (same uplo), in
 * place: L z = B; D w = z; L^T X = w (MKL_LOWER), or U^T z = B; D w = z;
 * U X = w (MKL_UPPER). LAPACK ?sytrs minus ipiv, plus layout/format/nm. B is
 * n x nrhs with ldbp >= n (column-major) or >= nrhs (row-major); the factor is
 * not modified. A zero D(i) yields Inf/NaN in that lane's X, not an error. */
void cbk_dsytrsnp_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, MKL_INT nrhs,
                          const double *ap, MKL_INT ldap, double *bp, MKL_INT ldbp,
                          MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

void cbk_ssytrsnp_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, MKL_INT nrhs,
                          const float *ap, MKL_INT ldap, float *bp, MKL_INT ldbp,
                          MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

/* Symmetric solve A X = B in one pass: cbk_?sytrfnp_compact followed by
 * cbk_?sytrsnp_compact, fused per group of V matrices so each factor is
 * solved with while cache-resident and the whole solve is one threaded group
 * loop. LAPACK ?sysv minus ipiv and workspace. On exit ap holds the (D, L|U)
 * factor exactly as ?sytrfnp leaves it and bp holds X; the result is
 * bit-identical to the two separate calls. As in LAPACK ?sysv, nrhs = 0 still
 * factors ap (the nrhs quick return is ?sytrs's); bp is then never
 * referenced. */
void cbk_dsysvnp_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, MKL_INT nrhs,
                         double *ap, MKL_INT ldap, double *bp, MKL_INT ldbp,
                         MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

void cbk_ssysvnp_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, MKL_INT nrhs,
                         float *ap, MKL_INT ldap, float *bp, MKL_INT ldbp, MKL_INT *info,
                         MKL_COMPACT_PACK format, MKL_INT nm);

/* Triangular solve with multiple right-hand sides, in place:
 *     op(A) X = alpha B  (MKL_LEFT)   or   X op(A) = alpha B  (MKL_RIGHT),
 * A the order-s (s = m left, n right) unit/non-unit upper/lower triangular
 * factor, op(A) = A (MKL_NOTRANS) or A^T (MKL_TRANS / MKL_CONJTRANS, the same
 * for real types), B (m x n) overwritten by X. Drop-in for mkl_?trsm_compact
 * (identical signature): like the BLAS ?trsm it batches, no workspace, no info. */
void cbk_dtrsm_compact(MKL_LAYOUT layout, MKL_SIDE side, MKL_UPLO uplo,
                       MKL_TRANSPOSE transa, MKL_DIAG diag, MKL_INT m, MKL_INT n,
                       double alpha, const double *ap, MKL_INT ldap, double *bp,
                       MKL_INT ldbp, MKL_COMPACT_PACK format, MKL_INT nm);

void cbk_strsm_compact(MKL_LAYOUT layout, MKL_SIDE side, MKL_UPLO uplo,
                       MKL_TRANSPOSE transa, MKL_DIAG diag, MKL_INT m, MKL_INT n,
                       float alpha, const float *ap, MKL_INT ldap, float *bp,
                       MKL_INT ldbp, MKL_COMPACT_PACK format, MKL_INT nm);

/* Least-squares (op(A) with more rows than columns) or minimum-norm (more
 * columns than rows) solution of the full-rank systems op(A) X = B, op(A) = A
 * ('N') or A^T ('T'/'C'), A m x n, B max(m,n) x nrhs: LAPACK ?gels plus layout,
 * format and nm. On exit ap holds the QR (m >= n) or LQ (m < n) factorization
 * of A, B's leading rows the solution (and, for least squares, its trailing
 * m - n rows the residual), and work the reflector scalars tau. lwork = -1
 * queries the workspace (see the workspace note above); min(m,n) = 0 sets
 * B := 0. Rank deficiency is not detected (no info > 0): a zero diagonal of R
 * divides through to Inf/NaN in that lane. */
void cbk_dgels_compact(MKL_LAYOUT layout, char trans, MKL_INT m, MKL_INT n, MKL_INT nrhs,
                       double *ap, MKL_INT ldap, double *bp, MKL_INT ldbp, double *work,
                       MKL_INT lwork, MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

void cbk_sgels_compact(MKL_LAYOUT layout, char trans, MKL_INT m, MKL_INT n, MKL_INT nrhs,
                       float *ap, MKL_INT ldap, float *bp, MKL_INT ldbp, float *work,
                       MKL_INT lwork, MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

/* ------------------------------------------------------------------ *
 * C++-only internal helper (for this project's implementation, tests   *
 * and examples; not part of the FFI-stable C surface above). It uses   *
 * MKL only for its *types* (MKL_COMPACT_PACK) -- like the C API, it     *
 * needs the MKL headers but adds no MKL link dependency. The RAII       *
 * buffer helpers that *call* mkl_malloc / mkl_free live in the separate *
 * cbk_mkl_alloc.h, so including this header does not pull in the MKL    *
 * link line.                                                            *
 * ------------------------------------------------------------------ */

#include <cstddef>

namespace cbk::detail {

/* Interleave width V for a given MKL Compact pack format and scalar type T.
 * MKL packs V = (SIMD register bytes) / sizeof(T):
 *   SSE = 16 B, AVX = 32 B, AVX512 = 64 B.
 * Returns 0 for an unrecognised format. */
template <typename T> inline int vlen_for_format(MKL_COMPACT_PACK format)
{
    int bytes;
    switch (format) {
    case MKL_COMPACT_SSE: bytes = 16; break;
    case MKL_COMPACT_AVX: bytes = 32; break;
    case MKL_COMPACT_AVX512: bytes = 64; break;
    default: return 0;
    }
    return bytes / static_cast<int>(sizeof(T));
}

/* Inverse of vlen_for_format: the pack format whose interleave width for scalar
 * type T is v, i.e. v * sizeof(T) register bytes (16/32/64 -> SSE/AVX/AVX512).
 * Any other v returns MKL_COMPACT_SSE; callers validate v against the host's
 * native width before use. */
template <typename T> inline MKL_COMPACT_PACK format_for_vlen(int v)
{
    switch (v * static_cast<int>(sizeof(T))) {
    case 16: return MKL_COMPACT_SSE;
    case 32: return MKL_COMPACT_AVX;
    case 64: return MKL_COMPACT_AVX512;
    default: return MKL_COMPACT_SSE; /* unrecognised; caller validates */
    }
}

/* Human-readable name of the SIMD ISA behind an MKL Compact pack format
 * ("SSE"/"AVX"/"AVX512", or "unknown"). */
inline const char *compact_format_name(MKL_COMPACT_PACK format)
{
    switch (format) {
    case MKL_COMPACT_SSE: return "SSE";
    case MKL_COMPACT_AVX: return "AVX";
    case MKL_COMPACT_AVX512: return "AVX512";
    default: return "unknown";
    }
}

} /* namespace cbk::detail */

#endif /* __cplusplus */

#endif /* CBK_COMPAT_H */
