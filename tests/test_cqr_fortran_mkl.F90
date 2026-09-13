! Fortran-interface test for the MKL-style API (include/cqr_mkl_ext.fi),
! compiled as free-form source; test_cqr_fortran_mkl_fixed.F includes the
! same file as fixed form. Every entry point is called through its bind(c)
! generic name -- the enumerators of cqr_mkl_enums.fi select layout, uplo,
! side, trans, diag and the pack format -- and each solve is checked
! against the exact x the right-hand sides were built from; nothing here
! calls or links MKL itself. The point is that the interface blocks match
! the C signatures and the link line closes, not the numerics (the C++
! suites own those). CMake compiles this source once per work precision
! wp (CQR_SINGLE selects c_float); MKL_COMPACT_SSE packs 128 bits, V = 2
! doubles or 4 floats, so nmat = 3 leaves a padding lane either way. The
! scaffolding (check, fill, pack_c) is shared with the portable test:
! test_cqr_fortran_util.inc.
!
! Compiled with CQR_ILP64 defined (an ilp64 library build,
! -DMKLCompact_INTERFACE=ilp64) the same test runs through
! cqr_mkl_ext_ilp64.fi: the include file and the integer kind ik of
! every MKL_INT actual argument are the only differences.
!
! Assisted-by: Claude:claude-fable-5

program test_cqr_fortran_mkl
   use, intrinsic :: iso_c_binding, only: c_int, c_long_long, c_double, &
                                          c_float
   implicit none
#ifdef CQR_ILP64
   include 'cqr_mkl_ext_ilp64.fi'
   integer, parameter :: ik = c_long_long ! MKL_INT under MKL_ILP64
#else
   include 'cqr_mkl_ext.fi'
   integer, parameter :: ik = c_int       ! LP64 MKL_INT
#endif

#ifdef CQR_SINGLE
   integer, parameter :: wp = c_float ! work precision of this build
#else
   integer, parameter :: wp = c_double
#endif

   integer(ik), parameter :: nmat = 3, n = 3, nrhs = 2
   ! MKL_COMPACT_SSE: a 128-bit register of wp lanes.
   integer(ik), parameter :: vw = int(128 / storage_size(1.0_wp), ik)
   integer(ik), parameter :: ng = (nmat + vw - 1) / vw ! compact groups
   real(wp), parameter :: tol = 1.0e4_wp * epsilon(1.0_wp)

   call run_tests()
   call test_format_check()
   print '(a,i0,a)', 'test_cqr_fortran_mkl: all checks passed (', &
      storage_size(1.0_wp), '-bit reals)'

contains

   include 'test_cqr_fortran_util.inc'

   subroutine run_tests()
      real(wp) :: a(n, n, nmat), x(n, nrhs, nmat), b(n, nrhs, nmat)
      real(wp) :: eye(n, n), zed(n, nrhs)
      real(wp), target :: ap(vw, n, n, ng), bp(vw, n, nrhs, ng)
      real(wp), target :: taup(vw, n, ng)
      real(wp), target :: aq(vw, n, n, ng), bq(vw, n, n, ng)
      real(wp) :: xp(vw, n, nrhs, ng)
      real(wp) :: af(vw, n, n, ng), xf(vw, n, nrhs, ng)
      real(wp), pointer :: ap1(:), bp1(:), taup1(:)
      real(wp), pointer :: aq1(:), bq1(:)
      ! One workspace per routine, each sized by its own lwork = -1 query
      ! into probe -- never shared and never hand-sized (see the workspace
      ! contract in cqr_mkl_ext.h: lwork is not checked).
      real(wp) :: probe(1)
      real(wp), allocatable :: qwork(:), owork(:), gwork(:)
      real(wp), allocatable :: ogwork(:)
      integer(ik) :: info

      call fill(a, x, b, eye, zed)

      ! Rank-1 views for the generic calls: the interfaces declare the
      ! compact buffers assumed-size, and generic resolution matches rank.
      ap1(1:size(ap)) => ap
      bp1(1:size(bp)) => bp
      taup1(1:size(taup)) => taup
      aq1(1:size(aq)) => aq
      bq1(1:size(bq)) => bq

      ! The expected compact solution: x in the real lanes, zero in the
      ! padding lanes (identity system, zero right-hand side).
      xp = pack_c(x, zed)

      ! QR chain: geqrf -> ormqr (Q^T) -> trsm (R), workspace queries
      ! first. These kernels need no scratch, so each query reports 1.
      ap = pack_c(a, eye)
      bp = pack_c(b, zed)
      call cqr_mkl_geqrf_compact(MKL_COL_MAJOR, n, n, ap1, n, taup1, &
                                 probe, -1_ik, info, MKL_COMPACT_SSE, nmat)
      call check(info == 0 .and. nint(probe(1)) == 1, 'geqrf lwork query')
      allocate(qwork(nint(probe(1))))
      call cqr_mkl_geqrf_compact(MKL_COL_MAJOR, n, n, ap1, n, taup1, &
                                 qwork, int(size(qwork), ik), info, &
                                 MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_geqrf_compact info')
      call cqr_mkl_ormqr_compact(MKL_COL_MAJOR, 'L', 'T', n, nrhs, n, &
                                 ap1, n, taup1, bp1, n, probe, -1_ik, &
                                 info, MKL_COMPACT_SSE, nmat)
      call check(info == 0 .and. nint(probe(1)) == 1, 'ormqr lwork query')
      allocate(owork(nint(probe(1))))
      call cqr_mkl_ormqr_compact(MKL_COL_MAJOR, 'L', 'T', n, nrhs, n, &
                                 ap1, n, taup1, bp1, n, owork, &
                                 int(size(owork), ik), info, &
                                 MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_ormqr_compact info')
      call cqr_mkl_trsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, &
                                MKL_NOTRANS, MKL_NONUNIT, n, nrhs, &
                                1.0_wp, ap1, n, bp1, n, &
                                MKL_COMPACT_SSE, nmat)
      call check(maxval(abs(bp - xp)) < tol, 'QR chain solution')

      ! gels: its work is the tau scratch, one slot per group -- the query
      ! reports min(m,n) * V * ceil(nm/V) slots, a compact tau buffer.
      ap = pack_c(a, eye)
      bp = pack_c(b, zed)
      call cqr_mkl_gels_compact(MKL_COL_MAJOR, 'N', n, n, nrhs, ap1, n, &
                                bp1, n, probe, -1_ik, info, &
                                MKL_COMPACT_SSE, nmat)
      call check(info == 0 .and. nint(probe(1)) == n * vw * ng, &
                 'gels lwork query')
      allocate(gwork(nint(probe(1))))
      call cqr_mkl_gels_compact(MKL_COL_MAJOR, 'N', n, n, nrhs, ap1, n, &
                                bp1, n, gwork, int(size(gwork), ik), &
                                info, MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_gels_compact info')
      call check(maxval(abs(bp - xp)) < tol, 'gels solution')

      ! orgqr: the explicit Q must equal Q applied to the identity.
      ap = pack_c(a, eye)
      call cqr_mkl_geqrf_compact(MKL_COL_MAJOR, n, n, ap1, n, taup1, &
                                 qwork, int(size(qwork), ik), info, &
                                 MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_geqrf_compact info (for orgqr)')
      aq = ap
      call cqr_mkl_orgqr_compact(MKL_COL_MAJOR, n, n, n, aq1, n, &
                                 taup1, probe, -1_ik, info, &
                                 MKL_COMPACT_SSE, nmat)
      call check(info == 0 .and. nint(probe(1)) == 1, 'orgqr lwork query')
      allocate(ogwork(nint(probe(1))))
      call cqr_mkl_orgqr_compact(MKL_COL_MAJOR, n, n, n, aq1, n, &
                                 taup1, ogwork, int(size(ogwork), ik), &
                                 info, MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_orgqr_compact info')
      bq = pack_c(spread(eye, 3, int(nmat)), eye)
      call cqr_mkl_ormqr_compact(MKL_COL_MAJOR, 'L', 'N', n, n, n, &
                                 ap1, n, taup1, bq1, n, owork, &
                                 int(size(owork), ik), info, &
                                 MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_ormqr_compact (N) info')
      call check(maxval(abs(aq - bq)) < tol, 'orgqr == ormqr on identity')

      ! Cholesky: potrf, then two triangular solves close A x = b.
      ap = pack_c(a, eye)
      bp = pack_c(b, zed)
      call cqr_mkl_potrf_compact(MKL_COL_MAJOR, MKL_LOWER, n, ap1, n, &
                                 info, MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_potrf_compact info')
      call cqr_mkl_trsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, &
                                MKL_NOTRANS, MKL_NONUNIT, n, nrhs, &
                                1.0_wp, ap1, n, bp1, n, &
                                MKL_COMPACT_SSE, nmat)
      call cqr_mkl_trsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, &
                                MKL_TRANS, MKL_NONUNIT, n, nrhs, &
                                1.0_wp, ap1, n, bp1, n, &
                                MKL_COMPACT_SSE, nmat)
      call check(maxval(abs(bp - xp)) < tol, 'potrf + trsm')

      ! Cholesky solve: potrf + potrs, then the fused posv (bit-identical)
      ap = pack_c(a, eye)
      bp = pack_c(b, zed)
      call cqr_mkl_potrf_compact(MKL_COL_MAJOR, MKL_LOWER, n, ap1, n, &
                                 info, MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_potrf_compact info (for potrs)')
      call cqr_mkl_potrs_compact(MKL_COL_MAJOR, MKL_LOWER, n, nrhs, &
                                 ap1, n, bp1, n, info, &
                                 MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_potrs_compact info')
      call check(maxval(abs(bp - xp)) < tol, 'potrf + potrs solution')
      af = ap
      xf = bp
      ap = pack_c(a, eye)
      bp = pack_c(b, zed)
      call cqr_mkl_posv_compact(MKL_COL_MAJOR, MKL_LOWER, n, nrhs, &
                                ap1, n, bp1, n, info, &
                                MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_posv_compact info')
      call check(all(ap == af) .and. all(bp == xf), &
                 'posv == potrf + potrs')

      ! LDL^T: sytrfnp + sytrsnp, then the fused sysvnp (bit-identical).
      ap = pack_c(a, eye)
      bp = pack_c(b, zed)
      call cqr_mkl_sytrfnp_compact(MKL_COL_MAJOR, MKL_LOWER, n, ap1, n, &
                                   info, MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_sytrfnp_compact info')
      call cqr_mkl_sytrsnp_compact(MKL_COL_MAJOR, MKL_LOWER, n, nrhs, &
                                   ap1, n, bp1, n, info, &
                                   MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_sytrsnp_compact info')
      call check(maxval(abs(bp - xp)) < tol, 'sytrfnp + sytrsnp solution')
      af = ap
      xf = bp
      ap = pack_c(a, eye)
      bp = pack_c(b, zed)
      call cqr_mkl_sysvnp_compact(MKL_COL_MAJOR, MKL_LOWER, n, nrhs, &
                                  ap1, n, bp1, n, info, &
                                  MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_sysvnp_compact info')
      call check(all(ap == af) .and. all(bp == xf), &
                 'sysvnp == sytrfnp + sytrsnp')
   end subroutine run_tests

   ! An unrecognized pack format is the one condition info reports (-1);
   ! the buffer is never touched, so a placeholder suffices.
   subroutine test_format_check()
      real(wp) :: ap(1)
      integer(ik) :: info
      call cqr_mkl_potrf_compact(MKL_COL_MAJOR, MKL_LOWER, n, ap, n, &
                                 info, 999, nmat)
      call check(info == -1, 'unrecognized format reports info = -1')
   end subroutine test_format_check

end program test_cqr_fortran_mkl
