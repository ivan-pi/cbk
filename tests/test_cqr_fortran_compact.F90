! Fortran-interface test for the portable C API (include/cqr_compact.fi),
! compiled as free-form source; the fixed-form twin of this file is
! test_cqr_fortran_compact_fixed.f, and compiling both is what verifies the
! include file's dual-form layout. Every entry point is called through its
! bind(c) generic name on a small batch and each solve is checked against
! the exact x the right-hand sides were built from -- the point is that
! the interface blocks match the C signatures and the link line closes,
! not the numerics (the C++ suites own those). CMake compiles this source
! once per work precision wp (CQR_SINGLE selects c_float), and nmat = 3 at
! V = 2 leaves one padding lane, so the padded-group convention is
! exercised, not just prepared. The scaffolding (check, fill, pack_c) is
! shared with the MKL-style test: test_cqr_fortran_util.inc.
!
! Assisted-by: Claude:claude-fable-5

program test_cqr_fortran_compact
   use, intrinsic :: iso_c_binding, only: c_int, c_double, c_float
   implicit none
   include 'cqr_compact.fi'

#ifdef CQR_SINGLE
   integer, parameter :: wp = c_float ! work precision of this build
#else
   integer, parameter :: wp = c_double
#endif

   integer(c_int), parameter :: vw = 2, nmat = 3, n = 3, nrhs = 2
   integer(c_int), parameter :: ng = (nmat + vw - 1) / vw ! compact groups
   real(wp), parameter :: tol = 1.0e4_wp * epsilon(1.0_wp)

   call run_tests()
   call test_validation()
   print '(a,i0,a)', 'test_cqr_fortran_compact: all checks passed (', &
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
      integer(c_int) :: info

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

      ! QR chain: geqrf -> ormqr (Q^T) -> trsm (R)
      ap = pack_c(a, eye)
      bp = pack_c(b, zed)
      info = geqrf_compact('C', n, n, ap1, n, taup1, vw, nmat)
      call check(info == 0, 'geqrf_compact info')
      info = ormqr_compact('T', n, nrhs, n, ap1, n, taup1, bp1, n, vw, nmat)
      call check(info == 0, 'ormqr_compact info')
      info = trsm_compact('C', 'L', 'U', 'N', 'N', n, nrhs, &
                          1.0_wp, ap1, n, bp1, n, vw, nmat)
      call check(info == 0, 'trsm_compact info')
      call check(maxval(abs(bp - xp)) < tol, 'QR chain solution')

      ! gels: the same square solve in one call
      ap = pack_c(a, eye)
      bp = pack_c(b, zed)
      info = gels_compact('C', 'N', n, n, nrhs, ap1, n, bp1, n, taup1, &
                          vw, nmat)
      call check(info == 0, 'gels_compact info')
      call check(maxval(abs(bp - xp)) < tol, 'gels solution')

      ! orgqr: the explicit Q must equal Q applied to the identity
      ap = pack_c(a, eye)
      info = geqrf_compact('C', n, n, ap1, n, taup1, vw, nmat)
      call check(info == 0, 'geqrf_compact info (for orgqr)')
      aq = ap
      info = orgqr_compact('C', n, n, n, aq1, n, taup1, vw, nmat)
      call check(info == 0, 'orgqr_compact info')
      bq = pack_c(spread(eye, 3, int(nmat)), eye)
      info = ormqr_compact('N', n, n, n, ap1, n, taup1, bq1, n, vw, nmat)
      call check(info == 0, 'ormqr_compact (N) info')
      call check(maxval(abs(aq - bq)) < tol, 'orgqr == ormqr on identity')

      ! Cholesky: potrf, then two triangular solves close A x = b
      ap = pack_c(a, eye)
      bp = pack_c(b, zed)
      info = potrf_compact('C', 'L', n, ap1, n, vw, nmat)
      call check(info == 0, 'potrf_compact info')
      info = trsm_compact('C', 'L', 'L', 'N', 'N', n, nrhs, &
                          1.0_wp, ap1, n, bp1, n, vw, nmat)
      call check(info == 0, 'trsm_compact (L) info')
      info = trsm_compact('C', 'L', 'L', 'T', 'N', n, nrhs, &
                          1.0_wp, ap1, n, bp1, n, vw, nmat)
      call check(info == 0, 'trsm_compact (L^T) info')
      call check(maxval(abs(bp - xp)) < tol, 'potrf + trsm')

      ! Cholesky solve: potrf + potrs, then the fused posv (bit-identical)
      ap = pack_c(a, eye)
      bp = pack_c(b, zed)
      info = potrf_compact('C', 'L', n, ap1, n, vw, nmat)
      call check(info == 0, 'potrf_compact info (for potrs)')
      info = potrs_compact('C', 'L', n, nrhs, ap1, n, bp1, n, vw, nmat)
      call check(info == 0, 'potrs_compact info')
      call check(maxval(abs(bp - xp)) < tol, 'potrf + potrs solution')
      af = ap
      xf = bp
      ap = pack_c(a, eye)
      bp = pack_c(b, zed)
      info = posv_compact('C', 'L', n, nrhs, ap1, n, bp1, n, vw, nmat)
      call check(info == 0, 'posv_compact info')
      call check(all(ap == af) .and. all(bp == xf), &
                 'posv_compact == potrf + potrs')

      ! LDL^T: sytrfnp + sytrsnp, then the fused sysvnp (bit-identical)
      ap = pack_c(a, eye)
      bp = pack_c(b, zed)
      info = sytrfnp_compact('C', 'L', n, ap1, n, vw, nmat)
      call check(info == 0, 'sytrfnp_compact info')
      info = sytrsnp_compact('C', 'L', n, nrhs, ap1, n, bp1, n, vw, nmat)
      call check(info == 0, 'sytrsnp_compact info')
      call check(maxval(abs(bp - xp)) < tol, 'sytrfnp + sytrsnp solution')
      af = ap
      xf = bp
      ap = pack_c(a, eye)
      bp = pack_c(b, zed)
      info = sysvnp_compact('C', 'L', n, nrhs, ap1, n, bp1, n, vw, nmat)
      call check(info == 0, 'sysvnp_compact info')
      call check(all(ap == af) .and. all(bp == xf), &
                 'sysvnp_compact == sytrfnp + sytrsnp')
   end subroutine run_tests

   ! The portable API's LAPACK-style validation: -j for a bad j-th argument.
   subroutine test_validation()
      real(wp) :: da(1), dt(1)
      integer(c_int) :: info
      info = geqrf_compact('C', -1, 1, da, 1, dt, vw, 0)
      call check(info == -2, 'geqrf_compact rejects m < 0')
      info = potrf_compact('C', 'X', 1, da, 1, vw, 0)
      call check(info == -2, 'potrf_compact rejects bad uplo')
      info = trsm_compact('C', 'L', 'U', 'N', 'N', 1, 1, 1.0_wp, &
                          da, 1, da, 1, 3, 0)
      call check(info == -13, 'trsm_compact rejects V = 3')
      info = gels_compact('C', 'N', 1, 1, 1, da, 1, da, 1, dt, vw, -1)
      call check(info == -12, 'gels_compact rejects nm < 0')
   end subroutine test_validation

end program test_cqr_fortran_compact
