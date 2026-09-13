! Fortran-interface test for the portable C API (include/cqr_compact.fi),
! compiled as free-form source; the fixed-form twin of this file is
! test_cqr_fortran_compact_fixed.f, and compiling both is what verifies the
! include file's dual-form layout. Every entry point is called through its
! bind(c) interface on a small batch and each solve is checked against the
! exact x the right-hand sides were built from -- the point is that the
! interface blocks match the C signatures and the link line closes, not the
! numerics (the C++ suites own those).
!
! The test body is precision-generic: it calls through the include file's
! generic names (geqrf_compact, ...), which resolve against the work
! precision wp -- set to c_float by the CQR_SINGLE preprocessor guard,
! default c_double. CMake compiles this source once per precision, so both
! sets of specifics stay covered.
!
! Each compact (interleaved) buffer keeps its natural shape,
! (V, rows, cols, ngroups), and is built with a single RESHAPE: ORDER
! walks the result lane-innermost -- which *is* the compact layout -- and
! PAD's copies complete the partial final group along that same walk,
! identity matrices for an A operand, zero columns for a B operand.
! nmat = 3 at V = 2 leaves one padding lane, so the padded-group
! convention is exercised, not just prepared; its lanes solve identity
! systems with zero right-hand sides, so the computed solutions compare
! directly against pack_c of the exact x, padding included. The buffers
! pass to the generic calls as they are: the portable interfaces declare
! their array dummies in the same compact shape (rank-4 matrices, rank-3
! tau), which is what generic resolution matches.
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

   subroutine check(ok, what)
      logical, intent(in) :: ok
      character(*), intent(in) :: what
      if (.not. ok) then
         print '(2a)', 'FAILED: ', what
         error stop 1
      end if
   end subroutine check

   ! Diagonally dominant symmetric test matrices (usable by every routine:
   ! QR, Cholesky, LDL^T), lane-dependent so no two matrices are equal, and
   ! right-hand sides built from a known x.
   subroutine fill(a, x, b)
      real(wp), intent(out) :: a(n, n, nmat), x(n, nrhs, nmat)
      real(wp), intent(out) :: b(n, nrhs, nmat)
      integer :: i, j, r, im
      do im = 1, nmat
         do j = 1, n
            do i = 1, n
               a(i, j, im) = 1.0_wp / real(i + j - 1, wp)
               if (i == j) a(i, j, im) = a(i, j, im) + real(n + im, wp)
            end do
         end do
         do r = 1, nrhs
            do i = 1, n
               x(i, r, im) = real(i + n * (r - 1), wp) &
                             + 0.25_wp * real(im, wp)
            end do
         end do
         b(:, :, im) = matmul(a(:, :, im), x(:, :, im))
      end do
   end subroutine fill

   ! Dense (i, j, matrix) -> compact (lane, i, j, group), in one reshape:
   ! ORDER interleaves the lanes, PAD completes the final group.
   function pack_c(dense, ncol, pad) result(packed)
      integer, intent(in) :: ncol
      real(wp), intent(in) :: dense(n, ncol, nmat), pad(n, ncol)
      real(wp) :: packed(vw, n, ncol, ng)
      packed = reshape(dense, shape(packed), pad=pad, order=[2, 3, 1, 4])
   end function pack_c

   subroutine run_tests()
      real(wp) :: a(n, n, nmat), x(n, nrhs, nmat), b(n, nrhs, nmat)
      real(wp) :: eye(n, n), zed(n, nrhs)
      real(wp) :: ap(vw, n, n, ng), bp(vw, n, nrhs, ng)
      real(wp) :: taup(vw, n, ng), xp(vw, n, nrhs, ng)
      real(wp) :: af(vw, n, n, ng), xf(vw, n, nrhs, ng)
      integer(c_int) :: info
      integer :: i

      call fill(a, x, b)
      eye = 0.0_wp
      do i = 1, n
         eye(i, i) = 1.0_wp
      end do
      zed = 0.0_wp

      ! The expected compact solution: x in the real lanes, zero in the
      ! padding lanes (identity system, zero right-hand side).
      xp = pack_c(x, nrhs, zed)

      ! QR chain: geqrf -> ormqr (Q^T) -> trsm (R)
      ap = pack_c(a, n, eye)
      bp = pack_c(b, nrhs, zed)
      info = geqrf_compact('C', n, n, ap, n, taup, vw, nmat)
      call check(info == 0, 'geqrf_compact info')
      info = ormqr_compact('T', n, nrhs, n, ap, n, taup, bp, n, vw, nmat)
      call check(info == 0, 'ormqr_compact info')
      info = trsm_compact('C', 'L', 'U', 'N', 'N', n, nrhs, &
                          1.0_wp, ap, n, bp, n, vw, nmat)
      call check(info == 0, 'trsm_compact info')
      call check(maxval(abs(bp - xp)) < tol, 'QR chain solution')

      ! gels: the same square solve in one call
      ap = pack_c(a, n, eye)
      bp = pack_c(b, nrhs, zed)
      info = gels_compact('C', 'N', n, n, nrhs, ap, n, bp, n, taup, &
                          vw, nmat)
      call check(info == 0, 'gels_compact info')
      call check(maxval(abs(bp - xp)) < tol, 'gels solution')

      ! Cholesky: potrf, then two triangular solves close A x = b
      ap = pack_c(a, n, eye)
      bp = pack_c(b, nrhs, zed)
      info = potrf_compact('C', 'L', n, ap, n, vw, nmat)
      call check(info == 0, 'potrf_compact info')
      info = trsm_compact('C', 'L', 'L', 'N', 'N', n, nrhs, &
                          1.0_wp, ap, n, bp, n, vw, nmat)
      call check(info == 0, 'trsm_compact (L) info')
      info = trsm_compact('C', 'L', 'L', 'T', 'N', n, nrhs, &
                          1.0_wp, ap, n, bp, n, vw, nmat)
      call check(info == 0, 'trsm_compact (L^T) info')
      call check(maxval(abs(bp - xp)) < tol, 'potrf + trsm')

      ! LDL^T: sytrfnp + sytrsnp, then the fused sysvnp (bit-identical)
      ap = pack_c(a, n, eye)
      bp = pack_c(b, nrhs, zed)
      info = sytrfnp_compact('C', 'L', n, ap, n, vw, nmat)
      call check(info == 0, 'sytrfnp_compact info')
      info = sytrsnp_compact('C', 'L', n, nrhs, ap, n, bp, n, vw, nmat)
      call check(info == 0, 'sytrsnp_compact info')
      call check(maxval(abs(bp - xp)) < tol, 'sytrfnp + sytrsnp solution')
      af = ap
      xf = bp
      ap = pack_c(a, n, eye)
      bp = pack_c(b, nrhs, zed)
      info = sysvnp_compact('C', 'L', n, nrhs, ap, n, bp, n, vw, nmat)
      call check(info == 0, 'sysvnp_compact info')
      call check(all(ap == af) .and. all(bp == xf), &
                 'sysvnp_compact == sytrfnp + sytrsnp')
   end subroutine run_tests

   ! The portable API's LAPACK-style validation: -j for a bad j-th argument.
   subroutine test_validation()
      real(wp) :: da(1, 1, 1, 1), dt(1, 1, 1)
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
