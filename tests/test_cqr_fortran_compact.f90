! Fortran-interface test for the portable C API (include/cqr_compact.fi),
! compiled as free-form source; the fixed-form twin of this file is
! test_cqr_fortran_compact_fixed.f, and compiling both is what verifies the
! include file's dual-form layout. Every entry point is called through its
! bind(c) interface on a small batch and each solve is checked against the
! exact x the right-hand sides were built from -- the point is that the
! interface blocks match the C signatures and the link line closes, not the
! numerics (the C++ suites own those).
!
! The compact (interleaved) buffers are built with RESHAPE alone: the first
! reshape splits the batch index into (lane, group) and its PAD argument
! fills the partial final group -- identity matrices for an A operand,
! zero columns for a B operand -- and the second reshape's ORDER permutes
! the lane index innermost, which *is* the compact layout. nmat = 3 at
! V = 2 leaves one padding lane, so the padded-group convention is
! exercised, not just prepared.
!
! Assisted-by: Claude:claude-fable-5

program test_cqr_fortran_compact
   use, intrinsic :: iso_c_binding, only: c_int, c_double, c_float
   implicit none
   include 'cqr_compact.fi'

   integer(c_int), parameter :: vw = 2, nmat = 3, n = 3, nrhs = 2
   integer(c_int), parameter :: ng = (nmat + vw - 1) / vw ! compact groups

   call test_double()
   call test_single()
   call test_validation()
   print '(a)', 'test_cqr_fortran_compact: all checks passed'

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
   subroutine fill_d(a, x, b)
      real(c_double), intent(out) :: a(n, n, nmat), x(n, nrhs, nmat)
      real(c_double), intent(out) :: b(n, nrhs, nmat)
      integer :: i, j, r, im
      do im = 1, nmat
         do j = 1, n
            do i = 1, n
               a(i, j, im) = 1.0_c_double / real(i + j - 1, c_double)
               if (i == j) a(i, j, im) = a(i, j, im) + real(n + im, c_double)
            end do
         end do
         do r = 1, nrhs
            do i = 1, n
               x(i, r, im) = real(i + n * (r - 1), c_double) &
                             + 0.25_c_double * real(im, c_double)
            end do
         end do
         b(:, :, im) = matmul(a(:, :, im), x(:, :, im))
      end do
   end subroutine fill_d

   ! Dense (i, j, matrix) -> compact (lane, i, j, group): pad the batch to
   ! full groups, then permute the lane index innermost.
   function pack_d(dense, ncol, pad) result(packed)
      integer, intent(in) :: ncol
      real(c_double), intent(in) :: dense(n, ncol, nmat), pad(n, ncol)
      real(c_double) :: packed(vw, n, ncol, ng)
      packed = reshape(reshape(dense, [n, ncol, vw, ng], pad=pad), &
                       [vw, n, ncol, ng], order=[2, 3, 1, 4])
   end function pack_d

   ! Compact solution block back to dense, padding lanes dropped.
   function unpack_d(packed) result(dense)
      real(c_double), intent(in) :: packed(vw, n, nrhs, ng)
      real(c_double) :: full(n, nrhs, vw*ng), dense(n, nrhs, nmat)
      full = reshape(reshape(packed, [n, nrhs, vw, ng], &
                             order=[3, 1, 2, 4]), shape(full))
      dense = full(:, :, 1:nmat)
   end function unpack_d

   subroutine test_double()
      real(c_double) :: a(n, n, nmat), x(n, nrhs, nmat), b(n, nrhs, nmat)
      real(c_double) :: eye(n, n), zed(n, nrhs)
      real(c_double) :: ap(vw, n, n, ng), bp(vw, n, nrhs, ng)
      real(c_double) :: taup(vw, n, ng)
      real(c_double) :: af(vw, n, n, ng), xf(vw, n, nrhs, ng)
      real(c_double), parameter :: tol = 1.0e-11_c_double
      integer(c_int) :: info
      integer :: i

      call fill_d(a, x, b)
      eye = 0.0_c_double
      do i = 1, n
         eye(i, i) = 1.0_c_double
      end do
      zed = 0.0_c_double

      ! QR chain: geqrf -> ormqr (Q^T) -> trsm (R)
      ap = pack_d(a, n, eye)
      bp = pack_d(b, nrhs, zed)
      info = dgeqrf_compact('C', n, n, ap, n, taup, vw, nmat)
      call check(info == 0, 'dgeqrf_compact info')
      info = dormqr_compact('T', n, nrhs, n, ap, n, taup, bp, n, vw, nmat)
      call check(info == 0, 'dormqr_compact info')
      info = dtrsm_compact('C', 'L', 'U', 'N', 'N', n, nrhs, &
                           1.0_c_double, ap, n, bp, n, vw, nmat)
      call check(info == 0, 'dtrsm_compact info')
      call check(maxval(abs(unpack_d(bp) - x)) < tol, 'd QR chain solution')

      ! gels: the same square solve in one call
      ap = pack_d(a, n, eye)
      bp = pack_d(b, nrhs, zed)
      info = dgels_compact('C', 'N', n, n, nrhs, ap, n, bp, n, taup, &
                           vw, nmat)
      call check(info == 0, 'dgels_compact info')
      call check(maxval(abs(unpack_d(bp) - x)) < tol, 'dgels solution')

      ! Cholesky: potrf, then two triangular solves close A x = b
      ap = pack_d(a, n, eye)
      bp = pack_d(b, nrhs, zed)
      info = dpotrf_compact('C', 'L', n, ap, n, vw, nmat)
      call check(info == 0, 'dpotrf_compact info')
      info = dtrsm_compact('C', 'L', 'L', 'N', 'N', n, nrhs, &
                           1.0_c_double, ap, n, bp, n, vw, nmat)
      call check(info == 0, 'dtrsm_compact (L) info')
      info = dtrsm_compact('C', 'L', 'L', 'T', 'N', n, nrhs, &
                           1.0_c_double, ap, n, bp, n, vw, nmat)
      call check(info == 0, 'dtrsm_compact (L^T) info')
      call check(maxval(abs(unpack_d(bp) - x)) < tol, 'dpotrf + dtrsm')

      ! LDL^T: sytrfnp + sytrsnp, then the fused sysvnp (bit-identical)
      ap = pack_d(a, n, eye)
      bp = pack_d(b, nrhs, zed)
      info = dsytrfnp_compact('C', 'L', n, ap, n, vw, nmat)
      call check(info == 0, 'dsytrfnp_compact info')
      info = dsytrsnp_compact('C', 'L', n, nrhs, ap, n, bp, n, vw, nmat)
      call check(info == 0, 'dsytrsnp_compact info')
      call check(maxval(abs(unpack_d(bp) - x)) < tol, &
                 'dsytrfnp + dsytrsnp solution')
      af = ap
      xf = bp
      ap = pack_d(a, n, eye)
      bp = pack_d(b, nrhs, zed)
      info = dsysvnp_compact('C', 'L', n, nrhs, ap, n, bp, n, vw, nmat)
      call check(info == 0, 'dsysvnp_compact info')
      call check(all(ap == af) .and. all(bp == xf), &
                 'dsysvnp_compact == dsytrfnp + dsytrsnp')
   end subroutine test_double

   ! -- single precision: the same batch and chains at float tolerance --

   subroutine fill_s(a, x, b)
      real(c_float), intent(out) :: a(n, n, nmat), x(n, nrhs, nmat)
      real(c_float), intent(out) :: b(n, nrhs, nmat)
      integer :: i, j, r, im
      do im = 1, nmat
         do j = 1, n
            do i = 1, n
               a(i, j, im) = 1.0_c_float / real(i + j - 1, c_float)
               if (i == j) a(i, j, im) = a(i, j, im) + real(n + im, c_float)
            end do
         end do
         do r = 1, nrhs
            do i = 1, n
               x(i, r, im) = real(i + n * (r - 1), c_float) &
                             + 0.25_c_float * real(im, c_float)
            end do
         end do
         b(:, :, im) = matmul(a(:, :, im), x(:, :, im))
      end do
   end subroutine fill_s

   function pack_s(dense, ncol, pad) result(packed)
      integer, intent(in) :: ncol
      real(c_float), intent(in) :: dense(n, ncol, nmat), pad(n, ncol)
      real(c_float) :: packed(vw, n, ncol, ng)
      packed = reshape(reshape(dense, [n, ncol, vw, ng], pad=pad), &
                       [vw, n, ncol, ng], order=[2, 3, 1, 4])
   end function pack_s

   function unpack_s(packed) result(dense)
      real(c_float), intent(in) :: packed(vw, n, nrhs, ng)
      real(c_float) :: full(n, nrhs, vw*ng), dense(n, nrhs, nmat)
      full = reshape(reshape(packed, [n, nrhs, vw, ng], &
                             order=[3, 1, 2, 4]), shape(full))
      dense = full(:, :, 1:nmat)
   end function unpack_s

   subroutine test_single()
      real(c_float) :: a(n, n, nmat), x(n, nrhs, nmat), b(n, nrhs, nmat)
      real(c_float) :: eye(n, n), zed(n, nrhs)
      real(c_float) :: ap(vw, n, n, ng), bp(vw, n, nrhs, ng)
      real(c_float) :: taup(vw, n, ng)
      real(c_float) :: af(vw, n, n, ng), xf(vw, n, nrhs, ng)
      real(c_float), parameter :: tol = 1.0e-3_c_float
      integer(c_int) :: info
      integer :: i

      call fill_s(a, x, b)
      eye = 0.0_c_float
      do i = 1, n
         eye(i, i) = 1.0_c_float
      end do
      zed = 0.0_c_float

      ap = pack_s(a, n, eye)
      bp = pack_s(b, nrhs, zed)
      info = sgeqrf_compact('C', n, n, ap, n, taup, vw, nmat)
      call check(info == 0, 'sgeqrf_compact info')
      info = sormqr_compact('T', n, nrhs, n, ap, n, taup, bp, n, vw, nmat)
      call check(info == 0, 'sormqr_compact info')
      info = strsm_compact('C', 'L', 'U', 'N', 'N', n, nrhs, &
                           1.0_c_float, ap, n, bp, n, vw, nmat)
      call check(info == 0, 'strsm_compact info')
      call check(maxval(abs(unpack_s(bp) - x)) < tol, 's QR chain solution')

      ap = pack_s(a, n, eye)
      bp = pack_s(b, nrhs, zed)
      info = sgels_compact('C', 'N', n, n, nrhs, ap, n, bp, n, taup, &
                           vw, nmat)
      call check(info == 0, 'sgels_compact info')
      call check(maxval(abs(unpack_s(bp) - x)) < tol, 'sgels solution')

      ap = pack_s(a, n, eye)
      bp = pack_s(b, nrhs, zed)
      info = spotrf_compact('C', 'L', n, ap, n, vw, nmat)
      call check(info == 0, 'spotrf_compact info')
      info = strsm_compact('C', 'L', 'L', 'N', 'N', n, nrhs, &
                           1.0_c_float, ap, n, bp, n, vw, nmat)
      call check(info == 0, 'strsm_compact (L) info')
      info = strsm_compact('C', 'L', 'L', 'T', 'N', n, nrhs, &
                           1.0_c_float, ap, n, bp, n, vw, nmat)
      call check(info == 0, 'strsm_compact (L^T) info')
      call check(maxval(abs(unpack_s(bp) - x)) < tol, 'spotrf + strsm')

      ap = pack_s(a, n, eye)
      bp = pack_s(b, nrhs, zed)
      info = ssytrfnp_compact('C', 'L', n, ap, n, vw, nmat)
      call check(info == 0, 'ssytrfnp_compact info')
      info = ssytrsnp_compact('C', 'L', n, nrhs, ap, n, bp, n, vw, nmat)
      call check(info == 0, 'ssytrsnp_compact info')
      call check(maxval(abs(unpack_s(bp) - x)) < tol, &
                 'ssytrfnp + ssytrsnp solution')
      af = ap
      xf = bp
      ap = pack_s(a, n, eye)
      bp = pack_s(b, nrhs, zed)
      info = ssysvnp_compact('C', 'L', n, nrhs, ap, n, bp, n, vw, nmat)
      call check(info == 0, 'ssysvnp_compact info')
      call check(all(ap == af) .and. all(bp == xf), &
                 'ssysvnp_compact == ssytrfnp + ssytrsnp')
   end subroutine test_single

   ! The portable API's LAPACK-style validation: -j for a bad j-th argument.
   subroutine test_validation()
      real(c_double) :: dd(1), dt(1)
      real(c_float) :: sd(1), st(1)
      integer(c_int) :: info
      info = dgeqrf_compact('C', -1, 1, dd, 1, dt, vw, 0)
      call check(info == -2, 'dgeqrf_compact rejects m < 0')
      info = dpotrf_compact('C', 'X', 1, dd, 1, vw, 0)
      call check(info == -2, 'dpotrf_compact rejects bad uplo')
      info = strsm_compact('C', 'L', 'U', 'N', 'N', 1, 1, 1.0_c_float, &
                           sd, 1, sd, 1, 3, 0)
      call check(info == -13, 'strsm_compact rejects V = 3')
      info = sgels_compact('C', 'N', 1, 1, 1, sd, 1, sd, 1, st, vw, -1)
      call check(info == -12, 'sgels_compact rejects nm < 0')
   end subroutine test_validation

end program test_cqr_fortran_compact
