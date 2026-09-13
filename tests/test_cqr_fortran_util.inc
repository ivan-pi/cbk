! test_cqr_fortran_util.inc -- scaffolding shared by the two free-form
! Fortran-interface tests, INCLUDEd after their CONTAINS: everything
! host-associates, so the work precision wp and the batch parameters
! (n, nrhs, nmat, vw, ng) come from the including program and nothing
! here needs arguments for them. Free-form only; the fixed-form tests
! stay deliberately self-contained.
!
! Assisted-by: Claude:claude-fable-5

   subroutine check(ok, what)
      logical, intent(in) :: ok
      character(*), intent(in) :: what
      if (.not. ok) then
         print '(2a)', 'FAILED: ', what
         error stop 1
      end if
   end subroutine check

   ! Diagonally dominant symmetric test matrices (usable by every routine:
   ! QR, Cholesky, LDL^T), lane-dependent so no two matrices are equal,
   ! right-hand sides built from a known x, and the two pack paddings:
   ! identity (eye) for an A operand, zero columns (zed) for a B operand.
   subroutine fill(a, x, b, eye, zed)
      real(wp), intent(out) :: a(n, n, nmat), x(n, nrhs, nmat)
      real(wp), intent(out) :: b(n, nrhs, nmat)
      real(wp), intent(out) :: eye(n, n), zed(n, nrhs)
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
      eye = 0.0_wp
      do i = 1, n
         eye(i, i) = 1.0_wp
      end do
      zed = 0.0_wp
   end subroutine fill

   ! Dense (i, j, matrix) -> compact (lane, i, j, group), in one reshape:
   ! ORDER interleaves the lanes, PAD completes the final group.
   function pack_c(dense, pad) result(packed)
      real(wp), intent(in) :: dense(:, :, :), pad(:, :)
      real(wp) :: packed(vw, n, size(dense, 2), ng)
      packed = reshape(dense, shape(packed), pad=pad, order=[2, 3, 1, 4])
   end function pack_c
