! Fixed-form includer of cqr_compact.fi: compiling this file is what
! proves the interface file reads as fixed-form source too (its twin,
! test_cqr_fortran_compact.F90, includes the same file as free form and
! owns the real numerics). A QR of identity matrices leaves R = I and
! tau = 0 to rounding; one argument-validation call closes the check.
! The buffers keep their compact shape (lane, row, column) and pass to
! the specific name as they are: a call to a specific accepts any rank
! by sequence association.
!
! Assisted-by: Claude:claude-fable-5

      program tcqrfix
      use, intrinsic :: iso_c_binding, only: c_int, c_double
      implicit none
      include 'cqr_compact.fi'
      integer(c_int), parameter :: vw = 2, nmat = 2, n = 2
      real(c_double) :: ap(vw, n, n), taup(vw, n)
      real(c_double), parameter :: tol = 10*epsilon(1.0_c_double)
      integer(c_int) :: info
      integer i, im

! One full group of vw lanes; lane im holds the n x n identity.
      ap = 0.0_c_double
      do im = 1, nmat
         do i = 1, n
            ap(im, i, i) = 1.0_c_double
         end do
      end do

      info = dgeqrf_compact('C', n, n, ap, n, taup, vw, nmat)
      if (info .ne. 0) stop 1
      do im = 1, nmat
         do i = 1, n
            if (abs(taup(im, i)) .gt. tol) stop 2
            if (abs(ap(im, i, i) - 1.0_c_double) .gt. tol) stop 3
         end do
      end do

      info = dgeqrf_compact('C', -1, n, ap, n, taup, vw, nmat)
      if (info .ne. -2) stop 4

      print *, 'test_cqr_fortran_compact_fixed: all checks passed'
      end program tcqrfix
