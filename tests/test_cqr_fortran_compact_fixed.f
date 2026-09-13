! Fixed-form includer of cqr_compact.fi: compiling this file is what
! proves the interface file reads as fixed-form source too (its twin,
! test_cqr_fortran_compact.f90, includes the same file as free form and
! owns the real numerics). A QR of identity matrices leaves R = I and
! tau = 0 exactly; one argument-validation call closes the check.
!
! Assisted-by: Claude:claude-fable-5

      program tcqrfix
      use, intrinsic :: iso_c_binding, only: c_int, c_double
      implicit none
      include 'cqr_compact.fi'
      integer(c_int), parameter :: vw = 2, nmat = 2, n = 2
      real(c_double) :: ap(n*n*nmat), taup(n*nmat)
      real(c_double), parameter :: tol = 1.0e-14_c_double
      integer(c_int) :: info
      integer i, im

! One full group of vw lanes; lane im holds the n x n identity.
      ap = 0.0_c_double
      do im = 1, nmat
         do i = 1, n
            ap(((i-1)*n + (i-1))*vw + im) = 1.0_c_double
         end do
      end do

      info = dgeqrf_compact('C', n, n, ap, n, taup, vw, nmat)
      if (info .ne. 0) stop 1
      do i = 1, n*nmat
         if (abs(taup(i)) .gt. tol) stop 2
      end do
      do im = 1, nmat
         do i = 1, n
            if (abs(ap(((i-1)*n + (i-1))*vw + im) - 1.0_c_double)
     &          .gt. tol) stop 3
         end do
      end do

      info = dgeqrf_compact('C', -1, n, ap, n, taup, vw, nmat)
      if (info .ne. -2) stop 4

      print *, 'test_cqr_fortran_compact_fixed: all checks passed'
      end program tcqrfix
