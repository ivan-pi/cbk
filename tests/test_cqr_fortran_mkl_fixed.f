! Fixed-form includer of cqr_mkl_ext.fi: compiling this file is what
! proves the interface file reads as fixed-form source too (its twin,
! test_cqr_fortran_mkl.f90, includes the same file as free form and owns
! the real numerics). Cholesky of identity matrices leaves the factor
! exactly the identity; an unrecognized pack format reports info = -1.
!
! Assisted-by: Claude:claude-fable-5

      program tmklfix
      use, intrinsic :: iso_c_binding, only: c_int, c_double
      implicit none
      include 'cqr_mkl_ext.fi'
      integer(c_int), parameter :: vw = 2, nmat = 2, n = 2
      real(c_double) :: ap(n*n*nmat)
      real(c_double), parameter :: tol = 1.0e-14_c_double
      integer(c_int) :: info
      integer i, im

! One full MKL_COMPACT_SSE group of vw doubles; lane im holds the
! n x n identity.
      ap = 0.0_c_double
      do im = 1, nmat
         do i = 1, n
            ap(((i-1)*n + (i-1))*vw + im) = 1.0_c_double
         end do
      end do

      call cqr_mkl_dpotrf_compact(MKL_COL_MAJOR, MKL_LOWER, n, ap, n,
     &    info, MKL_COMPACT_SSE, nmat)
      if (info .ne. 0) stop 1
      do im = 1, nmat
         do i = 1, n
            if (abs(ap(((i-1)*n + (i-1))*vw + im) - 1.0_c_double)
     &          .gt. tol) stop 2
         end do
      end do

      call cqr_mkl_dpotrf_compact(MKL_COL_MAJOR, MKL_LOWER, n, ap, n,
     &    info, 999, nmat)
      if (info .ne. -1) stop 3

      print *, 'test_cqr_fortran_mkl_fixed: all checks passed'
      end program tmklfix
