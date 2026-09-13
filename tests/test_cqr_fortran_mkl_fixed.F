! Fixed-form includer of cqr_mkl_ext.fi: compiling this file is what
! proves the interface file reads as fixed-form source too (its twin,
! test_cqr_fortran_mkl.F90, includes the same file as free form and owns
! the real numerics). Cholesky of identity matrices leaves the factor
! the identity to rounding; an unrecognized pack format reports
! info = -1. The buffer keeps its compact shape (lane, row, column) and
! passes to the specific name as it is: a call to a specific accepts
! any rank by sequence association.
! Compiled with CQR_ILP64 defined (an ilp64 library build) the same
! test runs through cqr_mkl_ext_ilp64.fi, with integer kind ik for
! every MKL_INT actual argument.
!
! Assisted-by: Claude:claude-fable-5

      program tmklfix
      use, intrinsic :: iso_c_binding, only: c_int, c_long_long,
     &    c_double
      implicit none
#ifdef CQR_ILP64
      include 'cqr_mkl_ext_ilp64.fi'
      integer, parameter :: ik = c_long_long
#else
      include 'cqr_mkl_ext.fi'
      integer, parameter :: ik = c_int
#endif
      integer(ik), parameter :: vw = 2, nmat = 2, n = 2
      real(c_double) :: ap(vw, n, n)
      real(c_double), parameter :: tol = 10*epsilon(1.0_c_double)
      integer(ik) :: info
      integer i, im

! One full MKL_COMPACT_SSE group of vw doubles; lane im holds the
! n x n identity.
      ap = 0.0_c_double
      do im = 1, nmat
         do i = 1, n
            ap(im, i, i) = 1.0_c_double
         end do
      end do

      call cqr_mkl_dpotrf_compact(MKL_COL_MAJOR, MKL_LOWER, n, ap, n,
     &    info, MKL_COMPACT_SSE, nmat)
      if (info .ne. 0) stop 1
      do im = 1, nmat
         do i = 1, n
            if (abs(ap(im, i, i) - 1.0_c_double) .gt. tol) stop 2
         end do
      end do

      call cqr_mkl_dpotrf_compact(MKL_COL_MAJOR, MKL_LOWER, n, ap, n,
     &    info, 999, nmat)
      if (info .ne. -1) stop 3

      print *, 'test_cqr_fortran_mkl_fixed: all checks passed'
      end program tmklfix
