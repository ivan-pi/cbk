! Fortran-interface test for the MKL-style API (include/cqr_mkl_ext.fi),
! compiled as free-form source; test_cqr_fortran_mkl_fixed.F includes the
! same file as fixed form. Every entry point is called through its bind(c)
! interface -- the enumerators transcribed from mkl_types.h select layout,
! uplo, side, trans, diag and the pack format -- and each solve is checked
! against the exact x the right-hand sides were built from. The compact
! buffers are built with RESHAPE (see test_cqr_fortran_compact.F90: PAD
! fills the partial final group, ORDER permutes the lane index innermost),
! so nothing here calls or links MKL itself; the point is that the
! interface blocks match the C signatures and the link line closes.
!
! The test body is precision-generic: it calls through the include file's
! generic names (cqr_mkl_geqrf_compact, ...), which resolve against the
! work precision wp -- set to c_float by the CQR_SINGLE preprocessor
! guard, default c_double. CMake compiles this source once per precision.
! The interleave width follows: MKL_COMPACT_SSE packs 128 bits, V = 2
! doubles or 4 floats, so nmat = 3 leaves a padding lane either way. The
! compact buffers are rank-1: generic resolution matches rank against the
! assumed-size dummies.
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

   subroutine check(ok, what)
      logical, intent(in) :: ok
      character(*), intent(in) :: what
      if (.not. ok) then
         print '(2a)', 'FAILED: ', what
         error stop 1
      end if
   end subroutine check

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

   function pack_c(dense, ncol, pad) result(packed)
      integer(ik), intent(in) :: ncol
      real(wp), intent(in) :: dense(n, ncol, nmat), pad(n, ncol)
      real(wp) :: packed(vw*n*ncol*ng)
      packed = reshape(reshape(reshape(dense, [n, ncol, vw, ng], &
                                       pad=pad), &
                               [vw, n, ncol, ng], order=[2, 3, 1, 4]), &
                       shape(packed))
   end function pack_c

   function unpack_c(packed) result(dense)
      real(wp), intent(in) :: packed(vw*n*nrhs*ng)
      real(wp) :: full(n, nrhs, vw*ng), dense(n, nrhs, nmat)
      full = reshape(reshape(packed, [n, nrhs, vw, ng], &
                             order=[3, 1, 2, 4]), shape(full))
      dense = full(:, :, 1:nmat)
   end function unpack_c

   subroutine run_tests()
      real(wp) :: a(n, n, nmat), x(n, nrhs, nmat), b(n, nrhs, nmat)
      real(wp) :: eye(n, n), zed(n, nrhs)
      real(wp) :: ap(vw*n*n*ng), bp(vw*n*nrhs*ng)
      real(wp) :: taup(vw*n*ng), work(vw*n*ng)
      real(wp) :: af(vw*n*n*ng), xf(vw*n*nrhs*ng)
      integer(ik) :: info
      integer :: i

      call fill(a, x, b)
      eye = 0.0_wp
      do i = 1, n
         eye(i, i) = 1.0_wp
      end do
      zed = 0.0_wp

      ! QR chain, with the lwork = -1 workspace queries first: these
      ! kernels need no scratch, so each query reports 1.
      ap = pack_c(a, n, eye)
      bp = pack_c(b, nrhs, zed)
      call cqr_mkl_geqrf_compact(MKL_COL_MAJOR, n, n, ap, n, taup, &
                                 work, -1_ik, info, MKL_COMPACT_SSE, nmat)
      call check(info == 0 .and. nint(work(1)) == 1, 'geqrf lwork query')
      call cqr_mkl_geqrf_compact(MKL_COL_MAJOR, n, n, ap, n, taup, &
                                 work, 1_ik, info, MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_geqrf_compact info')
      call cqr_mkl_ormqr_compact(MKL_COL_MAJOR, 'L', 'T', n, nrhs, n, &
                                 ap, n, taup, bp, n, work, -1_ik, info, &
                                 MKL_COMPACT_SSE, nmat)
      call check(info == 0 .and. nint(work(1)) == 1, 'ormqr lwork query')
      call cqr_mkl_ormqr_compact(MKL_COL_MAJOR, 'L', 'T', n, nrhs, n, &
                                 ap, n, taup, bp, n, work, 1_ik, info, &
                                 MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_ormqr_compact info')
      call cqr_mkl_trsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, &
                                MKL_NOTRANS, MKL_NONUNIT, n, nrhs, &
                                1.0_wp, ap, n, bp, n, &
                                MKL_COMPACT_SSE, nmat)
      call check(maxval(abs(unpack_c(bp) - x)) < tol, 'QR chain solution')

      ! gels: its work is the tau scratch, one slot per group -- the query
      ! reports min(m,n) * V * ceil(nm/V) slots, a compact tau buffer.
      ap = pack_c(a, n, eye)
      bp = pack_c(b, nrhs, zed)
      call cqr_mkl_gels_compact(MKL_COL_MAJOR, 'N', n, n, nrhs, ap, n, &
                                bp, n, work, -1_ik, info, &
                                MKL_COMPACT_SSE, nmat)
      call check(info == 0 .and. nint(work(1)) == n * vw * ng, &
                 'gels lwork query')
      call cqr_mkl_gels_compact(MKL_COL_MAJOR, 'N', n, n, nrhs, ap, n, &
                                bp, n, work, n * vw * ng, info, &
                                MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_gels_compact info')
      call check(maxval(abs(unpack_c(bp) - x)) < tol, 'gels solution')

      ! Cholesky: potrf, then two triangular solves close A x = b.
      ap = pack_c(a, n, eye)
      bp = pack_c(b, nrhs, zed)
      call cqr_mkl_potrf_compact(MKL_COL_MAJOR, MKL_LOWER, n, ap, n, &
                                 info, MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_potrf_compact info')
      call cqr_mkl_trsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, &
                                MKL_NOTRANS, MKL_NONUNIT, n, nrhs, &
                                1.0_wp, ap, n, bp, n, &
                                MKL_COMPACT_SSE, nmat)
      call cqr_mkl_trsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, &
                                MKL_TRANS, MKL_NONUNIT, n, nrhs, &
                                1.0_wp, ap, n, bp, n, &
                                MKL_COMPACT_SSE, nmat)
      call check(maxval(abs(unpack_c(bp) - x)) < tol, 'potrf + trsm')

      ! LDL^T: sytrfnp + sytrsnp, then the fused sysvnp (bit-identical).
      ap = pack_c(a, n, eye)
      bp = pack_c(b, nrhs, zed)
      call cqr_mkl_sytrfnp_compact(MKL_COL_MAJOR, MKL_LOWER, n, ap, n, &
                                   info, MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_sytrfnp_compact info')
      call cqr_mkl_sytrsnp_compact(MKL_COL_MAJOR, MKL_LOWER, n, nrhs, &
                                   ap, n, bp, n, info, &
                                   MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_sytrsnp_compact info')
      call check(maxval(abs(unpack_c(bp) - x)) < tol, &
                 'sytrfnp + sytrsnp solution')
      af = ap
      xf = bp
      ap = pack_c(a, n, eye)
      bp = pack_c(b, nrhs, zed)
      call cqr_mkl_sysvnp_compact(MKL_COL_MAJOR, MKL_LOWER, n, nrhs, &
                                  ap, n, bp, n, info, &
                                  MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_sysvnp_compact info')
      call check(all(ap == af) .and. all(bp == xf), &
                 'sysvnp == sytrfnp + sytrsnp')
   end subroutine run_tests

   ! An unrecognized pack format is the one condition info reports (-1).
   subroutine test_format_check()
      real(wp) :: ap(vw*n*n*ng)
      integer(ik) :: info
      ap = 0.0_wp
      call cqr_mkl_potrf_compact(MKL_COL_MAJOR, MKL_LOWER, n, ap, n, &
                                 info, 999, nmat)
      call check(info == -1, 'unrecognized format reports info = -1')
   end subroutine test_format_check

end program test_cqr_fortran_mkl
