! Fortran-interface test for the MKL-style API (include/cqr_mkl_ext.fi),
! compiled as free-form source; test_cqr_fortran_mkl_fixed.f includes the
! same file as fixed form. Every entry point is called through its bind(c)
! interface -- the enumerators transcribed from mkl_types.h select layout,
! uplo, side, trans, diag and the pack format -- and each solve is checked
! against the exact x the right-hand sides were built from. The compact
! buffers are built with RESHAPE (see test_cqr_fortran_compact.f90: PAD
! fills the partial final group, ORDER permutes the lane index innermost),
! so nothing here calls or links MKL itself; the point is that the
! interface blocks match the C signatures and the link line closes.
! MKL_COMPACT_SSE packs V = 2 doubles or 4 floats, so nmat = 3 leaves a
! padding lane in both precisions.
!
! Assisted-by: Claude:claude-fable-5

program test_cqr_fortran_mkl
   use, intrinsic :: iso_c_binding, only: c_int, c_double, c_float
   implicit none
   include 'cqr_mkl_ext.fi'

   integer(c_int), parameter :: nmat = 3, n = 3, nrhs = 2
   integer(c_int), parameter :: vwd = 2, vws = 4 ! MKL_COMPACT_SSE widths
   integer(c_int), parameter :: ngd = (nmat + vwd - 1) / vwd
   integer(c_int), parameter :: ngs = (nmat + vws - 1) / vws

   call test_double()
   call test_single()
   call test_format_check()
   print '(a)', 'test_cqr_fortran_mkl: all checks passed'

contains

   subroutine check(ok, what)
      logical, intent(in) :: ok
      character(*), intent(in) :: what
      if (.not. ok) then
         print '(2a)', 'FAILED: ', what
         error stop 1
      end if
   end subroutine check

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

   function pack_d(dense, ncol, pad) result(packed)
      integer, intent(in) :: ncol
      real(c_double), intent(in) :: dense(n, ncol, nmat), pad(n, ncol)
      real(c_double) :: packed(vwd, n, ncol, ngd)
      packed = reshape(reshape(dense, [n, ncol, vwd, ngd], pad=pad), &
                       [vwd, n, ncol, ngd], order=[2, 3, 1, 4])
   end function pack_d

   function unpack_d(packed) result(dense)
      real(c_double), intent(in) :: packed(vwd, n, nrhs, ngd)
      real(c_double) :: full(n, nrhs, vwd*ngd), dense(n, nrhs, nmat)
      full = reshape(reshape(packed, [n, nrhs, vwd, ngd], &
                             order=[3, 1, 2, 4]), shape(full))
      dense = full(:, :, 1:nmat)
   end function unpack_d

   subroutine test_double()
      real(c_double) :: a(n, n, nmat), x(n, nrhs, nmat), b(n, nrhs, nmat)
      real(c_double) :: eye(n, n), zed(n, nrhs)
      real(c_double) :: ap(vwd, n, n, ngd), bp(vwd, n, nrhs, ngd)
      real(c_double) :: taup(vwd, n, ngd), work(vwd*n*ngd)
      real(c_double) :: af(vwd, n, n, ngd), xf(vwd, n, nrhs, ngd)
      real(c_double), parameter :: tol = 1.0e-11_c_double
      integer(c_int) :: info
      integer :: i

      call fill_d(a, x, b)
      eye = 0.0_c_double
      do i = 1, n
         eye(i, i) = 1.0_c_double
      end do
      zed = 0.0_c_double

      ! QR chain, with the lwork = -1 workspace queries first: these
      ! kernels need no scratch, so each query reports 1.
      ap = pack_d(a, n, eye)
      bp = pack_d(b, nrhs, zed)
      call cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap, n, taup, &
                                  work, -1, info, MKL_COMPACT_SSE, nmat)
      call check(info == 0 .and. nint(work(1)) == 1, 'dgeqrf lwork query')
      call cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap, n, taup, &
                                  work, 1, info, MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_dgeqrf_compact info')
      call cqr_mkl_dormqr_compact(MKL_COL_MAJOR, 'L', 'T', n, nrhs, n, &
                                  ap, n, taup, bp, n, work, -1, info, &
                                  MKL_COMPACT_SSE, nmat)
      call check(info == 0 .and. nint(work(1)) == 1, 'dormqr lwork query')
      call cqr_mkl_dormqr_compact(MKL_COL_MAJOR, 'L', 'T', n, nrhs, n, &
                                  ap, n, taup, bp, n, work, 1, info, &
                                  MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_dormqr_compact info')
      call cqr_mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, &
                                 MKL_NOTRANS, MKL_NONUNIT, n, nrhs, &
                                 1.0_c_double, ap, n, bp, n, &
                                 MKL_COMPACT_SSE, nmat)
      call check(maxval(abs(unpack_d(bp) - x)) < tol, 'd QR chain solution')

      ! gels: its work is the tau scratch, one slot per group -- the query
      ! reports min(m,n) * V * ceil(nm/V) slots, a compact tau buffer.
      ap = pack_d(a, n, eye)
      bp = pack_d(b, nrhs, zed)
      call cqr_mkl_dgels_compact(MKL_COL_MAJOR, 'N', n, n, nrhs, ap, n, &
                                 bp, n, work, -1, info, &
                                 MKL_COMPACT_SSE, nmat)
      call check(info == 0 .and. nint(work(1)) == n * vwd * ngd, &
                 'dgels lwork query')
      call cqr_mkl_dgels_compact(MKL_COL_MAJOR, 'N', n, n, nrhs, ap, n, &
                                 bp, n, work, n * vwd * ngd, info, &
                                 MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_dgels_compact info')
      call check(maxval(abs(unpack_d(bp) - x)) < tol, 'dgels solution')

      ! Cholesky: potrf, then two triangular solves close A x = b.
      ap = pack_d(a, n, eye)
      bp = pack_d(b, nrhs, zed)
      call cqr_mkl_dpotrf_compact(MKL_COL_MAJOR, MKL_LOWER, n, ap, n, &
                                  info, MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_dpotrf_compact info')
      call cqr_mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, &
                                 MKL_NOTRANS, MKL_NONUNIT, n, nrhs, &
                                 1.0_c_double, ap, n, bp, n, &
                                 MKL_COMPACT_SSE, nmat)
      call cqr_mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, &
                                 MKL_TRANS, MKL_NONUNIT, n, nrhs, &
                                 1.0_c_double, ap, n, bp, n, &
                                 MKL_COMPACT_SSE, nmat)
      call check(maxval(abs(unpack_d(bp) - x)) < tol, 'dpotrf + dtrsm')

      ! LDL^T: sytrfnp + sytrsnp, then the fused sysvnp (bit-identical).
      ap = pack_d(a, n, eye)
      bp = pack_d(b, nrhs, zed)
      call cqr_mkl_dsytrfnp_compact(MKL_COL_MAJOR, MKL_LOWER, n, ap, n, &
                                    info, MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_dsytrfnp_compact info')
      call cqr_mkl_dsytrsnp_compact(MKL_COL_MAJOR, MKL_LOWER, n, nrhs, &
                                    ap, n, bp, n, info, &
                                    MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_dsytrsnp_compact info')
      call check(maxval(abs(unpack_d(bp) - x)) < tol, &
                 'dsytrfnp + dsytrsnp solution')
      af = ap
      xf = bp
      ap = pack_d(a, n, eye)
      bp = pack_d(b, nrhs, zed)
      call cqr_mkl_dsysvnp_compact(MKL_COL_MAJOR, MKL_LOWER, n, nrhs, &
                                   ap, n, bp, n, info, &
                                   MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_dsysvnp_compact info')
      call check(all(ap == af) .and. all(bp == xf), &
                 'dsysvnp == dsytrfnp + dsytrsnp')
   end subroutine test_double

   ! -- single precision: MKL_COMPACT_SSE packs V = 4 floats --

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
      real(c_float) :: packed(vws, n, ncol, ngs)
      packed = reshape(reshape(dense, [n, ncol, vws, ngs], pad=pad), &
                       [vws, n, ncol, ngs], order=[2, 3, 1, 4])
   end function pack_s

   function unpack_s(packed) result(dense)
      real(c_float), intent(in) :: packed(vws, n, nrhs, ngs)
      real(c_float) :: full(n, nrhs, vws*ngs), dense(n, nrhs, nmat)
      full = reshape(reshape(packed, [n, nrhs, vws, ngs], &
                             order=[3, 1, 2, 4]), shape(full))
      dense = full(:, :, 1:nmat)
   end function unpack_s

   subroutine test_single()
      real(c_float) :: a(n, n, nmat), x(n, nrhs, nmat), b(n, nrhs, nmat)
      real(c_float) :: eye(n, n), zed(n, nrhs)
      real(c_float) :: ap(vws, n, n, ngs), bp(vws, n, nrhs, ngs)
      real(c_float) :: taup(vws, n, ngs), work(vws*n*ngs)
      real(c_float) :: af(vws, n, n, ngs), xf(vws, n, nrhs, ngs)
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
      call cqr_mkl_sgeqrf_compact(MKL_COL_MAJOR, n, n, ap, n, taup, &
                                  work, -1, info, MKL_COMPACT_SSE, nmat)
      call check(info == 0 .and. nint(work(1)) == 1, 'sgeqrf lwork query')
      call cqr_mkl_sgeqrf_compact(MKL_COL_MAJOR, n, n, ap, n, taup, &
                                  work, 1, info, MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_sgeqrf_compact info')
      call cqr_mkl_sormqr_compact(MKL_COL_MAJOR, 'L', 'T', n, nrhs, n, &
                                  ap, n, taup, bp, n, work, 1, info, &
                                  MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_sormqr_compact info')
      call cqr_mkl_strsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, &
                                 MKL_NOTRANS, MKL_NONUNIT, n, nrhs, &
                                 1.0_c_float, ap, n, bp, n, &
                                 MKL_COMPACT_SSE, nmat)
      call check(maxval(abs(unpack_s(bp) - x)) < tol, 's QR chain solution')

      ap = pack_s(a, n, eye)
      bp = pack_s(b, nrhs, zed)
      call cqr_mkl_sgels_compact(MKL_COL_MAJOR, 'N', n, n, nrhs, ap, n, &
                                 bp, n, work, -1, info, &
                                 MKL_COMPACT_SSE, nmat)
      call check(info == 0 .and. nint(work(1)) == n * vws * ngs, &
                 'sgels lwork query')
      call cqr_mkl_sgels_compact(MKL_COL_MAJOR, 'N', n, n, nrhs, ap, n, &
                                 bp, n, work, n * vws * ngs, info, &
                                 MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_sgels_compact info')
      call check(maxval(abs(unpack_s(bp) - x)) < tol, 'sgels solution')

      ap = pack_s(a, n, eye)
      bp = pack_s(b, nrhs, zed)
      call cqr_mkl_spotrf_compact(MKL_COL_MAJOR, MKL_LOWER, n, ap, n, &
                                  info, MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_spotrf_compact info')
      call cqr_mkl_strsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, &
                                 MKL_NOTRANS, MKL_NONUNIT, n, nrhs, &
                                 1.0_c_float, ap, n, bp, n, &
                                 MKL_COMPACT_SSE, nmat)
      call cqr_mkl_strsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, &
                                 MKL_TRANS, MKL_NONUNIT, n, nrhs, &
                                 1.0_c_float, ap, n, bp, n, &
                                 MKL_COMPACT_SSE, nmat)
      call check(maxval(abs(unpack_s(bp) - x)) < tol, 'spotrf + strsm')

      ap = pack_s(a, n, eye)
      bp = pack_s(b, nrhs, zed)
      call cqr_mkl_ssytrfnp_compact(MKL_COL_MAJOR, MKL_LOWER, n, ap, n, &
                                    info, MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_ssytrfnp_compact info')
      call cqr_mkl_ssytrsnp_compact(MKL_COL_MAJOR, MKL_LOWER, n, nrhs, &
                                    ap, n, bp, n, info, &
                                    MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_ssytrsnp_compact info')
      call check(maxval(abs(unpack_s(bp) - x)) < tol, &
                 'ssytrfnp + ssytrsnp solution')
      af = ap
      xf = bp
      ap = pack_s(a, n, eye)
      bp = pack_s(b, nrhs, zed)
      call cqr_mkl_ssysvnp_compact(MKL_COL_MAJOR, MKL_LOWER, n, nrhs, &
                                   ap, n, bp, n, info, &
                                   MKL_COMPACT_SSE, nmat)
      call check(info == 0, 'cqr_mkl_ssysvnp_compact info')
      call check(all(ap == af) .and. all(bp == xf), &
                 'ssysvnp == ssytrfnp + ssytrsnp')
   end subroutine test_single

   ! An unrecognized pack format is the one condition info reports (-1).
   subroutine test_format_check()
      real(c_double) :: ap(vwd, n, n, ngd)
      integer(c_int) :: info
      ap = 0.0_c_double
      call cqr_mkl_dpotrf_compact(MKL_COL_MAJOR, MKL_LOWER, n, ap, n, &
                                  info, 999, nmat)
      call check(info == -1, 'unrecognized format reports info = -1')
   end subroutine test_format_check

end program test_cqr_fortran_mkl
