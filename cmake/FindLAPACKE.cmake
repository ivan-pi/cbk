# FindLAPACKE.cmake
#
# The LAPACKE + CBLAS stack the test suites validate against, as one target:
#
#   LAPACKE::LAPACKE   include dirs, link line, and the definition that tells
#                      tests/test_lapack_util.hpp which headers to include
#                      (CBK_LAPACKE_MKL, CBK_LAPACKE_ACCELERATE, or none for
#                      the plain <lapacke.h> / <cblas.h>).
#
# With the MKL extension (MKL::Compact defined by FindMKLCompact.cmake) the
# stack is MKL's own -- a test binary must never link two LAPACK
# implementations. Otherwise CMake's FindLAPACK picks the implementation,
# steered by BLA_VENDOR as usual (OpenBLAS, Generic for the distro's plain
# liblapack, Apple for Accelerate). An unset BLA_VENDOR means Generic here,
# not FindLAPACK's "All": that would take an installed MKL first, and MKL's
# LAPACK under Netlib's liblapacke and OpenBLAS's cblas.h is three stacks in
# one binary (measured: the suites fail that way). This module then adds
# what FindLAPACK does not know about:
#
#   lapacke.h and cblas.h  in the default include dirs or a distro's
#                          OpenBLAS subdirectory (openblas/, openblas-pthread/,
#                          ...); with Accelerate, which has CBLAS but no
#                          LAPACKE, lapacke.h comes from an installed
#                          accelerate-lapacke (CMAKE_PREFIX_PATH at its
#                          prefix; https://github.com/lepus2589/accelerate-lapacke)
#                          and CBLAS from the SDK's vecLib sub-framework, whose
#                          framework directory xcrun locates.
#   liblapacke             where LAPACKE is a library of its own: Netlib's,
#                          which Debian also uses for its OpenBLAS, and
#                          accelerate-lapacke's. A full OpenBLAS carries the
#                          symbols itself and has no such library.
#
# One link test of LAPACKE_dgeqrf, LAPACKE_dgeqr2, cblas_dgemm and cblas_dtrsm
# accepts the result. ILP64 comes through MKL only.
#
# Assisted-by: Claude:claude-fable-5

include(CMakePushCheckState)
include(CheckCXXSourceCompiles)

set(_lapacke_includes "")
set(_lapacke_libs "")
set(_lapacke_defs "")
set(_lapacke_options "")

if(TARGET MKL::Compact)
  set(_lapacke_libs MKL::Compact)
  set(_lapacke_defs CBK_LAPACKE_MKL)
  set(LAPACKE_VENDOR_FOUND "MKL")
else()
  if(NOT BLA_VENDOR)
    set(BLA_VENDOR Generic)
  endif()
  find_package(LAPACK REQUIRED)
  set(_openblas_dirs openblas openblas-pthread openblas-openmp openblas-serial)
  find_path(LAPACKE_INCLUDE_DIR NAMES lapacke.h PATH_SUFFIXES ${_openblas_dirs})
  find_library(LAPACKE_LIBRARY NAMES lapacke)
  mark_as_advanced(LAPACKE_INCLUDE_DIR LAPACKE_LIBRARY)
  set(_lapacke_includes ${LAPACKE_INCLUDE_DIR})
  set(_lapacke_libs LAPACK::LAPACK)
  if(LAPACKE_LIBRARY)
    list(PREPEND _lapacke_libs "${LAPACKE_LIBRARY}")
  endif()
  if(APPLE AND LAPACK_LIBRARIES MATCHES "Accelerate")
    execute_process(COMMAND xcrun --show-sdk-path
      OUTPUT_VARIABLE _sdk OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    set(_lapacke_options "-F${_sdk}/System/Library/Frameworks/Accelerate.framework/Frameworks")
    set(_lapacke_defs CBK_LAPACKE_ACCELERATE ACCELERATE_NEW_LAPACK)
  else()
    find_path(CBLAS_INCLUDE_DIR NAMES cblas.h PATH_SUFFIXES ${_openblas_dirs})
    mark_as_advanced(CBLAS_INCLUDE_DIR)
    list(APPEND _lapacke_includes ${CBLAS_INCLUDE_DIR})
  endif()
  set(LAPACKE_VENDOR_FOUND "${BLA_VENDOR}")
endif()

# The link test, against exactly what the target will carry (cached per stack).
string(MAKE_C_IDENTIFIER "LAPACKE_LINK_TEST_${LAPACKE_VENDOR_FOUND}" _lapacke_check)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_INCLUDES ${_lapacke_includes})
set(CMAKE_REQUIRED_LIBRARIES ${_lapacke_libs})
set(CMAKE_REQUIRED_FLAGS "${_lapacke_options}")
foreach(_d IN LISTS _lapacke_defs)
  list(APPEND CMAKE_REQUIRED_DEFINITIONS "-D${_d}")
endforeach()
set(CMAKE_REQUIRED_QUIET TRUE)
check_cxx_source_compiles("
#if defined(CBK_LAPACKE_MKL)
#include <mkl_lapacke.h>
#include <mkl_cblas.h>
#elif defined(CBK_LAPACKE_ACCELERATE)
#include <lapacke.h>
#include <vecLib/cblas.h>
#else
#include <lapacke.h>
#include <cblas.h>
#endif
int main() {
    double a[4] = {2, 0, 0, 2}, tau[2], c[4] = {1, 0, 0, 1};
    lapack_int info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, 2, 2, a, 2, tau);
    info += LAPACKE_dgeqr2(LAPACK_COL_MAJOR, 2, 2, a, 2, tau);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1.0, a, 2, c, 2, 0.0, c, 2);
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 2, 2, 1.0, a, 2, c, 2);
    return (int)info;
}
" ${_lapacke_check})
cmake_pop_check_state()
set(LAPACKE_LINKS ${${_lapacke_check}})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LAPACKE
  REQUIRED_VARS LAPACKE_LINKS
  REASON_FAILURE_MESSAGE
    "no LAPACKE + CBLAS next to the LAPACK found. Debian/Ubuntu: libopenblas-dev liblapacke-dev, or liblapacke-dev liblapack-dev libblas-dev with -DBLA_VENDOR=Generic. macOS: an installed accelerate-lapacke on CMAKE_PREFIX_PATH with -DBLA_VENDOR=Apple. Or -DCBK_WITH_MKL=ON for MKL's. The tests need one, the library does not (-DCBK_BUILD_TESTS=OFF).")

if(LAPACKE_FOUND AND NOT TARGET LAPACKE::LAPACKE)
  add_library(LAPACKE::LAPACKE INTERFACE IMPORTED)
  set_target_properties(LAPACKE::LAPACKE PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_lapacke_includes}"
    INTERFACE_LINK_LIBRARIES "${_lapacke_libs}"
    INTERFACE_COMPILE_DEFINITIONS "${_lapacke_defs}"
    INTERFACE_COMPILE_OPTIONS "${_lapacke_options}")
  message(STATUS "LAPACKE for the tests: ${LAPACKE_VENDOR_FOUND} (${_lapacke_libs})")
endif()
