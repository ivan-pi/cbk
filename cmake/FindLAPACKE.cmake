# FindLAPACKE.cmake
#
# Locate a LAPACKE + CBLAS stack for the test suites and expose it as
#
#   LAPACKE::LAPACKE   -- the include dirs, the link line, and one compile
#                         definition naming the stack's headers:
#                         CBK_LAPACKE_MKL        <mkl_lapacke.h>, <mkl_cblas.h>
#                         CBK_LAPACKE_ACCELERATE <lapacke.h>, <vecLib/cblas.h>
#                         (none)                 <lapacke.h>, <cblas.h>
#
#   LAPACKE_VENDOR   Any (default) | MKL | OpenBLAS | Netlib | Accelerate
#                    which stack to look for; Any tries OpenBLAS, then Netlib,
#                    then whatever `lapacke` library and find_package(LAPACK)
#                    turn up.
#   LAPACKE_VENDOR_FOUND   the stack that was found, as one of the names above.
#
# Every candidate is link-tested with a program that calls LAPACKE_dgeqrf,
# LAPACKE_dgeqr2, cblas_dgemm and cblas_dtrsm, so a header without its
# library (or the other way round) is rejected rather than found.
#
#   MKL         reuses FindMKLCompact.cmake's MKL::Compact (the whole MKL link
#               line, which carries LAPACKE and CBLAS). With -DCBK_WITH_MKL=ON
#               this is the only acceptable stack: a test binary must never
#               link two LAPACK implementations.
#   OpenBLAS    pkg-config `openblas` (the distro's include subdirectory --
#               Debian's openblas-pthread/, Fedora's openblas/ -- comes from
#               there), else cblas.h + libopenblas by hand. Debian and Ubuntu
#               strip LAPACKE out of libopenblas and ship Netlib's
#               liblapacke separately, so when the link test fails on
#               libopenblas alone it is retried with liblapacke added.
#   Netlib      liblapacke + liblapack + libblas. On Debian the plain
#               liblapack.so / libblas.so are update-alternatives symlinks
#               that usually point at OpenBLAS, so the reference libraries
#               are looked for under the lapack/ and blas/ subdirectories
#               first (PATH_SUFFIXES), which is where the distro keeps them.
#   Accelerate  macOS: Apple's Accelerate framework provides LAPACK and
#               CBLAS but no LAPACKE; https://github.com/lepus2589/accelerate-lapacke
#               builds one on top of it (Netlib's LAPACKE, so it needs a
#               Fortran compiler to configure) and installs a `lapacke`
#               CMake package. .github/workflows/macos.yml builds it on the
#               macOS runners.
#
# ILP64 is supported through MKL only (MKL_ILP64 comes with MKL::Compact);
# the other stacks are used at their default 32-bit integer interface.
#
# Assisted-by: Claude:claude-fable-5

set(LAPACKE_VENDOR "Any" CACHE STRING
  "LAPACKE/CBLAS stack for the tests: Any | MKL | OpenBLAS | Netlib | Accelerate")
set_property(CACHE LAPACKE_VENDOR PROPERTY STRINGS Any MKL OpenBLAS Netlib Accelerate)
if(NOT LAPACKE_VENDOR MATCHES "^(Any|MKL|OpenBLAS|Netlib|Accelerate)$")
  message(FATAL_ERROR "LAPACKE_VENDOR must be Any, MKL, OpenBLAS, Netlib or Accelerate (got '${LAPACKE_VENDOR}')")
endif()

include(CMakePushCheckState)
include(CheckCXXSourceCompiles)

# The link test. `defs` selects the header set exactly as the test util does.
function(_lapacke_link_test vendor includes libs defs outvar)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_INCLUDES ${includes})
  set(CMAKE_REQUIRED_LIBRARIES ${libs})
  set(CMAKE_REQUIRED_DEFINITIONS ${defs})
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
  " LAPACKE_LINKS_${vendor})
  cmake_pop_check_state()
  set(${outvar} ${LAPACKE_LINKS_${vendor}} PARENT_SCOPE)
endfunction()

set(_lapacke_found_vendor "")
set(_lapacke_includes "")
set(_lapacke_libs "")
set(_lapacke_defs "")
set(_lapacke_options "")

# ---- MKL -------------------------------------------------------------------
if(NOT _lapacke_found_vendor AND LAPACKE_VENDOR STREQUAL "MKL")
  find_package(MKLCompact REQUIRED)
  _lapacke_link_test(MKL "${MKLCompact_INCLUDE_DIR}" "MKL::Compact" "-DCBK_LAPACKE_MKL" _ok)
  if(_ok)
    set(_lapacke_found_vendor MKL)
    set(_lapacke_includes "")           # MKL::Compact carries them
    set(_lapacke_libs MKL::Compact)
    set(_lapacke_defs CBK_LAPACKE_MKL)
  endif()
endif()

# ---- OpenBLAS ---------------------------------------------------------------
if(NOT _lapacke_found_vendor AND LAPACKE_VENDOR MATCHES "^(Any|OpenBLAS)$")
  find_package(PkgConfig QUIET)
  set(_ob_includes "")
  set(_ob_libs "")
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(_ob QUIET openblas)
    if(_ob_FOUND)
      set(_ob_includes ${_ob_INCLUDE_DIRS})
      set(_ob_libs ${_ob_LINK_LIBRARIES})
    endif()
  endif()
  if(NOT _ob_libs)
    find_library(LAPACKE_OpenBLAS_LIBRARY NAMES openblas
      PATH_SUFFIXES openblas openblas-pthread openblas-openmp openblas-serial)
    find_path(LAPACKE_OpenBLAS_INCLUDE_DIR NAMES openblas_config.h
      PATH_SUFFIXES openblas openblas-pthread openblas-openmp openblas-serial)
    mark_as_advanced(LAPACKE_OpenBLAS_LIBRARY LAPACKE_OpenBLAS_INCLUDE_DIR)
    if(LAPACKE_OpenBLAS_LIBRARY AND LAPACKE_OpenBLAS_INCLUDE_DIR)
      set(_ob_libs "${LAPACKE_OpenBLAS_LIBRARY}")
      set(_ob_includes "${LAPACKE_OpenBLAS_INCLUDE_DIR}")
    endif()
  endif()
  if(_ob_libs)
    # lapacke.h sits next to cblas.h in a full OpenBLAS install, or in the
    # default include dir when the distro ships Netlib's LAPACKE separately.
    find_path(LAPACKE_OpenBLAS_LAPACKE_INCLUDE_DIR NAMES lapacke.h
      HINTS ${_ob_includes}
      PATH_SUFFIXES openblas openblas-pthread openblas-openmp openblas-serial)
    mark_as_advanced(LAPACKE_OpenBLAS_LAPACKE_INCLUDE_DIR)
    if(LAPACKE_OpenBLAS_LAPACKE_INCLUDE_DIR)
      list(APPEND _ob_includes "${LAPACKE_OpenBLAS_LAPACKE_INCLUDE_DIR}")
      list(REMOVE_DUPLICATES _ob_includes)
      _lapacke_link_test(OpenBLAS "${_ob_includes}" "${_ob_libs}" "" _ok)
      if(NOT _ok)
        # Debian/Ubuntu: LAPACKE lives in Netlib's liblapacke, which calls
        # back into whichever LAPACK is loaded -- libopenblas here, since it
        # is linked first and exports the LAPACK symbols itself.
        find_library(LAPACKE_OpenBLAS_LAPACKE_LIBRARY NAMES lapacke)
        mark_as_advanced(LAPACKE_OpenBLAS_LAPACKE_LIBRARY)
        if(LAPACKE_OpenBLAS_LAPACKE_LIBRARY)
          list(PREPEND _ob_libs "${LAPACKE_OpenBLAS_LAPACKE_LIBRARY}")
          _lapacke_link_test(OpenBLAS_lapacke "${_ob_includes}" "${_ob_libs}" "" _ok)
        endif()
      endif()
      if(_ok)
        set(_lapacke_found_vendor OpenBLAS)
        set(_lapacke_includes ${_ob_includes})
        set(_lapacke_libs ${_ob_libs})
      endif()
    endif()
  endif()
endif()

# ---- Netlib -----------------------------------------------------------------
if(NOT _lapacke_found_vendor AND LAPACKE_VENDOR MATCHES "^(Any|Netlib)$")
  find_path(LAPACKE_Netlib_INCLUDE_DIR NAMES lapacke.h PATH_SUFFIXES lapacke)
  find_path(LAPACKE_Netlib_CBLAS_INCLUDE_DIR NAMES cblas.h PATH_SUFFIXES cblas)
  find_library(LAPACKE_Netlib_LAPACKE_LIBRARY NAMES lapacke)
  # The reference libraries in their own subdirectories first (Debian), then
  # the plain names.
  find_library(LAPACKE_Netlib_LAPACK_LIBRARY NAMES lapack PATH_SUFFIXES lapack)
  find_library(LAPACKE_Netlib_BLAS_LIBRARY NAMES blas PATH_SUFFIXES blas)
  mark_as_advanced(LAPACKE_Netlib_INCLUDE_DIR LAPACKE_Netlib_CBLAS_INCLUDE_DIR
    LAPACKE_Netlib_LAPACKE_LIBRARY LAPACKE_Netlib_LAPACK_LIBRARY LAPACKE_Netlib_BLAS_LIBRARY)
  if(LAPACKE_Netlib_INCLUDE_DIR AND LAPACKE_Netlib_CBLAS_INCLUDE_DIR
     AND LAPACKE_Netlib_LAPACKE_LIBRARY AND LAPACKE_Netlib_LAPACK_LIBRARY
     AND LAPACKE_Netlib_BLAS_LIBRARY)
    set(_nl_includes "${LAPACKE_Netlib_INCLUDE_DIR}" "${LAPACKE_Netlib_CBLAS_INCLUDE_DIR}")
    list(REMOVE_DUPLICATES _nl_includes)
    set(_nl_libs "${LAPACKE_Netlib_LAPACKE_LIBRARY}" "${LAPACKE_Netlib_LAPACK_LIBRARY}"
                 "${LAPACKE_Netlib_BLAS_LIBRARY}")
    # A test binary calls LAPACKE and CBLAS, never LAPACK itself, so GNU ld's
    # --as-needed (Debian's default) would drop liblapack from its NEEDED
    # list and liblapacke would then load `liblapack.so.3` through the
    # loader cache -- the alternatives symlink, i.e. OpenBLAS again. Keeping
    # it a direct dependency makes the loader take the reference library
    # from the RUNPATH CMake sets for the build tree, and liblapacke's own
    # request for the same soname is then satisfied by the copy already
    # loaded.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT MSVC)
      list(PREPEND _nl_libs "-Wl,--no-as-needed")
    endif()
    _lapacke_link_test(Netlib "${_nl_includes}" "${_nl_libs}" "" _ok)
    if(_ok)
      set(_lapacke_found_vendor Netlib)
      set(_lapacke_includes ${_nl_includes})
      set(_lapacke_libs ${_nl_libs})
    endif()
  endif()
endif()

# ---- Accelerate (macOS) -----------------------------------------------------
# accelerate-lapacke installs Netlib's lapacke-config.cmake (looked up under
# its own lower-case file name, so this module is not re-entered) with the
# `lapacke` target; CBLAS is Accelerate's, reached as <vecLib/cblas.h> through
# the framework directory of the SDK's Accelerate.framework, which
# `xcrun --show-sdk-path` locates. Only the vecLib sub-framework's header is
# included, never the <Accelerate/Accelerate.h> umbrella: its LAPACK
# prototypes and the ones lapacke.h declares are the same C symbols with
# different parameter types.
if(NOT _lapacke_found_vendor AND LAPACKE_VENDOR STREQUAL "Accelerate")
  find_package(lapacke CONFIG QUIET)
  find_library(LAPACKE_Accelerate_FRAMEWORK Accelerate)
  mark_as_advanced(LAPACKE_Accelerate_FRAMEWORK)
  execute_process(COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE _ac_sdk OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  set(_ac_frameworks "${_ac_sdk}/System/Library/Frameworks/Accelerate.framework/Frameworks")
  if(TARGET lapacke AND LAPACKE_Accelerate_FRAMEWORK AND EXISTS "${_ac_frameworks}/vecLib.framework")
    set(_ac_libs lapacke "${LAPACKE_Accelerate_FRAMEWORK}")
    set(_ac_defs -DCBK_LAPACKE_ACCELERATE -DACCELERATE_NEW_LAPACK "-F${_ac_frameworks}")
    _lapacke_link_test(Accelerate "" "${_ac_libs}" "${_ac_defs}" _ok)
    if(_ok)
      set(_lapacke_found_vendor Accelerate)
      set(_lapacke_libs ${_ac_libs})
      set(_lapacke_defs CBK_LAPACKE_ACCELERATE ACCELERATE_NEW_LAPACK)
      set(_lapacke_options "-F${_ac_frameworks}")
    endif()
  endif()
endif()

# ---- Any other lapacke + LAPACK/BLAS pair -----------------------------------
if(NOT _lapacke_found_vendor AND LAPACKE_VENDOR STREQUAL "Any")
  find_path(LAPACKE_Generic_INCLUDE_DIR NAMES lapacke.h)
  find_path(LAPACKE_Generic_CBLAS_INCLUDE_DIR NAMES cblas.h)
  find_library(LAPACKE_Generic_LAPACKE_LIBRARY NAMES lapacke)
  mark_as_advanced(LAPACKE_Generic_INCLUDE_DIR LAPACKE_Generic_CBLAS_INCLUDE_DIR
    LAPACKE_Generic_LAPACKE_LIBRARY)
  find_package(LAPACK QUIET)
  if(LAPACKE_Generic_INCLUDE_DIR AND LAPACKE_Generic_CBLAS_INCLUDE_DIR
     AND LAPACKE_Generic_LAPACKE_LIBRARY AND LAPACK_FOUND)
    set(_ge_includes "${LAPACKE_Generic_INCLUDE_DIR}" "${LAPACKE_Generic_CBLAS_INCLUDE_DIR}")
    list(REMOVE_DUPLICATES _ge_includes)
    set(_ge_libs "${LAPACKE_Generic_LAPACKE_LIBRARY}" ${LAPACK_LIBRARIES})
    _lapacke_link_test(Generic "${_ge_includes}" "${_ge_libs}" "" _ok)
    if(_ok)
      set(_lapacke_found_vendor Generic)
      set(_lapacke_includes ${_ge_includes})
      set(_lapacke_libs ${_ge_libs})
    endif()
  endif()
endif()

set(LAPACKE_VENDOR_FOUND "${_lapacke_found_vendor}")
set(LAPACKE_INCLUDE_DIRS "${_lapacke_includes}")
set(LAPACKE_LIBRARIES "${_lapacke_libs}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LAPACKE
  REQUIRED_VARS LAPACKE_VENDOR_FOUND
  REASON_FAILURE_MESSAGE
    "No LAPACKE + CBLAS stack for LAPACKE_VENDOR=${LAPACKE_VENDOR}. Install one (Debian/Ubuntu: libopenblas-dev, or liblapacke-dev with liblapack-dev, or libmkl-dev with -DCBK_WITH_MKL=ON) or point CMAKE_PREFIX_PATH at it. The tests need one, the library itself does not (-DCBK_BUILD_TESTS=OFF).")

if(LAPACKE_FOUND AND NOT TARGET LAPACKE::LAPACKE)
  add_library(LAPACKE::LAPACKE INTERFACE IMPORTED)
  set_target_properties(LAPACKE::LAPACKE PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${LAPACKE_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${LAPACKE_LIBRARIES}"
    INTERFACE_COMPILE_DEFINITIONS "${_lapacke_defs}"
    INTERFACE_COMPILE_OPTIONS "${_lapacke_options}")
  message(STATUS "LAPACKE for the tests: ${LAPACKE_VENDOR_FOUND} (${LAPACKE_LIBRARIES})")
endif()
