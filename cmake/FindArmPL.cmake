# FindArmPL.cmake
#
# Locate Arm Performance Libraries (BLAS/LAPACK with the standard CBLAS and
# LAPACKE C interfaces, AArch64) and expose:
#
#   ArmPL::ArmPL  -- the ArmPL include dir plus the ArmPL link line.
#
# Searched in the same prefix order for headers and libraries so they come
# from one installation: the package root (-DArmPL_ROOT=<prefix>), then the
# environment ArmPL's own setup exports ($ARMPL_INCLUDES / $ARMPL_LIBRARIES
# and their parent $ARMPL_DIR -- both `module load armpl` and the install's
# armpl_env_vars.sh set all three), then the versioned prefixes under
# /opt/arm (where the arm-performance-libraries apt package and the tarballs
# install; see https://learn.arm.com/install-guides/armpl/), newest first.
#
# The serial LP64 library is chosen on purpose: 32-bit integers match the
# benchmark's int arguments, and the benchmarks drive their per-matrix loops
# with OpenMP themselves, so ArmPL threading inside each call would only
# oversubscribe -- the same reason the MKL benchmarks link sequential MKL.
# ArmPL's LAPACK is Fortran-built, so the link is tested twice: plain first
# (a shared ArmPL usually carries its Fortran runtime itself -- the flang
# variants always do), then with gfortran's runtime appended, which the
# gcc-variant packages need on the line (have the gfortran package installed
# for it, and pair the gcc-variant ArmPL with g++, whose driver knows the
# runtime's path).
#
# Assisted-by: Claude

# Versioned prefixes of the apt/tarball install (e.g. /opt/arm/armpl_26.07_gcc),
# newest first so a multi-version install resolves to the latest.
file(GLOB _armpl_prefixes "/opt/arm/armpl_*")
list(SORT _armpl_prefixes COMPARE NATURAL ORDER DESCENDING)

# armpl.h is the find marker (unambiguously ArmPL, unlike lapacke.h, which a
# reference-LAPACKE install also has); the same directory holds lapacke.h.
find_path(ArmPL_INCLUDE_DIR NAMES armpl.h
  HINTS ENV ARMPL_INCLUDES ENV ARMPL_DIR PATHS ${_armpl_prefixes}
  PATH_SUFFIXES include)
mark_as_advanced(ArmPL_INCLUDE_DIR)

# libarmpl in current releases; libarmpl_lp64 is the older name of the same
# serial LP64 library.
find_library(ArmPL_LIBRARY NAMES armpl armpl_lp64
  HINTS ENV ARMPL_LIBRARIES ENV ARMPL_DIR PATHS ${_armpl_prefixes}
  PATH_SUFFIXES lib)
mark_as_advanced(ArmPL_LIBRARY)

# libamath, ArmPL's vectorized libm, ships alongside and backs some kernels;
# harmless to omit where absent, so it is optional.
find_library(ArmPL_amath_LIBRARY NAMES amath
  HINTS ENV ARMPL_LIBRARIES ENV ARMPL_DIR PATHS ${_armpl_prefixes}
  PATH_SUFFIXES lib)
mark_as_advanced(ArmPL_amath_LIBRARY)

set(_armpl_link "${ArmPL_LIBRARY}")
if(ArmPL_amath_LIBRARY)
  list(APPEND _armpl_link "${ArmPL_amath_LIBRARY}")
endif()
list(APPEND _armpl_link m)

# Link test: the LAPACKE interface must resolve against the chosen line --
# plain, else with gfortran appended (see above). A failure of both (say, a
# gcc-variant ArmPL without the gfortran package) fails here rather than in
# the build.
if(ArmPL_INCLUDE_DIR AND ArmPL_LIBRARY)
  include(CMakePushCheckState)
  include(CheckCXXSourceCompiles)
  set(_armpl_test_src "
#include <lapacke.h>
int main() { double a = 1.0, tau = 0.0; return (int)LAPACKE_dgeqrf(LAPACK_COL_MAJOR, 1, 1, &a, 1, &tau); }
")
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_INCLUDES "${ArmPL_INCLUDE_DIR}")
  set(CMAKE_REQUIRED_LIBRARIES ${_armpl_link})
  check_cxx_source_compiles("${_armpl_test_src}" ArmPL_LINKS_PLAIN)
  if(NOT ArmPL_LINKS_PLAIN)
    set(CMAKE_REQUIRED_LIBRARIES ${_armpl_link} gfortran)
    check_cxx_source_compiles("${_armpl_test_src}" ArmPL_LINKS_WITH_GFORTRAN)
    if(ArmPL_LINKS_WITH_GFORTRAN)
      list(APPEND _armpl_link gfortran)
    endif()
  endif()
  cmake_pop_check_state()
  if(ArmPL_LINKS_PLAIN OR ArmPL_LINKS_WITH_GFORTRAN)
    set(ArmPL_HAS_LAPACKE TRUE)
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ArmPL
  REQUIRED_VARS ArmPL_INCLUDE_DIR ArmPL_LIBRARY ArmPL_HAS_LAPACKE
  REASON_FAILURE_MESSAGE
    "Install Arm Performance Libraries (Ubuntu: the arm-performance-libraries package from Arm's apt repository, https://learn.arm.com/install-guides/armpl/) plus gfortran for the gcc variant, and for a non-default location set the ArmPL environment (module load armpl, or source armpl_env_vars.sh) or -DArmPL_ROOT=<prefix>. Or build without the benchmark: -DCQR_WITH_ARMPL=OFF.")

if(ArmPL_FOUND AND NOT TARGET ArmPL::ArmPL)
  add_library(ArmPL::ArmPL INTERFACE IMPORTED)
  set_target_properties(ArmPL::ArmPL PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ArmPL_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${_armpl_link}")
endif()
