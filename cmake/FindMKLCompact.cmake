# FindMKLCompact.cmake
#
# Locate Intel MKL's Compact-format API directly (no FindBLAS) and expose:
#
#   MKL::CompactHeaders  -- the MKL include dir only (no link line).
#   MKL::Compact         -- the headers plus the MKL link line.
#
# The compact routines (mkl_get_format_compact, mkl_?geqrf_compact, ...) are an
# Intel MKL extension that no other BLAS provides, so this module looks for MKL
# itself: mkl_compact.h, then the interface / threading / core libraries of the
# *same* installation (the library dir is derived from the include dir, so a
# oneAPI install under MKLROOT is never paired with a distro libmkl in
# /usr/lib), and link-tests mkl_get_format_compact against them.
#
# Controls (cache variables, or the environment):
#
#   MKLCompact_ROOT       MKL install prefix (default: $ENV{MKLROOT}; then the
#                         distro package under /usr and /opt/intel/oneapi/mkl/latest).
#   MKLCompact_THREADING  sequential (default) | gnu | intel
#                         -> mkl_sequential | mkl_gnu_thread + libgomp | mkl_intel_thread + libiomp5
#   MKLCompact_INTERFACE  lp64 (default) | ilp64   (ilp64 also defines MKL_ILP64)
#
# A BLA_VENDOR of Intel10_64lp / Intel10_64lp_seq / Intel10_64ilp(_seq), the
# form earlier versions of this module took, is still honoured and mapped onto
# the two variables above; any other BLA_VENDOR is an error, since a non-Intel
# BLAS cannot provide the compact API.
#
# Assisted-by: Claude:claude-opus-4.8 Claude:claude-fable-5

if(TARGET MKL::Compact)
  set(MKLCompact_FOUND TRUE)
  return()
endif()

set(MKLCompact_ROOT "$ENV{MKLROOT}" CACHE PATH "Intel MKL install prefix (MKLROOT)")
set(MKLCompact_THREADING "sequential" CACHE STRING "MKL threading layer: sequential | gnu | intel")
set(MKLCompact_INTERFACE "lp64" CACHE STRING "MKL integer interface: lp64 | ilp64")
set_property(CACHE MKLCompact_THREADING PROPERTY STRINGS sequential gnu intel)
set_property(CACHE MKLCompact_INTERFACE PROPERTY STRINGS lp64 ilp64)

# Legacy: the BLA_VENDOR spellings this module used to take.
if(DEFINED BLA_VENDOR AND NOT BLA_VENDOR STREQUAL "")
  if(BLA_VENDOR MATCHES "^Intel10_64(lp|ilp)(_seq)?$")
    if(CMAKE_MATCH_1 STREQUAL "ilp")
      set(MKLCompact_INTERFACE "ilp64")
    endif()
    if(CMAKE_MATCH_2 STREQUAL "_seq")
      set(MKLCompact_THREADING "sequential")
    else()
      set(MKLCompact_THREADING "gnu")
    endif()
  else()
    message(FATAL_ERROR
      "BLA_VENDOR='${BLA_VENDOR}' cannot provide the MKL compact API "
      "(mkl_?geqrf_compact, ...); it is an Intel MKL extension. Unset BLA_VENDOR "
      "(Intel MKL is located directly) or use -DCQR_WITH_MKL=OFF.")
  endif()
endif()

if(NOT MKLCompact_THREADING MATCHES "^(sequential|gnu|intel)$")
  message(FATAL_ERROR "MKLCompact_THREADING must be sequential, gnu or intel (got '${MKLCompact_THREADING}')")
endif()
if(NOT MKLCompact_INTERFACE MATCHES "^(lp64|ilp64)$")
  message(FATAL_ERROR "MKLCompact_INTERFACE must be lp64 or ilp64 (got '${MKLCompact_INTERFACE}')")
endif()

# 1. Headers.
find_path(MKLCompact_INCLUDE_DIR
  NAMES mkl_compact.h
  HINTS
    "${MKLCompact_ROOT}/include"
    "${MKLCompact_ROOT}/include/mkl"
  PATHS
    /usr/include/mkl
    /usr/include
    /opt/intel/oneapi/mkl/latest/include
)

# 2. Libraries, from the same installation as the headers. A oneAPI tree is
#    <root>/include + <root>/lib (older: lib/intel64); the Debian package puts
#    headers in /usr/include/mkl and libraries on the default library path.
set(_mkl_lib_hints)
if(MKLCompact_INCLUDE_DIR)
  get_filename_component(_mkl_inc_parent "${MKLCompact_INCLUDE_DIR}" DIRECTORY)
  foreach(_d "${MKLCompact_INCLUDE_DIR}/.." "${_mkl_inc_parent}/..")
    get_filename_component(_d "${_d}" ABSOLUTE)
    list(APPEND _mkl_lib_hints "${_d}/lib" "${_d}/lib/intel64" "${_d}/lib/x86_64-linux-gnu")
  endforeach()
endif()
if(MKLCompact_ROOT)
  list(APPEND _mkl_lib_hints "${MKLCompact_ROOT}/lib" "${MKLCompact_ROOT}/lib/intel64")
endif()

set(_mkl_interface_lib mkl_intel_${MKLCompact_INTERFACE})
if(MKLCompact_THREADING STREQUAL "sequential")
  set(_mkl_threading_lib mkl_sequential)
else()
  set(_mkl_threading_lib mkl_${MKLCompact_THREADING}_thread)
endif()

foreach(_lib ${_mkl_interface_lib} ${_mkl_threading_lib} mkl_core)
  # Prefer the installation the headers came from; fall back to default paths.
  find_library(MKLCompact_${_lib}_LIBRARY NAMES ${_lib}
    HINTS ${_mkl_lib_hints} NO_DEFAULT_PATH)
  find_library(MKLCompact_${_lib}_LIBRARY NAMES ${_lib})
  mark_as_advanced(MKLCompact_${_lib}_LIBRARY)
endforeach()

set(MKLCompact_LIBRARIES
  "${MKLCompact_${_mkl_interface_lib}_LIBRARY}"
  "${MKLCompact_${_mkl_threading_lib}_LIBRARY}"
  "${MKLCompact_mkl_core_LIBRARY}")

# 3. The OpenMP runtime a threaded layer needs: the compiler's own for
#    mkl_gnu_thread (libgomp with g++; clang's libomp exports the GOMP entry
#    points), libiomp5 for mkl_intel_thread.
set(_mkl_omp_lib)
if(MKLCompact_THREADING STREQUAL "gnu")
  find_package(OpenMP QUIET COMPONENTS CXX)
  set(_mkl_omp_lib ${OpenMP_CXX_LIBRARIES})
elseif(MKLCompact_THREADING STREQUAL "intel")
  find_library(MKLCompact_iomp5_LIBRARY NAMES iomp5
    HINTS ${_mkl_lib_hints}
          "${MKLCompact_ROOT}/../compiler/latest/lib"
          "${MKLCompact_ROOT}/../../compiler/latest/lib"
          /opt/intel/oneapi/compiler/latest/lib)
  mark_as_advanced(MKLCompact_iomp5_LIBRARY)
  set(_mkl_omp_lib "${MKLCompact_iomp5_LIBRARY}")
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^Intel")
    message(WARNING
      "MKLCompact_THREADING=intel links libiomp5 while ${CMAKE_CXX_COMPILER_ID}'s "
      "OpenMP (used by the cqr group loops) is a different runtime; two OpenMP "
      "runtimes in one process are unsupported and gave wrong results here. Use "
      "MKLCompact_THREADING=gnu with GCC/clang, or an Intel compiler.")
  endif()
endif()

# 4. Link test: the compact API must resolve against this exact line. (The
#    result is cached by check_cxx_source_compiles; a fresh build dir re-tests.)
set(_mkl_complete TRUE)
foreach(_v MKLCompact_INCLUDE_DIR MKLCompact_${_mkl_interface_lib}_LIBRARY
           MKLCompact_${_mkl_threading_lib}_LIBRARY MKLCompact_mkl_core_LIBRARY)
  if(NOT ${_v})
    set(_mkl_complete FALSE)
  endif()
endforeach()
if(NOT MKLCompact_THREADING STREQUAL "sequential" AND NOT _mkl_omp_lib)
  set(_mkl_complete FALSE)
endif()
if(NOT _mkl_complete)
  set(MKLCompact_HAS_COMPACT_API FALSE)
else()
  find_package(Threads QUIET)
  include(CheckCXXSourceCompiles)
  set(_save_inc "${CMAKE_REQUIRED_INCLUDES}")
  set(_save_lib "${CMAKE_REQUIRED_LIBRARIES}")
  set(_save_def "${CMAKE_REQUIRED_DEFINITIONS}")
  set(CMAKE_REQUIRED_INCLUDES "${MKLCompact_INCLUDE_DIR}")
  set(CMAKE_REQUIRED_LIBRARIES ${MKLCompact_LIBRARIES} ${_mkl_omp_lib}
      ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS} m)
  if(MKLCompact_INTERFACE STREQUAL "ilp64")
    set(CMAKE_REQUIRED_DEFINITIONS -DMKL_ILP64)
  endif()
  check_cxx_source_compiles("
#include <mkl_compact.h>
int main() { return (int) mkl_get_format_compact(); }
" MKLCompact_HAS_COMPACT_API)
  set(CMAKE_REQUIRED_INCLUDES "${_save_inc}")
  set(CMAKE_REQUIRED_LIBRARIES "${_save_lib}")
  set(CMAKE_REQUIRED_DEFINITIONS "${_save_def}")
endif()

if(NOT MKLCompact_HAS_COMPACT_API)
  message(FATAL_ERROR
    "Intel MKL with the compact API (mkl_get_format_compact, mkl_?geqrf_compact, "
    "...) was not found.\n"
    "  mkl_compact.h        = '${MKLCompact_INCLUDE_DIR}'\n"
    "  ${_mkl_interface_lib}       = '${MKLCompact_${_mkl_interface_lib}_LIBRARY}'\n"
    "  ${_mkl_threading_lib}       = '${MKLCompact_${_mkl_threading_lib}_LIBRARY}'\n"
    "  mkl_core             = '${MKLCompact_mkl_core_LIBRARY}'\n"
    "  OpenMP runtime       = '${_mkl_omp_lib}' (threading=${MKLCompact_THREADING})\n"
    "Install Intel MKL (Debian/Ubuntu: libmkl-dev; or intel-oneapi-mkl-devel from "
    "apt.repos.intel.com) and, for a non-default location, set MKLROOT or "
    "-DMKLCompact_ROOT=<prefix>. See .claude/mkl-install.md. Or build without "
    "the MKL extension: -DCQR_WITH_MKL=OFF.")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKLCompact
  REQUIRED_VARS MKLCompact_INCLUDE_DIR MKLCompact_LIBRARIES MKLCompact_HAS_COMPACT_API)

if(MKLCompact_FOUND)
  add_library(MKL::CompactHeaders INTERFACE IMPORTED)
  set_target_properties(MKL::CompactHeaders PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MKLCompact_INCLUDE_DIR}")
  if(MKLCompact_INTERFACE STREQUAL "ilp64")
    set_target_properties(MKL::CompactHeaders PROPERTIES
      INTERFACE_COMPILE_DEFINITIONS MKL_ILP64)
  endif()

  add_library(MKL::Compact INTERFACE IMPORTED)
  set(_link MKL::CompactHeaders ${MKLCompact_LIBRARIES} ${_mkl_omp_lib})
  if(Threads_FOUND)
    list(APPEND _link Threads::Threads)
  endif()
  if(CMAKE_DL_LIBS)
    list(APPEND _link ${CMAKE_DL_LIBS})
  endif()
  list(APPEND _link m)
  set_target_properties(MKL::Compact PROPERTIES
    INTERFACE_LINK_LIBRARIES "${_link}")
endif()

mark_as_advanced(MKLCompact_INCLUDE_DIR MKLCompact_HAS_COMPACT_API)
