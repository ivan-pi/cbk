# FindMKLCompact.cmake
#
# Locate Intel MKL's Compact-format API and expose:
#
#   MKL::CompactHeaders  -- the MKL include dir only (no link line).
#   MKL::Compact         -- the headers plus the MKL link line.
#
# The compact routines (mkl_get_format_compact, mkl_?geqrf_compact, ...) are an
# Intel MKL extension no other BLAS provides, so this looks for MKL itself:
# mkl_compact.h, then the interface / threading / core libraries, both searched
# in the same prefix order so they come from one installation -- the package
# root (-DMKLCompact_ROOT=<prefix>), then $MKLROOT, then the default paths (the
# distro package), then oneAPI's default location -- and link-tests the API.
#
#   MKLCompact_THREADING  sequential (default) | threaded
#                         threaded links MKL's OpenMP layer for the compiler's
#                         own OpenMP runtime: mkl_gnu_thread (the GOMP ABI, which
#                         libgomp and clang's libomp both provide) or, with an
#                         Intel compiler, mkl_intel_thread (libiomp5).
#   MKLCompact_INTERFACE  lp64 (default) | ilp64   (ilp64 also defines MKL_ILP64)
#
#   find_package(MKLCompact COMPONENTS Headers) stops at the include dir: no
#   library search, no link test, MKL::CompactHeaders only (what the installed
#   cqr package needs). A later full find_package adds MKL::Compact.
#
# Intel's own MKLConfig.cmake is deliberately not used: only oneAPI ships it
# (the Debian/Ubuntu libmkl-dev that CI builds against does not), it is not
# found from MKLROOT alone, its defaults are ilp64 + intel_thread, it links
# -lgomp for the GNU layer under every compiler (a second OpenMP runtime under
# clang), and it puts -m64, -DMKL_ILP64 and an rpath on every consumer. One
# search here covers both installations.
#
# Assisted-by: Claude:claude-opus-4.8 Claude:claude-fable-5

set(MKLCompact_THREADING "sequential" CACHE STRING "MKL threading layer: sequential | threaded")
set(MKLCompact_INTERFACE "lp64" CACHE STRING "MKL integer interface: lp64 | ilp64")
set_property(CACHE MKLCompact_THREADING PROPERTY STRINGS sequential threaded)
set_property(CACHE MKLCompact_INTERFACE PROPERTY STRINGS lp64 ilp64)
if(NOT MKLCompact_THREADING MATCHES "^(sequential|threaded)$")
  message(FATAL_ERROR "MKLCompact_THREADING must be sequential or threaded (got '${MKLCompact_THREADING}')")
endif()
if(NOT MKLCompact_INTERFACE MATCHES "^(lp64|ilp64)$")
  message(FATAL_ERROR "MKLCompact_INTERFACE must be lp64 or ilp64 (got '${MKLCompact_INTERFACE}')")
endif()

# Only oneAPI's default location needs listing: the distro package
# (/usr/include/mkl, /usr/lib/<arch>) is on CMake's default search paths and
# is reached through the `mkl` include suffix and the architecture lib dir.
set(_mkl_paths /opt/intel/oneapi/mkl/latest)

find_path(MKLCompact_INCLUDE_DIR NAMES mkl_compact.h
  HINTS ENV MKLROOT PATHS ${_mkl_paths} PATH_SUFFIXES include include/mkl mkl)
mark_as_advanced(MKLCompact_INCLUDE_DIR)
set(_mkl_required MKLCompact_INCLUDE_DIR)

set(_mkl_headers_only FALSE)
if(MKLCompact_FIND_COMPONENTS STREQUAL "Headers")
  set(_mkl_headers_only TRUE)
endif()
set(MKLCompact_Headers_FOUND TRUE)

if(NOT _mkl_headers_only)
  set(_mkl_threading_lib mkl_sequential)
  set(_mkl_omp)
  if(MKLCompact_THREADING STREQUAL "threaded")
    find_package(OpenMP QUIET COMPONENTS CXX)
    if(CMAKE_CXX_COMPILER_ID MATCHES "^Intel")
      set(_mkl_threading_lib mkl_intel_thread)
    else()
      set(_mkl_threading_lib mkl_gnu_thread)
    endif()
    set(_mkl_omp ${OpenMP_CXX_LIBRARIES})
    list(APPEND _mkl_required OpenMP_CXX_FOUND)
  endif()

  set(MKLCompact_LIBRARIES)
  foreach(_lib mkl_intel_${MKLCompact_INTERFACE} ${_mkl_threading_lib} mkl_core)
    find_library(MKLCompact_${_lib}_LIBRARY NAMES ${_lib}
      HINTS ENV MKLROOT PATHS ${_mkl_paths} PATH_SUFFIXES lib lib/intel64)
    mark_as_advanced(MKLCompact_${_lib}_LIBRARY)
    list(APPEND _mkl_required MKLCompact_${_lib}_LIBRARY)
    list(APPEND MKLCompact_LIBRARIES "${MKLCompact_${_lib}_LIBRARY}")
  endforeach()

  # The shared MKL libraries record their own pthread/dl dependencies; dl and m
  # are listed for a static MKL picked up by find_library.
  set(_mkl_link ${MKLCompact_LIBRARIES} ${_mkl_omp} ${CMAKE_DL_LIBS} m)

  # Link test, once everything is found: the compact API must resolve against
  # this exact line (an MKL too old for it, or a threaded layer without its
  # OpenMP runtime, fails here). Cached per threading/interface choice.
  set(_mkl_ok TRUE)
  foreach(_v IN LISTS _mkl_required)
    if(NOT ${_v})
      set(_mkl_ok FALSE)
    endif()
  endforeach()
  set(_mkl_check MKLCompact_HAS_COMPACT_API_${MKLCompact_THREADING}_${MKLCompact_INTERFACE})
  if(_mkl_ok)
    include(CMakePushCheckState)
    include(CheckCXXSourceCompiles)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES "${MKLCompact_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_LIBRARIES ${_mkl_link})
    if(MKLCompact_INTERFACE STREQUAL "ilp64")
      set(CMAKE_REQUIRED_DEFINITIONS -DMKL_ILP64)
    endif()
    check_cxx_source_compiles("
  #include <mkl_compact.h>
  int main() { return (int) mkl_get_format_compact(); }
  " ${_mkl_check})
    cmake_pop_check_state()
  endif()
endif() # NOT _mkl_headers_only

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKLCompact
  REQUIRED_VARS ${_mkl_required} ${_mkl_check}
  REASON_FAILURE_MESSAGE
    "Install Intel MKL (Debian/Ubuntu: libmkl-dev, or intel-oneapi-mkl-devel from apt.repos.intel.com) and, for a non-default location, set MKLROOT or -DMKLCompact_ROOT=<prefix> (see .claude/mkl-install.md). Or build without the MKL extension: -DCQR_WITH_MKL=OFF.")

if(MKLCompact_FOUND AND NOT TARGET MKL::CompactHeaders)
  add_library(MKL::CompactHeaders INTERFACE IMPORTED)
  set_target_properties(MKL::CompactHeaders PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MKLCompact_INCLUDE_DIR}")
  if(MKLCompact_INTERFACE STREQUAL "ilp64")
    set_target_properties(MKL::CompactHeaders PROPERTIES
      INTERFACE_COMPILE_DEFINITIONS MKL_ILP64)
  endif()
endif()
if(MKLCompact_FOUND AND NOT _mkl_headers_only AND NOT TARGET MKL::Compact)
  add_library(MKL::Compact INTERFACE IMPORTED)
  set_target_properties(MKL::Compact PROPERTIES
    INTERFACE_LINK_LIBRARIES "MKL::CompactHeaders;${_mkl_link}")
endif()
