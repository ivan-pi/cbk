cmake_minimum_required(VERSION 3.18)

project(cqr
  VERSION 0.1.0
  DESCRIPTION "Batched QR, Cholesky and LDL^T for small matrices in the compact (interleaved) format"
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# compile_commands.json, which clang-tidy reads (.clang-tidy and the
# manual-stage hook in .pre-commit-config.yaml)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CQR_WITH_MKL    "Build the Intel MKL compact extension, examples and MKL tests" OFF)
option(CQR_WITH_OPENMP "Thread each routine's loop over compact groups with OpenMP"     ON)
option(CQR_BUILD_TESTS "Build tests and register them with CTest"                     ON)
# The install rules belong to cqr's own build, not to a project that
# add_subdirectory()s it.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(_cqr_top_level ON)
else()
  set(_cqr_top_level OFF)
endif()
option(CQR_INSTALL     "Generate the install target"                                  ${_cqr_top_level})

include(GNUInstallDirs) # the install layout; the targets' install interface uses it

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# C++17, no GNU extensions (the kernels' vector_size attribute compiles under
# strict -std=c++17). Optimization flags are the caller's to choose; for a
# performance build pass them through CMAKE_CXX_FLAGS, e.g.
#   cmake -S . -B build -DCMAKE_CXX_FLAGS="-O3 -march=native"
# so the SIMD kernels target the host's widest vectors.

# OpenMP threads the group loop inside every routine (see the threading note
# in src/cqr_compact_common.hpp); the benchmarks use it for their outer loops.
# Linked PRIVATE: consumers get the runtime library, not the -fopenmp flag.
if(CQR_WITH_OPENMP)
  find_package(OpenMP QUIET)
endif()

# ---------------------------------------------------------------------------
# The library: one target, cqr (cqr::cqr). By default the portable C API of
# include/cqr_compact.h (no BLAS, no MKL), which is what a default configure
# builds and tests. Static by default; -DBUILD_SHARED_LIBS=ON builds it shared.
# ---------------------------------------------------------------------------
add_library(cqr src/cqr_compact.cpp)
target_include_directories(cqr
  PUBLIC  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
          "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
# A build requirement of the kernels, not of the public headers (C, and
# C++-agnostic), so consumers of the installed package are not bound to it.
target_compile_features(cqr PRIVATE cxx_std_17)
if(CQR_WITH_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(cqr PRIVATE OpenMP::OpenMP_CXX)
endif()
# libcqr.so.<0.1.0> with soname libcqr.so.<0>, from project(... VERSION) above.
set_target_properties(cqr PROPERTIES
  VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
add_library(cqr::cqr ALIAS cqr)

# ---------------------------------------------------------------------------
# Optional Intel MKL compact extension (-DCQR_WITH_MKL=ON): cqr_mkl_?{geqrf,
# ormqr,orgqr,potrf,potrs,posv,sytrfnp,sytrsnp,sysvnp,trsm,gels}_compact, the
# MKL-style API of include/cqr_mkl_ext.h over the same kernels (see
# docs/*_design.md), compiled into the same library; plus the MKL-backed
# tests, the example and the benchmarks.
# ---------------------------------------------------------------------------
if(CQR_WITH_MKL)
  find_package(MKLCompact REQUIRED)

  target_sources(cqr PRIVATE src/cqr_mkl_ext.cpp)
  # MKL types appear in the public header (no link dependency). The integer
  # interface is part of cqr's ABI (MKL_INT in every prototype), so the ilp64
  # definition rides on the target itself into the installed package.
  target_link_libraries(cqr PUBLIC MKL::CompactHeaders)
  if(MKLCompact_INTERFACE STREQUAL "ilp64")
    target_compile_definitions(cqr PUBLIC MKL_ILP64)
  endif()

  # Examples: the worked batched solve and the five throughput benchmarks
  # (examples/BENCHMARKS.md). All link MKL for the pack/unpack routines and the
  # per-matrix LAPACK baselines; the benchmarks parallelize their outer loop
  # with OpenMP when available.
  foreach(prog solve_qr_compact bench_qr_compact bench_geqrf_compact bench_potrf_compact
               bench_posv_compact bench_sysvnp_compact)
    add_executable(${prog} examples/${prog}.cpp)
    target_link_libraries(${prog} PRIVATE cqr MKL::Compact)
    # src/ for the internal header-only helpers the examples share with the
    # tests (cqr_matrix_view.hpp); the public API stays include/.
    target_include_directories(${prog} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if(prog MATCHES "^bench_" AND CQR_WITH_OPENMP AND OpenMP_CXX_FOUND)
      target_link_libraries(${prog} PRIVATE OpenMP::OpenMP_CXX)
    endif()
  endforeach()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(CQR_BUILD_TESTS)
  enable_testing()

  # Portable self-contained suites (no BLAS): each kernel vs a scalar reference
  # of the same unblocked algorithm, plus the C API's LAPACK-style validation.
  foreach(routine geqrf ormqr orgqr potrf sytrfnp trsm gels)
    add_executable(test_cqr_${routine}_compact tests/test_cqr_${routine}_compact.cpp)
    target_link_libraries(test_cqr_${routine}_compact PRIVATE cqr)
    target_include_directories(test_cqr_${routine}_compact
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME portable_${routine} COMMAND test_cqr_${routine}_compact)
  endforeach()

  if(CQR_WITH_MKL)
    # MKL + dense-LAPACK validation of each cqr_mkl_?*_compact routine (the
    # design docs' section 7): invariants vs LAPACK, cross-checks vs MKL's own
    # compact kernels, and end-to-end solves.
    foreach(routine geqrf ormqr orgqr potrf sytrfnp trsm gels)
      add_executable(test_cqr_${routine}_mkl tests/test_cqr_${routine}_mkl.cpp)
      target_link_libraries(test_cqr_${routine}_mkl PRIVATE cqr MKL::Compact)
      target_include_directories(test_cqr_${routine}_mkl
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
      add_test(NAME mkl_${routine}_suites COMMAND test_cqr_${routine}_mkl)
    endforeach()

    # The example self-validates, and the benchmarks carry a correctness gate
    # against per-matrix LAPACK, so they double as integration tests. A small
    # pool with a single rep keeps CTest quick; 33 is not a multiple of any
    # SIMD width, so the padded last group is exercised too.
    add_test(NAME example_solve_qr_compact COMMAND solve_qr_compact)
    foreach(prog bench_qr_compact bench_geqrf_compact bench_potrf_compact
                 bench_posv_compact bench_sysvnp_compact)
      add_test(NAME ${prog}_integration COMMAND ${prog} 33 1)
      set_tests_properties(${prog}_integration PROPERTIES LABELS "integration")
    endforeach()
  endif()
endif()

# ---------------------------------------------------------------------------
# Install: the library, the public headers of what was built (the two MKL
# headers need mkl_types.h / mkl_service.h, so they go in with the extension
# only), and a CMake package so that a consumer can `find_package(cqr CONFIG)`
# and link cqr::cqr. The tests and examples are never installed.
# ---------------------------------------------------------------------------
if(CQR_INSTALL)
  include(CMakePackageConfigHelpers)
  set(_cqr_cmake_dir ${CMAKE_INSTALL_LIBDIR}/cmake/cqr)

  install(TARGETS cqr EXPORT cqrTargets)
  install(EXPORT cqrTargets NAMESPACE cqr:: DESTINATION ${_cqr_cmake_dir})
  install(FILES include/cqr_compact.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  if(CQR_WITH_MKL)
    install(FILES include/cqr_mkl_ext.h include/cqr_mkl_alloc.h
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    # The config file re-finds MKL's headers with it.
    install(FILES cmake/FindMKLCompact.cmake DESTINATION ${_cqr_cmake_dir})
  endif()

  # Only a static library exports its PRIVATE OpenMP link as a link-only
  # dependency the config file has to re-find; a shared one records the
  # runtime in the .so itself.
  set(CQR_OPENMP_LINK_ONLY OFF)
  if(CQR_WITH_OPENMP AND OpenMP_CXX_FOUND AND NOT BUILD_SHARED_LIBS)
    set(CQR_OPENMP_LINK_ONLY ON)
  endif()
  configure_package_config_file(cmake/cqrConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/cqrConfig.cmake"
    INSTALL_DESTINATION ${_cqr_cmake_dir})
  write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/cqrConfigVersion.cmake"
    COMPATIBILITY SameMinorVersion)
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cqrConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cqrConfigVersion.cmake"
    DESTINATION ${_cqr_cmake_dir})
endif()
