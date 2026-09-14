# Assisted-by: Claude:claude-fable-5

cmake_minimum_required(VERSION 3.28)

project(cbk
  VERSION 0.1.0
  DESCRIPTION "Compact Batch Kernels: batched QR, Cholesky and LDL^T for small matrices in the compact (interleaved) format"
  LANGUAGES C CXX) # C only for the portable example, examples/qr_reconstruct_compact.c

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# compile_commands.json, which clang-tidy reads (.clang-tidy and the
# manual-stage hook in .pre-commit-config.yaml)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CBK_WITH_MKL    "Build the Intel MKL compact extension, examples and MKL tests" OFF)
option(CBK_WITH_OPENMP "Thread each routine's loop over compact groups with OpenMP"     ON)
option(CBK_BUILD_TESTS "Build tests and register them with CTest"                     ON)
# The install rules belong to cbk's own build, not to a project that
# add_subdirectory()s it.
option(CBK_INSTALL     "Generate the install target"                     ${PROJECT_IS_TOP_LEVEL})

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
# in src/cbk_common.hpp); the benchmarks use it for their outer loops.
# Linked PRIVATE: consumers get the runtime library, not the -fopenmp flag.
if(CBK_WITH_OPENMP)
  find_package(OpenMP QUIET)
endif()

# ---------------------------------------------------------------------------
# The library: one target, cbk (cbk::cbk). By default the portable C API of
# include/cbk.h (no BLAS, no MKL), which is what a default configure
# builds and tests. Static by default; -DBUILD_SHARED_LIBS=ON builds it shared.
# ---------------------------------------------------------------------------
add_library(cbk src/cbk.cpp)
target_include_directories(cbk
  PUBLIC  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
          "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
# A build requirement of the kernels, not of the public headers (C, and
# C++-agnostic), so consumers of the installed package are not bound to it.
target_compile_features(cbk PRIVATE cxx_std_17)
# The pivots' lane-wise sqrt (vsqrt in src/cbk_common.hpp) is one vector
# instruction only without the C standard's errno-on-negative-input; the
# kernels never read errno. PRIVATE: a codegen setting for the kernels, not an
# ISA or fast-math choice, and consumers are not bound to it.
target_compile_options(cbk PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang,IntelLLVM>:-fno-math-errno>)
# cbk.h's CBK_VERSION_* macros must agree with project(VERSION) above;
# src/cbk.cpp static_asserts it.
target_compile_definitions(cbk PRIVATE
  CBK_CMAKE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  CBK_CMAKE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  CBK_CMAKE_VERSION_PATCH=${PROJECT_VERSION_PATCH})
if(CBK_WITH_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(cbk PRIVATE OpenMP::OpenMP_CXX)
endif()
# libcbk.so.<0.1.0> with soname libcbk.so.<0>, from project(... VERSION) above.
set_target_properties(cbk PROPERTIES
  VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
add_library(cbk::cbk ALIAS cbk)

# The portable example: the QR round trip of examples/qr_reconstruct_compact.c
# (pack, dgeqrf_compact, dormqr_compact, unpack, check) in plain C99 on the C
# API alone, so it builds in the default configure -- a downstream C caller in
# miniature. OpenMP times it and threads its own loops over groups; without
# OpenMP it falls back to clock().
add_executable(qr_reconstruct_compact examples/qr_reconstruct_compact.c)
target_compile_features(qr_reconstruct_compact PRIVATE c_std_99)
set_target_properties(qr_reconstruct_compact PROPERTIES C_EXTENSIONS OFF)
target_link_libraries(qr_reconstruct_compact PRIVATE cbk)
if(CBK_WITH_OPENMP AND OpenMP_C_FOUND)
  target_link_libraries(qr_reconstruct_compact PRIVATE OpenMP::OpenMP_C)
endif()

# ---------------------------------------------------------------------------
# Optional Intel MKL compact extension (-DCBK_WITH_MKL=ON): cbk_?{geqrf,
# ormqr,orgqr,potrf,potrs,posv,sytrfnp,sytrsnp,sysvnp,trsm,gels}_compact, the
# MKL-style API of include/cbk_compat.h over the same kernels (see
# docs/*_design.md), compiled into the same library; plus the MKL-backed
# tests, the example and the benchmarks.
# ---------------------------------------------------------------------------
if(CBK_WITH_MKL)
  find_package(MKLCompact REQUIRED)

  target_sources(cbk PRIVATE src/cbk_compat.cpp)
  # MKL types appear in the public header (no link dependency).
  target_link_libraries(cbk PUBLIC MKL::CompactHeaders)

  # The MKL examples: the worked batched solve and the five throughput
  # benchmarks (examples/BENCHMARKS.md). All link MKL for the pack/unpack
  # routines and the per-matrix LAPACK baselines; the benchmarks parallelize
  # their outer loop with OpenMP when available.
  foreach(prog solve_qr_compact bench_qr_compact bench_geqrf_compact bench_potrf_compact
               bench_posv_compact bench_sysvnp_compact)
    add_executable(${prog} examples/${prog}.cpp)
    target_link_libraries(${prog} PRIVATE cbk MKL::Compact)
    # src/ for the internal header-only helpers the examples share with the
    # tests (cbk_matrix_view.hpp); the public API stays include/.
    target_include_directories(${prog} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if(prog MATCHES "^bench_" AND CBK_WITH_OPENMP AND OpenMP_CXX_FOUND)
      target_link_libraries(${prog} PRIVATE OpenMP::OpenMP_CXX)
    endif()
  endforeach()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(CBK_BUILD_TESTS)
  enable_testing()

  # Portable self-contained suites (no BLAS): each kernel vs a scalar reference
  # of the same unblocked algorithm, plus the C API's LAPACK-style validation.
  foreach(routine geqrf ormqr orgqr potrf sytrfnp trsm gels)
    add_executable(test_${routine}_compact tests/test_${routine}_compact.cpp)
    target_link_libraries(test_${routine}_compact PRIVATE cbk)
    target_include_directories(test_${routine}_compact
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME portable_${routine} COMMAND test_${routine}_compact)
  endforeach()

  # The portable example self-validates (A = Q R rebuilt from the factors), so
  # it doubles as an integration test of the C API: a small batch of square
  # and of tall matrices, 33 of them so the padded last group is exercised too.
  add_test(NAME example_qr_reconstruct_square COMMAND qr_reconstruct_compact 33 4 5 5)
  add_test(NAME example_qr_reconstruct_tall COMMAND qr_reconstruct_compact 33 2 7 5)

  if(CBK_WITH_MKL)
    # MKL + dense-LAPACK validation of each cbk_?*_compact routine (the
    # design docs' section 7): invariants vs LAPACK, cross-checks vs MKL's own
    # compact kernels, and end-to-end solves.
    foreach(routine geqrf ormqr orgqr potrf sytrfnp trsm gels)
      add_executable(test_${routine}_mkl tests/test_${routine}_mkl.cpp)
      target_link_libraries(test_${routine}_mkl PRIVATE cbk MKL::Compact)
      target_include_directories(test_${routine}_mkl
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
      add_test(NAME mkl_${routine}_suites COMMAND test_${routine}_mkl)
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
# only), and a CMake package so that a consumer can `find_package(cbk CONFIG)`
# and link cbk::cbk. The tests and examples are never installed.
# ---------------------------------------------------------------------------
if(CBK_INSTALL)
  include(CMakePackageConfigHelpers)
  set(_cbk_cmake_dir ${CMAKE_INSTALL_LIBDIR}/cmake/cbk)

  install(TARGETS cbk EXPORT cbkTargets)
  install(EXPORT cbkTargets NAMESPACE cbk:: DESTINATION ${_cbk_cmake_dir})
  install(FILES include/cbk.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  if(CBK_WITH_MKL)
    install(FILES include/cbk_compat.h include/cbk_mkl_alloc.h
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    # The config file re-finds MKL's headers with it.
    install(FILES cmake/FindMKLCompact.cmake DESTINATION ${_cbk_cmake_dir})
  endif()

  # Only a static library exports its PRIVATE OpenMP link as a link-only
  # dependency the config file has to re-find; a shared one records the
  # runtime in the .so itself.
  set(CBK_OPENMP_LINK_ONLY OFF)
  if(CBK_WITH_OPENMP AND OpenMP_CXX_FOUND AND NOT BUILD_SHARED_LIBS)
    set(CBK_OPENMP_LINK_ONLY ON)
  endif()
  configure_package_config_file(cmake/cbkConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/cbkConfig.cmake"
    INSTALL_DESTINATION ${_cbk_cmake_dir})
  write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/cbkConfigVersion.cmake"
    COMPATIBILITY SameMinorVersion)
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cbkConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cbkConfigVersion.cmake"
    DESTINATION ${_cbk_cmake_dir})
endif()
