# Consumer of an *installed* cqr: the install check of
# .github/workflows/install.yml (see .github/check_install.sh) builds
# this against the install prefix, out of tree, exactly as a downstream
# project would:
#
#   cmake -S tests/install -B build-consumer -DCMAKE_PREFIX_PATH=<prefix> \
#         -DCQR_WITH_MKL=ON|OFF
#
# It links cqr::cqr and calls one routine of the portable C API and, when
# cqr was installed with the MKL extension, one of the MKL-style API next to
# MKL's own compact API (linked through the FindMKLCompact module the
# package installs).
cmake_minimum_required(VERSION 3.28)
project(cqr_install_check LANGUAGES CXX)

option(CQR_WITH_MKL "The installed cqr carries the MKL-style API" OFF)

find_package(cqr CONFIG REQUIRED)
if(NOT CQR_WITH_MKL STREQUAL cqr_WITH_MKL)
  message(FATAL_ERROR "expected CQR_WITH_MKL=${CQR_WITH_MKL}, the installed cqr reports cqr_WITH_MKL=${cqr_WITH_MKL}")
endif()

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE cqr::cqr)
if(CQR_WITH_MKL)
  list(APPEND CMAKE_MODULE_PATH "${cqr_DIR}")
  find_package(MKLCompact REQUIRED)
  target_compile_definitions(consumer PRIVATE CQR_CONSUMER_WITH_MKL=1)
  target_link_libraries(consumer PRIVATE MKL::Compact)
endif()
