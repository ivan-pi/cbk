# Consumer of an *installed* cbk: the install check of
# .github/workflows/install.yml (see .github/check_install.sh) builds
# this against the install prefix, out of tree, exactly as a downstream
# project would:
#
#   cmake -S tests/install -B build-consumer -DCMAKE_PREFIX_PATH=<prefix> \
#         -DCBK_WITH_MKL=ON|OFF
#
# It links cbk::cbk and calls one routine of the portable C API and, when
# cbk was installed with the MKL extension, one of the MKL-style API next to
# MKL's own compact API (linked through the FindMKLCompact module the
# package installs).
cmake_minimum_required(VERSION 3.28)
project(cbk_install_check LANGUAGES CXX)

option(CBK_WITH_MKL "The installed cbk carries the MKL-style API" OFF)

find_package(cbk CONFIG REQUIRED)
if(NOT CBK_WITH_MKL STREQUAL cbk_WITH_MKL)
  message(FATAL_ERROR "expected CBK_WITH_MKL=${CBK_WITH_MKL}, the installed cbk reports cbk_WITH_MKL=${cbk_WITH_MKL}")
endif()

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE cbk::cbk)
if(CBK_WITH_MKL)
  list(APPEND CMAKE_MODULE_PATH "${cbk_DIR}")
  find_package(MKLCompact REQUIRED)
  target_compile_definitions(consumer PRIVATE CBK_CONSUMER_WITH_MKL=1)
  target_link_libraries(consumer PRIVATE MKL::Compact)
endif()
