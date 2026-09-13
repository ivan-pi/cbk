#!/usr/bin/env bash
# Verify an installed cqr: the expected files are present, what should not be
# is absent, and a downstream project can find_package(cqr) and link cqr::cqr.
#
#   tests/install/check_install.sh <prefix> <mkl ON|OFF> <shared ON|OFF>
#
# <prefix> is where `cmake --install <build> --prefix <prefix>` put cqr; the
# two flags say what that build was configured with (CQR_WITH_MKL and
# BUILD_SHARED_LIBS). Run by .github/workflows/install.yml; usable by hand.
set -euo pipefail

prefix=$(cd "$1" && pwd)
mkl=$2
shared=$3
here=$(cd "$(dirname "$0")" && pwd)

fail=0
expect() {
    if [ -e "$1" ]; then echo "ok       $1"; else echo "MISSING  $1"; fail=1; fi
}
expect_absent() {
    if [ -e "$1" ]; then echo "UNWANTED $1"; fail=1; else echo "absent   $1"; fi
}

echo "== install tree under $prefix (MKL=$mkl, shared=$shared)"
# The package lives in <libdir>/cmake/cqr, whatever GNUInstallDirs named libdir.
config=$(find "$prefix" -name cqrConfig.cmake | head -n 1)
expect "${config:-$prefix/lib/cmake/cqr/cqrConfig.cmake}"
cmakedir=$(dirname "$config")
libdir=$(dirname "$(dirname "$cmakedir")")
expect "$cmakedir/cqrConfigVersion.cmake"
expect "$cmakedir/cqrTargets.cmake"
expect "$prefix/include/cqr_compact.h"
if [ "$shared" = ON ]; then
    expect "$libdir/libcqr.so"
    expect_absent "$libdir/libcqr.a"
else
    expect "$libdir/libcqr.a"
    expect_absent "$libdir/libcqr.so"
fi
for f in include/cqr_mkl_ext.h include/cqr_mkl_alloc.h "${cmakedir#"$prefix/"}/FindMKLCompact.cmake"; do
    if [ "$mkl" = ON ]; then expect "$prefix/$f"; else expect_absent "$prefix/$f"; fi
done

if [ "$fail" -ne 0 ]; then
    echo "== full listing"
    find "$prefix" -type f -o -type l | sort
    echo "install tree check FAILED"
    exit 1
fi

echo "== downstream project against the install"
consumer=${CQR_CONSUMER_BUILD_DIR:-$here/../../build-consumer}
cmake -S "$here" -B "$consumer" -DCMAKE_PREFIX_PATH="$prefix" -DCQR_WITH_MKL="$mkl"
cmake --build "$consumer"
"$consumer/consumer"
echo "install check passed"
