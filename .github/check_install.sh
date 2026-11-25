#!/usr/bin/env bash
# Verify an installed cbk: the expected files are present, what should not be
# is absent, and a downstream project can find_package(cbk) and link cbk::cbk.
#
#   .github/check_install.sh <prefix> <mkl ON|OFF> <shared ON|OFF>
#
# <prefix> is where `cmake --install <build> --prefix <prefix>` put cbk; the
# two flags say what that build was configured with (CBK_WITH_MKL and
# BUILD_SHARED_LIBS). Run by .github/workflows/install.yml.
set -euo pipefail

prefix=$(cd "$1" && pwd)
mkl=$2
shared=$3
root=$(cd "$(dirname "$0")/.." && pwd)

fail=0
expect() {
    if [ -e "$1" ]; then echo "ok       $1"; else echo "MISSING  $1"; fail=1; fi
}
expect_absent() {
    if [ -e "$1" ]; then echo "UNWANTED $1"; fail=1; else echo "absent   $1"; fi
}

echo "== install tree under $prefix (MKL=$mkl, shared=$shared)"
# The package lives in <libdir>/cmake/cbk, whatever GNUInstallDirs named libdir.
config=$(find "$prefix" -name cbkConfig.cmake | head -n 1)
expect "${config:-$prefix/lib/cmake/cbk/cbkConfig.cmake}"
cmakedir=$(dirname "$config")
libdir=$(dirname "$(dirname "$cmakedir")")
expect "$cmakedir/cbkConfigVersion.cmake"
expect "$cmakedir/cbkTargets.cmake"
expect "$prefix/include/cbk.h"
if [ "$shared" = ON ]; then
    expect "$libdir/libcbk.so"
    expect_absent "$libdir/libcbk.a"
else
    expect "$libdir/libcbk.a"
    expect_absent "$libdir/libcbk.so"
fi
for f in include/cbk_compat.h include/cbk_mkl_alloc.h "${cmakedir#"$prefix/"}/FindMKLCompact.cmake"; do
    if [ "$mkl" = ON ]; then expect "$prefix/$f"; else expect_absent "$prefix/$f"; fi
done

# A shared libcbk exports its C entry points and nothing else (the kernels'
# template instantiations stay hidden), and its soname carries the minor
# version until 1.0.0, when a minor release may still change the ABI.
if [ "$shared" = ON ]; then
    version=$(sed -n 's/^set(PACKAGE_VERSION "\(.*\)")$/\1/p' "$cmakedir/cbkConfigVersion.cmake")
    major=${version%%.*}
    minor=${version#*.}; minor=${minor%%.*}
    if [ "$major" = 0 ]; then want="libcbk.so.$major.$minor"; else want="libcbk.so.$major"; fi
    soname=$(objdump -p "$libdir/libcbk.so" | awk '$1 == "SONAME" { print $2 }')
    if [ "$soname" = "$want" ]; then
        echo "ok       soname $soname"
    else
        echo "SONAME   $soname (want $want, version $version)"; fail=1
    fi
    exported=$(nm -D --defined-only "$libdir/libcbk.so" | awk '{ print $NF }')
    stray=$(echo "$exported" | grep -v -E '^(cbk_)?[sd][a-z_]+_compact$|^cbk_get_version$' || true)
    if [ -z "$stray" ] && [ -n "$exported" ]; then
        echo "ok       exports: $(echo "$exported" | wc -l) entry points, nothing else"
    else
        echo "EXPORTS  beyond the public API:"; echo "$stray" | head -n 20; fail=1
    fi
fi

if [ "$fail" -ne 0 ]; then
    echo "== full listing"
    find "$prefix" -type f -o -type l | sort
    echo "install tree check FAILED"
    exit 1
fi

echo "== downstream project against the install"
consumer=${CBK_CONSUMER_BUILD_DIR:-$root/build-consumer}
cmake -S "$root/tests/install" -B "$consumer" -DCMAKE_PREFIX_PATH="$prefix" -DCBK_WITH_MKL="$mkl"
cmake --build "$consumer"
"$consumer/consumer"
echo "install check passed"
