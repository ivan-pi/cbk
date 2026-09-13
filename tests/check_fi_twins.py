#!/usr/bin/env python3
"""Verify cqr_mkl_ext.fi and cqr_mkl_ext_ilp64.fi differ only where they
may: the header comment and the integer kinds of the MKL_INT dummies and
info (c_int vs c_long_long).

Both files are canonicalized -- comments dropped, dual-form continuations
(column-73 '&' carried on by column-6 '&') joined, whitespace squeezed,
c_long_long mapped to c_int, iso_c_binding only-lists and runs of
integer-value declarations compared as name sets -- and the resulting
statement sequences must be identical. Registered as the fortran_fi_twins
CTest; run by hand as tests/check_fi_twins.py [include-dir].

Assisted-by: Claude:claude-fable-5
"""
import re
import sys
from pathlib import Path

USE_RE = re.compile(r"^use, intrinsic :: iso_c_binding, only: (.+)$")
INT_RE = re.compile(r"^integer\(c_int\), value :: (.+)$")


def statements(path):
    """Comment-free, continuation-joined, whitespace-squeezed statements."""
    joined, current = [], ""
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("!"):
            continue
        if raw.startswith("     &"):  # continuation of the previous line
            current += " " + raw[6:].strip()
        else:
            if current:
                joined.append(current)
            current = raw.strip()
        if current.endswith("&"):  # column-73 continuation marker
            current = current[:-1].strip()
    if current:
        joined.append(current)
    return [re.sub(r"\s+", " ", s) for s in joined]


def canonical(path):
    out = []
    for stmt in statements(path):
        stmt = stmt.replace("c_long_long", "c_int")
        m = USE_RE.match(stmt)
        if m:
            names = sorted({n.strip() for n in m.group(1).split(",")})
            out.append("use only: " + ", ".join(names))
            continue
        m = INT_RE.match(stmt)
        if m:
            names = {n.strip() for n in m.group(1).split(",")}
            if out and out[-1].startswith("integer-value:"):
                names |= set(out.pop().split(": ")[1].split(", "))
            out.append("integer-value: " + ", ".join(sorted(names)))
            continue
        out.append(stmt)
    return out


def main():
    inc = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent.parent / "include"
    lp64 = canonical(inc / "cqr_mkl_ext.fi")
    ilp64 = canonical(inc / "cqr_mkl_ext_ilp64.fi")
    ok = True
    for i, (a, b) in enumerate(zip(lp64, ilp64)):
        if a != b:
            print(f"statement {i} differs:\n  lp64:  {a}\n  ilp64: {b}")
            ok = False
    if len(lp64) != len(ilp64):
        extra = lp64[len(ilp64):] or ilp64[len(lp64):]
        print(f"statement counts differ ({len(lp64)} vs {len(ilp64)}); "
              f"first unmatched: {extra[0]}")
        ok = False
    if not ok:
        sys.exit(1)
    print(f"cqr_mkl_ext.fi and cqr_mkl_ext_ilp64.fi agree "
          f"({len(lp64)} statements)")


if __name__ == "__main__":
    main()
