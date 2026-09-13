# Documentation

## Guides

| Page | Contents |
|------|----------|
| [building.md](building.md) | Prerequisites, locating MKL, the workspace contract under a threaded MKL, performance builds, CMake options. |
| [threading.md](threading.md) | How the routines thread their loop over groups, and how they compose with a caller's parallel loop. |
| [layout.md](layout.md) | The source tree: the public headers, the kernels, the test and benchmark helpers. |
| [examples.md](examples.md) | The worked solve and the benchmarks, and their CTest registration. |
| [related_work.md](related_work.md) | Other batched / interleaved small-matrix linear algebra libraries and interfaces. |

Also:

* [`examples/BENCHMARKS.md`](../examples/BENCHMARKS.md) - the benchmarks in
  detail, with measured results.
* [`PLANS.md`](../PLANS.md) - the status of each routine against its design
  document, plus open items.
* [`.claude/CLAUDE.md`](../.claude/CLAUDE.md) - contributor notes: formatting,
  linting, and the conventions the kernels follow.
* [`.claude/mkl-install.md`](../.claude/mkl-install.md) - installing Intel MKL
  and pointing the build at it.
* [`.claude/mkl-compact-behavior.md`](../.claude/mkl-compact-behavior.md) -
  the measured behavior of MKL's own compact routines (argument dereferencing,
  workspace use, threading).

## Design documents

One per routine, each covering the algorithm, the compact-format mapping, the
SIMD kernel, validation, and the scoped-out extensions:

| Routine | Document |
|---------|----------|
| `?geqrf_compact` - QR factorization | [cqr_mkl_dgeqrf_compact_design.md](cqr_mkl_dgeqrf_compact_design.md) |
| `?ormqr_compact` - apply `Q` / `Q^T` | [cqr_mkl_dormqr_compact_design.md](cqr_mkl_dormqr_compact_design.md) |
| `?orgqr_compact` - form `Q` | [cqr_mkl_dorgqr_compact_design.md](cqr_mkl_dorgqr_compact_design.md) |
| `?potrf_compact` / `?potrs_compact` / `?posv_compact` - Cholesky | [cqr_mkl_dpotrf_compact_design.md](cqr_mkl_dpotrf_compact_design.md) |
| `?sytrfnp_compact` / `?sytrsnp_compact` / `?sysvnp_compact` - unpivoted LDL^T | [cqr_mkl_dsytrfnp_compact_design.md](cqr_mkl_dsytrfnp_compact_design.md) |
| `?trsm_compact` - triangular solve | [cqr_mkl_dtrsm_compact_design.md](cqr_mkl_dtrsm_compact_design.md) |
| `?gels_compact` - least-squares / minimum-norm solve | [cqr_mkl_dgels_compact_design.md](cqr_mkl_dgels_compact_design.md) |
