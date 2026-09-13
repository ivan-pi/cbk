# Threading

Every routine threads its loop over the *groups* of `V` interleaved matrices
with OpenMP (static schedule over a team of at most one thread per group; the
groups are independent and equal-sized), but only when the call has at least two
groups and enough work to pay for the fork/join -- about `5e4` flops, a
compromise between two measured break-evens (the `CBK_OMP_MIN_FLOPS`
preprocessor macro overrides it, e.g.
`-DCMAKE_CXX_FLAGS="-DCBK_OMP_MIN_FLOPS=1e5"`). Because the thread count is
OpenMP's team size *at the current nesting level*, the routines compose with a caller's own parallel loop: called
from inside it they run serially by default (no competing thread pools), and
the standard per-level thread list enables nested splitting when wanted, e.g.

```sh
OMP_NUM_THREADS=8,2 OMP_MAX_ACTIVE_LEVELS=2 ./my_app   # 8 outer x 2 inner threads
```

Results are independent of the thread count.

In the source, every all-groups driver is a call to
`for_each_group<V>(nm, flops_per_group, body)` in `src/cbk_common.hpp`;
the gate described above lives there, and no kernel threads inside a group.
`-DCBK_WITH_OPENMP=OFF` compiles the routines single-threaded.

The compact kernels of MKL itself thread internally only under MKL's threaded
layer (`-DMKLCompact_THREADING=threaded`), which the default build does not
link; see [building.md](building.md) for the workspace-size consequences of
that layer.
