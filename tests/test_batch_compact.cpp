// test_batch_compact.cpp
//
// The contracts of the batch as a whole, which no per-routine suite gates:
//
//   1. Lane isolation. A matrix the routine cannot handle -- a non-SPD one for
//      ?potrf/?posv, a zero pivot for ?sytrfnp/?sysvnp, a NaN entry for
//      ?geqrf, a rank-deficient one for ?gels -- gets Inf/NaN in its own
//      result and changes nothing else: the routine still returns 0, and
//      every other matrix of the batch, the ones sharing its pack included,
//      comes out bit for bit as it does from a batch without the bad matrix.
//      (The failure is not reported: cbk.h documents that there is no
//      info > 0. The caller's recourse is to test the result, which this
//      suite shows is a per-matrix test.)
//
//   2. Threading. Results do not depend on the number of threads (1, 2, 3 and
//      the runtime's maximum, on a batch large enough to pass the fork gate),
//      and the routines compose with a caller's own parallel loop: called
//      concurrently from the threads of an outer region, on separate batches,
//      with nested parallelism off (the inner loop then stays serial) and on.
//      All comparisons are bitwise against the one-thread result. Built
//      without OpenMP the routines are serial and this part says so and
//      passes.
//
// Both parts run the same operations: each is a closure from an input batch
// to an output batch through the portable C API.
//
// Assisted-by: Claude:claude-fable-5

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "test_compact_util.hpp" // compact<T>, generators, pack/unpack, finish

using namespace cbk::test;

// One operation of the suite: how its inputs are generated, how one matrix is
// made unusable for it, and the routine itself, input batch -> output batch
// (the factor, or the solution). run returns the routine's info through info.
template <class T> struct Op {
    std::string name;
    bool spd; /* SPD inputs (gen_spd), else general (gen_boosted) */
    std::function<void(MatrixView<T>)> spoil;
    std::function<MatrixBatch<T>(const MatrixBatch<T> &, int V, int &info)> run;
};

template <class T> static std::vector<Op<T>> make_ops(int n)
{
    static constexpr int nrhs = 2;
    /* one fixed right-hand side for every matrix of every batch */
    std::vector<T> rhs((size_t)n * nrhs);
    for (auto &x : rhs)
        x = frand<T>();
    const auto rhs_batch = [rhs, n](int nm) {
        MatrixBatch<T> B(nm, n, nrhs);
        for (int idx = 0; idx < nm; ++idx)
            std::copy(rhs.begin(), rhs.end(), B[idx]);
        return B;
    };

    const auto factor = [n](int (*f)(char, char, int, T *, int, int, int)) {
        return [f, n](const MatrixBatch<T> &A, int V, int &info) {
            std::vector<T> ap = pack_compact(A, n, V);
            info = f('C', 'L', n, ap.data(), n, V, A.count());
            MatrixBatch<T> out(A.count(), n, n);
            unpack_compact(out, ap.data(), n, V);
            return out;
        };
    };
    const auto solve =
        [n, rhs_batch](int (*f)(char, char, int, int, T *, int, T *, int, int, int)) {
            return [f, n, rhs_batch](const MatrixBatch<T> &A, int V, int &info) {
                std::vector<T> ap = pack_compact(A, n, V);
                std::vector<T> bp = pack_compact(rhs_batch(A.count()), n, V);
                info = f('C', 'L', n, nrhs, ap.data(), n, bp.data(), n, V, A.count());
                MatrixBatch<T> out(A.count(), n, nrhs);
                unpack_compact(out, bp.data(), n, V);
                return out;
            };
        };

    const auto negate_pivot = [](MatrixView<T> M) { M(2, 2) = -M(2, 2); };
    const auto zero_pivot = [](MatrixView<T> M) { M(0, 0) = T(0); };

    std::vector<Op<T>> ops;
    ops.push_back({"potrf", true, negate_pivot, factor(&compact<T>::potrf)});
    ops.push_back({"posv", true, negate_pivot, solve(&compact<T>::posv)});
    ops.push_back({"sytrfnp", true, zero_pivot, factor(&compact<T>::sytrfnp)});
    ops.push_back({"sysvnp", true, zero_pivot, solve(&compact<T>::sysvnp)});
    ops.push_back({"geqrf", false,
                   [](MatrixView<T> M) { M(1, 0) = std::numeric_limits<T>::quiet_NaN(); },
                   [n](const MatrixBatch<T> &A, int V, int &info) {
                       std::vector<T> ap = pack_compact(A, n, V);
                       std::vector<T> tp = compact_buffer<T>(A.count(), n, 1, n, V);
                       info = compact<T>::geqrf('C', n, n, ap.data(), n, tp.data(), V,
                                                A.count());
                       MatrixBatch<T> out(A.count(), n, n);
                       unpack_compact(out, ap.data(), n, V);
                       return out;
                   }});
    ops.push_back({"gels", false,
                   [](MatrixView<T> M) { /* a zero column: rank n - 1 */
                                         for (int i = 0; i < M.rows; ++i)
                                             M(i, 1) = T(0);
                   },
                   [n, rhs_batch](const MatrixBatch<T> &A, int V, int &info) {
                       std::vector<T> ap = pack_compact(A, n, V);
                       std::vector<T> bp = pack_compact(rhs_batch(A.count()), n, V);
                       std::vector<T> tp = compact_buffer<T>(A.count(), n, 1, n, V);
                       info = compact<T>::gels('C', 'N', n, n, nrhs, ap.data(), n,
                                               bp.data(), n, tp.data(), V, A.count());
                       MatrixBatch<T> out(A.count(), n, nrhs);
                       unpack_compact(out, bp.data(), n, V);
                       return out;
                   }});
    return ops;
}

template <class T> static MatrixBatch<T> make_inputs(const Op<T> &op, int nm, int n)
{
    MatrixBatch<T> A(nm, n, n);
    for (int idx = 0; idx < nm; ++idx) {
        if (op.spd)
            gen_spd(A.view(idx));
        else
            gen_boosted(A.view(idx));
    }
    return A;
}

template <class T> static bool same_bits(const MatrixBatch<T> &X, const MatrixBatch<T> &Y)
{
    return std::equal(X.data(), X.data() + (size_t)X.count() * X.stride(), Y.data());
}

// ------------------------------ lane isolation ------------------------

template <class T, int V> static int test_isolation(int n)
{
    const int nm = 2 * V + 1; /* two full groups and a padded one */
    int fails = 0;
    for (const Op<T> &op : make_ops<T>(n)) {
        const MatrixBatch<T> A = make_inputs(op, nm, n);
        int info_good = -1, info_bad = -1;
        const MatrixBatch<T> good = op.run(A, V, info_good);

        /* each position in turn: first and last lane of a full group, a lane
         * inside one, and the lone live lane of the padded group */
        int others_differ = 0, bad_finite = 0, good_nonfinite = 0;
        for (int b : {0, V - 1, V + 1, nm - 1}) {
            MatrixBatch<T> Ab = A;
            op.spoil(Ab.view(b));
            const MatrixBatch<T> out = op.run(Ab, V, info_bad);
            bool nonfinite = false;
            for (std::size_t e = 0; e < out.stride(); ++e)
                nonfinite = nonfinite || !std::isfinite(out[b][e]);
            bad_finite += !nonfinite;
            for (int idx = 0; idx < nm; ++idx) {
                if (idx == b) continue;
                others_differ +=
                    !std::equal(out[idx], out[idx] + out.stride(), good[idx]);
            }
            fails += (info_bad != 0);
        }
        for (std::size_t e = 0; e < (size_t)nm * good.stride(); ++e)
            good_nonfinite += !std::isfinite(good.data()[e]);

        const bool ok = !others_differ && !bad_finite && !good_nonfinite && !info_good;
        std::printf("T=%-6s V=%-2d nm=%-2d n=%-2d %-8s | bad matrix non-finite %s | "
                    "others bitwise %s | info %d | %s\n",
                    compact<T>::name, V, nm, n, op.name.c_str(),
                    bad_finite ? "NO" : "yes", others_differ ? "DIFFER" : "same",
                    info_good, ok ? "OK" : "FAIL");
        fails += !ok;
    }
    return fails;
}

// -------------------------------- threading ---------------------------

#ifdef _OPENMP
template <class T, int V> static int test_threading(int n)
{
    const int nm = 12 * V + 3; /* 13 groups: well past the fork gate at this n */
    const int max_threads = omp_get_max_threads();
    const int levels = omp_get_max_active_levels();
    int fails = 0;
    for (const Op<T> &op : make_ops<T>(n)) {
        const MatrixBatch<T> A = make_inputs(op, nm, n);
        int info = -1;

        omp_set_num_threads(1);
        const MatrixBatch<T> ref = op.run(A, V, info);
        int bad = (info != 0);

        /* the thread count */
        int counts = 0;
        for (int t : {2, 3, std::max(4, max_threads)}) {
            omp_set_num_threads(t);
            const MatrixBatch<T> out = op.run(A, V, info);
            counts += !same_bits(out, ref) || (info != 0);
        }

        /* the caller's own parallel loop, nested parallelism off then on */
        int nested[2] = {0, 0};
        for (int on = 0; on < 2; ++on) {
            omp_set_max_active_levels(on ? 2 : 1);
            omp_set_num_threads(on ? 2 : 4);
            int differ = 0;
#pragma omp parallel for num_threads(4) reduction(+ : differ)
            for (int rep = 0; rep < 8; ++rep) {
                int info_rep = -1;
                const MatrixBatch<T> out = op.run(A, V, info_rep);
                differ += !same_bits(out, ref) || (info_rep != 0);
            }
            nested[on] = differ;
        }
        omp_set_max_active_levels(levels);
        omp_set_num_threads(max_threads);

        bad += counts + nested[0] + nested[1];
        std::printf("T=%-6s V=%-2d nm=%-3d n=%-2d %-8s | threads 2/3/%d %s | outer loop "
                    "%s | outer loop, nested %s | %s\n",
                    compact<T>::name, V, nm, n, op.name.c_str(), std::max(4, max_threads),
                    counts ? "DIFFER" : "same", nested[0] ? "DIFFER" : "same",
                    nested[1] ? "DIFFER" : "same", bad ? "FAIL" : "OK");
        fails += (bad != 0);
    }
    return fails;
}
#endif

// ------------------------------- main --------------------------------

int main()
{
    int fails = 0;

    fails += test_isolation<double, 2>(6);
    fails += test_isolation<double, 4>(6);
    fails += test_isolation<double, 8>(12); /* past the blocked panel width */
    fails += test_isolation<float, 4>(6);
    fails += test_isolation<float, 8>(12);
    fails += test_isolation<float, 16>(6);

#ifdef _OPENMP
    std::printf("\n-- threading (OpenMP, %d threads available) --\n",
                omp_get_max_threads());
    fails += test_threading<double, 4>(24);
    fails += test_threading<double, 8>(24);
    fails += test_threading<float, 8>(24);
    fails += test_threading<float, 16>(24);
#else
    std::printf("\n-- threading: this suite was built without OpenMP; skipped --\n");
#endif

    return finish(fails);
}
