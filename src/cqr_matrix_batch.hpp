/* cqr_matrix_batch.hpp
 *
 * MatrixBatch<T, Alloc>: an owning batch of `count` dense column-major
 * rows x cols matrices in one contiguous buffer -- matrix idx starts at
 * idx * rows * cols with leading dimension rows. The owning companion of
 * MatrixView (cqr_matrix_view.hpp): the batch holds the storage, the view is
 * how one matrix inside it is addressed.
 *
 * Shared by the test suites and the benchmarks (internal, like the views:
 * src/ is on their include path, the public API stays include/). The
 * allocator parameter is what separates the two uses: the tests default it,
 * the benchmarks substitute a 64-byte-aligned one so a dense pool and its
 * LAPACK working copies start pack-aligned like the compact buffers (see
 * "Buffer alignment" in AGENTS.md).
 *
 * Assisted-by: Claude
 */

#ifndef CQR_MATRIX_BATCH_HPP
#define CQR_MATRIX_BATCH_HPP

#include "cqr_matrix_view.hpp"

#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

namespace cqr::detail {

template <typename T, typename Alloc = std::allocator<T>> class MatrixBatch {
  public:
    using vector_type = std::vector<T, Alloc>;

    MatrixBatch(int count, int rows, int cols)
        : count_(count), rows_(rows), cols_(cols), a_((std::size_t)count * rows * cols)
    {
    }

    // clang-format off
    int count() const { return count_; }
    int rows()  const { return rows_; }
    int cols()  const { return cols_; }
    /* Scalars per matrix: matrix idx occupies [idx*stride(), (idx+1)*stride()). */
    std::size_t stride() const { return (std::size_t)rows_ * cols_; }
    T       *data()       { return a_.data(); }
    const T *data() const { return a_.data(); }
    T       *operator[](int idx)       { return a_.data() + checked(idx) * stride(); }
    const T *operator[](int idx) const { return a_.data() + checked(idx) * stride(); }
    MatrixView<T>       view(int idx)       { return mat_view((*this)[idx], rows_, cols_); }
    ConstMatrixView<T>  view(int idx) const { return mat_view((*this)[idx], rows_, cols_); }
    T       &operator()(int idx, int i, int j)       { return a_[at(idx, i, j)]; }
    const T &operator()(int idx, int i, int j) const { return a_[at(idx, i, j)]; }
    // clang-format on

    /* Matrix idx read in the given layout, with its natural leading dimension
     * (rows for column-major, cols for row-major) -- for batches whose dense
     * matrices are stored in a runtime layout, so call sites stop deriving
     * the ld by hand. */
    MatrixView<T> view(int idx, bool rowmajor)
    {
        return mat_view((*this)[idx], rows_, cols_, rowmajor ? cols_ : rows_, rowmajor);
    }
    ConstMatrixView<T> view(int idx, bool rowmajor) const
    {
        return mat_view((*this)[idx], rows_, cols_, rowmajor ? cols_ : rows_, rowmajor);
    }

    /* The whole buffer, for wholesale copies into a destroyable working pool. */
    const vector_type &storage() const { return a_; }

    /* One base pointer per matrix -- the array form the MKL pack/unpack
     * routines take (the const overload is the packing-input one). */
    std::vector<T *> base_ptrs()
    {
        std::vector<T *> p(count_);
        for (int v = 0; v < count_; ++v)
            p[v] = (*this)[v];
        return p;
    }
    std::vector<const T *> base_ptrs() const
    {
        std::vector<const T *> p(count_);
        for (int v = 0; v < count_; ++v)
            p[v] = (*this)[v];
        return p;
    }

  private:
    /* idx must name a matrix in the batch -- the batch-level analogue of the
     * view's (i, j) bounds assert, and every accessor above routes through it. */
    std::size_t checked(int idx) const
    {
        assert(idx >= 0 && idx < count_);
        return (std::size_t)idx;
    }

    /* Element (idx, i, j), fully bounds-checked: the same address view(idx)(i, j)
     * yields, without constructing a view per element access. */
    std::size_t at(int idx, int i, int j) const
    {
        assert(i >= 0 && i < rows_ && j >= 0 && j < cols_);
        return checked(idx) * stride() + (std::size_t)i + (std::size_t)j * rows_;
    }

    int count_, rows_, cols_;
    vector_type a_;
};

} /* namespace cqr::detail */

#endif /* CQR_MATRIX_BATCH_HPP */
