/**
 * Copyright (C) Codeplay Software Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef INCLUDE_ML_UTILS_COPY_HPP
#define INCLUDE_ML_UTILS_COPY_HPP

#include "ml/utils/buffer_t.hpp"
#include "ml/utils/debug/assert.hpp"
#include "ml/utils/optimal_range.hpp"
#include "ml/utils/sycl_types.hpp"

namespace ml {

namespace detail {

template <data_dim D, access::mode write_mode, class T>
struct copy_vec_to_mat_impl {
  event operator()(queue& q, matrix_t<T>& matrix, vector_t<T>& vec,
                   const nd_range<1>& nd_rng, SYCLIndexT row) {
    auto rng = nd_rng.get_global_range();
    assert_less_or_eq(matrix.sub_buffer_offset.get(0) +
                          row * matrix.get_kernel_range()[1] + rng.size(),
                      matrix.get_kernel_size());
    assert_less_or_eq(rng.size() + vec.sub_buffer_offset.get(0),
                      vec.get_kernel_size());
    return q.submit([&matrix, &vec, rng, row](handler& cgh) {
      auto matrix_acc = matrix.template get_access<write_mode>(
          cgh, rng,
          id<1>(matrix.sub_buffer_offset.get(0) +
                row * matrix.get_kernel_range()[1]));
      auto vec_acc = vec.template get_access<access::mode::read>(
          cgh, rng, vec.sub_buffer_offset);
      cgh.copy(vec_acc, matrix_acc);
    });
  }
};

class ml_copy_vec_to_mat;

template <access::mode write_mode, class T>
struct copy_vec_to_mat_impl<COL, write_mode, T> {
  event operator()(queue& q, matrix_t<T>& matrix, vector_t<T>& vec,
                   const nd_range<1>& nd_rng, SYCLIndexT col) {
    auto rng = nd_rng.get_global_range();
    assert_less_or_eq(rng.size() + matrix.sub_buffer_offset.get(0),
                      matrix.get_kernel_size());
    assert_less_or_eq(rng.size() + vec.sub_buffer_offset.get(0),
                      vec.get_kernel_size());
    return q.submit([&matrix, &vec, col, nd_rng](handler& cgh) {
      auto matrix_acc = matrix.template get_access_2d<access::mode::write>(cgh);
      auto vec_acc = vec.template get_access_1d<access::mode::read>(cgh);
      using ker_name =
          NameGen<static_cast<int>(write_mode), ml_copy_vec_to_mat, T>;
      cgh.parallel_for<ker_name>(nd_rng, [=](nd_item<1> item) {
        auto id = item.get_global_id(0);
        matrix_acc(id, col) = vec_acc(id);
      });
    });
  }
};

template <data_dim D, access::mode write_mode, class T>
struct copy_mat_to_vec_impl {
  event operator()(queue& q, matrix_t<T>& matrix, vector_t<T>& vec,
                   const nd_range<1>& nd_rng, SYCLIndexT row) {
    auto rng = nd_rng.get_global_range();
    assert_less_or_eq(matrix.sub_buffer_offset.get(0) +
                          row * matrix.get_kernel_range()[1] + rng.size(),
                      matrix.get_kernel_size());
    assert_less_or_eq(rng.size() + vec.sub_buffer_offset.get(0),
                      vec.get_kernel_size());
    return q.submit([&matrix, &vec, rng, row](handler& cgh) {
      auto matrix_acc = matrix.template get_access<access::mode::read>(
          cgh, rng,
          id<1>(matrix.sub_buffer_offset.get(0) +
                row * matrix.get_kernel_range()[1]));
      auto vec_acc = vec.template get_access_1d<write_mode>(cgh);
      cgh.copy(matrix_acc, vec_acc.get());
    });
  }
};

class ml_copy_mat_to_vec;

template <access::mode write_mode, class T>
struct copy_mat_to_vec_impl<COL, write_mode, T> {
  event operator()(queue& q, matrix_t<T>& matrix, vector_t<T>& vec,
                   const nd_range<1>& nd_rng, SYCLIndexT col) {
    auto rng = nd_rng.get_global_range();
    assert_less_or_eq(rng.size() + matrix.sub_buffer_offset.get(0),
                      matrix.get_kernel_size());
    assert_less_or_eq(rng.size() + vec.sub_buffer_offset.get(0),
                      vec.get_kernel_size());
    return q.submit([&matrix, &vec, nd_rng, col](handler& cgh) {
      auto matrix_acc = matrix.template get_access_2d<access::mode::read>(cgh);
      auto vec_acc = vec.template get_access_1d<write_mode>(cgh);
      using ker_name =
          NameGen<static_cast<int>(write_mode), ml_copy_mat_to_vec, T>;
      cgh.parallel_for<ker_name>(nd_rng, [=](nd_item<1> item) {
        auto id = item.get_global_id(0);
        vec_acc(id) = matrix_acc(id, col);
      });
    });
  }
};

}  // namespace detail

class ml_memset;

/**
 * @brief Memset a specific range to a SYCL buffer.
 * @return A SYCL event corresponding to the submitted operation.
 */
template <class T, int DIM>
event sycl_memset(queue& q, buffer_t<T, DIM>& buffer, const nd_range<1>& r,
                  T val = T(0)) {
  assert_less_or_eq(r.get_offset()[0] + r.get_global_linear_range(),
                    buffer.sub_buffer_range.get(0));
  return q.submit([&buffer, r, val](handler& cgh) {
    auto acc = buffer.template get_access_1d<access::mode::write>(cgh);
    cgh.parallel_for<NameGen<DIM, ml_memset, T>>(
        r, [=](nd_item<1> item) { acc(item.get_global_linear_id()) = val; });
  });
}

/**
 * @brief Memset a whole SYCL buffer.
 * @return A SYCL event corresponding to the submitted operation.
 */
template <class T, int DIM>
event sycl_memset(queue& q, buffer_t<T, DIM>& buffer, T val = T(0)) {
  return q.submit([&buffer, val](handler& cgh) {
    auto acc = buffer.template get_access_1d<access::mode::discard_write>(cgh);
    cgh.fill(acc.get(), val);
  });
}

/**
 * @brief Copy a device buffer \p src to a device buffer \p dst.
 * @return A SYCL event corresponding to the submitted operation.
 */
template <class T, int SrcDIM, int DstDim>
event sycl_copy(queue& q, buffer_t<T, SrcDIM>& src, buffer_t<T, DstDim>& dst) {
  assert_eq(src.sub_buffer_range.get(0), dst.sub_buffer_range.get(0));

  return q.submit([&src, &dst](handler& cgh) {
    auto src_acc = src.template get_access_1d<access::mode::read>(cgh);
    auto dst_acc = dst.template get_access_1d<access::mode::discard_write>(cgh);
    cgh.copy(src_acc.get(), dst_acc.get());
  });
}

/**
 * @brief Copy a device sub-buffer \p src to a device sub-buffer \p dst.
 * @return A SYCL event corresponding to the submitted operation.
 */
template <class T, int SrcDIM, int DstDim>
event sycl_copy(queue& q, buffer_t<T, SrcDIM>& src, buffer_t<T, DstDim>& dst,
                size_t offset_src, size_t offset_dst, size_t count) {
  assert_less_or_eq(offset_src + src.sub_buffer_offset.get(0) + count,
                    src.sub_buffer_range.get(0));
  assert_less_or_eq(offset_dst + dst.sub_buffer_offset.get(0) + count,
                    dst.sub_buffer_range.get(0));

  return q.submit([&src, &dst, offset_src, offset_dst, count](handler& cgh) {
    auto src_acc = src.template get_access<access::mode::read>(
        cgh, range<1>(count), id<1>(offset_src + src.sub_buffer_offset.get(0)));
    auto dst_acc = dst.template get_access<access::mode::discard_write>(
        cgh, range<1>(count), id<1>(offset_dst + dst.sub_buffer_offset.get(0)));
    cgh.copy(src_acc, dst_acc);
  });
}

/**
 * @brief Copy a host buffer \p src to a device buffer \p dst.
 * The source pointer can only be accessed or destroyed after the returned
 * event is waited on.
 * @return A SYCL event corresponding to the submitted operation.
 */
template <class T, int DIM, class SrcPtrT>
event sycl_copy_host_to_device(queue& q, SrcPtrT src, buffer_t<T, DIM>& dst) {
  return q.submit([&dst, src](handler& cgh) {
    auto dst_acc = dst.template get_access_1d<access::mode::discard_write>(cgh);
    cgh.copy(src, dst_acc.get());
  });
}

/**
 * @brief Copy a host buffer \p src to a device sub-buffer \p dst.
 * The source pointer can only be accessed or destroyed after the returned
 * event is waited on.
 * @return A SYCL event corresponding to the submitted operation.
 */
template <class T, int DIM, class SrcPtrT>
event sycl_copy_host_to_device(queue& q, SrcPtrT src, buffer_t<T, DIM>& dst,
                               size_t offset_dst, size_t count) {
  assert_less_or_eq(offset_dst + dst.sub_buffer_offset.get(0) + count,
                    dst.sub_buffer_range.get(0));

  return q.submit([&dst, src, offset_dst, count](handler& cgh) {
    auto dst_acc = dst.template get_access<access::mode::discard_write>(
        cgh, range<1>(count), id<1>(offset_dst + dst.sub_buffer_offset.get(0)));
    cgh.copy(src, dst_acc);
  });
}

/**
 * @brief Copy a device buffer \p src to a host buffer \p dst.
 * The destination pointer can only be accessed or destroyed after the returned
 * event is waited on.
 * @return A SYCL event corresponding to the submitted operation.
 */
template <class T, int DIM, class DstPtrT>
event sycl_copy_device_to_host(queue& q, buffer_t<T, DIM>& src, DstPtrT dst) {
  return q.submit([&src, dst](handler& cgh) {
    auto src_acc = src.template get_access_1d<access::mode::read>(cgh);
    cgh.copy(src_acc.get(), dst);
  });
}

/**
 * @brief Copy a device sub-buffer \p src to a host buffer \p dst.
 * The destination pointer can only be accessed or destroyed after the returned
 * event is waited on.
 * @return A SYCL event corresponding to the submitted operation.
 */
template <class T, int DIM, class DstPtrT>
event sycl_copy_device_to_host(queue& q, buffer_t<T, DIM>& src, DstPtrT dst,
                               size_t offset_src, size_t count) {
  assert_less_or_eq(offset_src + src.sub_buffer_offset.get(0) + count,
                    src.sub_buffer_range.get(0));

  return q.submit([&src, dst, offset_src, count](handler& cgh) {
    auto src_acc = src.template get_access<access::mode::read>(
        cgh, range<1>(count), id<1>(offset_src + src.sub_buffer_offset.get(0)));
    cgh.copy(src_acc, dst);
  });
}

class ml_init_func_i;

/**
 * @brief Initialize a SYCL buffer with a function depending on the index.
 * @return A SYCL event corresponding to the submitted operation.
 */
template <class Op, class T, int DIM>
event sycl_init_func_i(queue& q, buffer_t<T, DIM>& buffer,
                       const nd_range<DIM>& r, Op op = Op()) {
  assert_less_or_eq(r.get_offset()[0] + r.get_global_linear_range(),
                    buffer.sub_buffer_range.get(0));
  return q.submit([&buffer, r, op](handler& cgh) {
    auto acc = buffer.template get_access_1d<access::mode::discard_write>(cgh);
    cgh.parallel_for<NameGen<DIM, ml_init_func_i, T, Op>>(
        r, [=](nd_item<DIM> item) {
          auto idx = item.get_global_linear_id();
          acc(idx) = op(idx);
        });
  });
}

/**
 * @brief Copy a vector to a row (resp. a column) of a matrix.
 *
 * @tparam D row or col
 * @tparam T
 * @tparam write_mode write by default, can be discard_write if copying the
 * whole row
 * @param q
 * @param matrix destination buffer
 * @param vec source buffer
 * @param nd_rng range to copy
 * @param row_col which row (resp. col) to copy
 * @return A SYCL event corresponding to the submitted operation
 */
template <data_dim D = ROW, access::mode write_mode = access::mode::write,
          class T>
event copy_vec_to_mat(queue& q, matrix_t<T>& matrix, vector_t<T>& vec,
                      const nd_range<1>& nd_rng, SYCLIndexT row_col) {
  static_assert(write_mode == access::mode::write ||
                    write_mode == access::mode::discard_write,
                "Access mode must be either write or discard_write");
  assert_less_or_eq(row_col, access_ker_dim<D>(matrix, 0));
  assert_less_or_eq(nd_rng.get_global_range()[0], access_ker_dim<D>(matrix, 1));
  assert_less_or_eq(nd_rng.get_global_range()[0], vec.get_kernel_range()[0]);
  return detail::copy_vec_to_mat_impl<D, write_mode, T>()(q, matrix, vec,
                                                          nd_rng, row_col);
}

class ml_copy_mat_to_vec;

/**
 * @brief Copy a row (resp. a column) of a matrix to a vector.
 *
 * @tparam D row or col
 * @tparam T
 * @tparam write_mode write by default, can be discard_write if copying the
 * whole row
 * @param q
 * @param matrix destination buffer
 * @param vec source buffer
 * @param row_col which row (resp. col) to copy
 * @return A SYCL event corresponding to the last submitted operation
 */
template <data_dim D = ROW, access::mode write_mode = access::mode::write,
          class T>
event copy_mat_to_vec(queue& q, matrix_t<T>& matrix, vector_t<T>& vec,
                      const nd_range<1>& nd_rng, SYCLIndexT row_col) {
  static_assert(write_mode == access::mode::write ||
                    write_mode == access::mode::discard_write,
                "Access mode must be either write or discard_write");
  assert_less_or_eq(row_col, access_ker_dim<D>(matrix, 0));
  assert_less_or_eq(nd_rng.get_global_range()[0], access_ker_dim<D>(matrix, 1));
  assert_less_or_eq(nd_rng.get_global_range()[0], vec.get_kernel_range()[0]);
  return detail::copy_mat_to_vec_impl<D, write_mode, T>()(q, matrix, vec,
                                                          nd_rng, row_col);
}

class ml_split_by_index;

/**
 * @brief Select the rows specified by indices from buffer.
 *
 * @tparam DataT
 * @tparam IndexT
 * @param q
 * @param[in] buffer
 * @param[in] indices
 * @return split_buffer
 */
template <class DataT, class IndexT>
matrix_t<DataT> split_by_index(queue& q, matrix_t<DataT>& buffer,
                               vector_t<IndexT>& indices) {
  matrix_t<DataT> split_buffer(
      range<2>(indices.data_range[0], access_data_dim(buffer, 1)),
      get_optimal_nd_range(indices.get_kernel_range()[0],
                           access_ker_dim(buffer, 1)));

  q.submit([&indices, &buffer, &split_buffer](handler& cgh) {
    auto indices_acc = indices.template get_access_1d<access::mode::read>(cgh);
    auto buffer_acc = buffer.template get_access_2d<access::mode::read>(cgh);
    auto split_buffer_acc =
        split_buffer.template get_access_2d<access::mode::discard_write>(cgh);
    cgh.parallel_for<NameGen<2, ml_split_by_index, DataT, IndexT>>(
        split_buffer.get_nd_range(), [=](nd_item<2> item) {
          auto row = item.get_global_id(0);
          auto col = item.get_global_id(1);
          split_buffer_acc(row, col) = buffer_acc(indices_acc(row), col);
        });
  });

  return split_buffer;
}

/**
 * @brief Select the rows specified by indices from buffer.
 *
 * @tparam DataT
 * @tparam IndexT
 * @param q
 * @param[in] buffer
 * @param[in] indices
 * @return split_buffer
 */
template <class DataT, class IndexT>
vector_t<DataT> split_by_index(queue& q, vector_t<DataT>& buffer,
                               vector_t<IndexT>& indices) {
  vector_t<DataT> split_buffer(indices.data_range, indices.kernel_range);

  q.submit([&buffer, &indices, &split_buffer](handler& cgh) {
    auto indices_acc = indices.template get_access_1d<access::mode::read>(cgh);
    auto buffer_acc = buffer.template get_access_1d<access::mode::read>(cgh);
    auto split_buffer_acc =
        split_buffer.template get_access_1d<access::mode::discard_write>(cgh);
    cgh.parallel_for<NameGen<1, ml_split_by_index, DataT, IndexT>>(
        split_buffer.get_nd_range(), [=](nd_item<1> item) {
          auto row = item.get_global_id(0);
          split_buffer_acc(row) = buffer_acc(indices_acc(row));
        });
  });

  return split_buffer;
}

/**
 * @brief Select the rows specified by indices from buffer.
 *
 * @see split_by_index(queue&, vector_t<DataT>&, vector_t<IndexT>&)
 * @see split_by_index(queue&, matrix_t<DataT>&, vector_t<IndexT>&)
 * @tparam DataT
 * @tparam IndexT
 * @tparam DIM
 * @param q
 * @param buffer
 * @param host_indices
 * @return split_buffer
 */
template <class DataT, class IndexT, int DIM>
buffer_t<DataT, DIM> split_by_index(queue& q, buffer_t<DataT, DIM>& buffer,
                                    const std::vector<IndexT>& host_indices) {
  vector_t<IndexT> indices(host_indices.data(), range<1>(host_indices.size()));
  return split_by_index(q, buffer, indices);
}

}  // namespace ml

#endif  // INCLUDE_ML_UTILS_COPY_HPP
