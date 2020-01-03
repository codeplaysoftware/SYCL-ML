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
#ifndef INCLUDE_ML_EIGEN_SYCL_TO_EIGEN_HPP
#define INCLUDE_ML_EIGEN_SYCL_TO_EIGEN_HPP

#include <memory>

#include "ml/utils/access.hpp"
#include "ml/utils/buffer_t.hpp"

namespace ml {

namespace detail {

template <int IN_DIM, int OUT_DIM = IN_DIM>
eig_dsize_t<OUT_DIM> range_to_dsize(const range<IN_DIM>& r) {
  static_assert(IN_DIM <= OUT_DIM, "");

  eig_dsize_t<OUT_DIM> dim;
  int i = 0;
  for (; i < IN_DIM; ++i) {
    dim[i] = static_cast<eig_index_t>(r[i]);
  }
  for (; i < OUT_DIM; ++i) {
    dim[i] = 1;
  }
  return dim;
}

template <>
eig_dsize_t<0> range_to_dsize<1, 0>(const range<1>&) {
  return {};
}

}  // namespace detail

/**
 * @brief Convert a SYCL buffer to an Eigen Tensor.
 *
 * The class holds the host pointer and makes sure that the Tensor is destroyed
 * at the end.\n Thus this object must stay alive as long as the Tensor is used.
 *
 * @todo Because of the way Eigen works if 2 \p sycl_to_eigen_t objects are
 * created with the same buffer and one is destroyed, the 2 Tensors become
 * invalid. The fix would require to either count the number of references for
 * each buffer or to create a different pointer if one already exist.
 *
 * @tparam T
 * @tparam IN_DIM dimension of the SYCL buffer
 * @tparam OUT_DIM dimension of the Eigen Tensor
 * @tparam DataLayout Eigen::RowMajor or Eigen::ColMajor
 */
template <class T, int IN_DIM, int OUT_DIM = IN_DIM,
          Eigen::StorageOptions DataLayout = Eigen::RowMajor>
class sycl_to_eigen_t {
 private:
  using Self = sycl_to_eigen_t<T, IN_DIM, OUT_DIM, DataLayout>;

 public:
  sycl_to_eigen_t() = default;

  sycl_to_eigen_t(buffer_t<T, IN_DIM>& b, const eig_dsize_t<OUT_DIM>& sizes) {
    auto reinterpret_buffer =
        b.template reinterpret<Eigen::TensorSycl::internal::buffer_data_type_t>(
            cl::sycl::range<1>(b.get_count() * sizeof(T)));
    _host_ptr =
        static_cast<T*>(get_eigen_device().attach_buffer(reinterpret_buffer)) +
        b.sub_buffer_offset.get(0);
    _tensor = std::make_unique<tensor_map_t<T, OUT_DIM, DataLayout>>(_host_ptr,
                                                                     sizes);
  }

  ~sycl_to_eigen_t() {
    if (_host_ptr) {
      get_eigen_device().detach_buffer(_host_ptr);
    }
  }

  /**
   * @return the Eigen Tensor
   */
  inline auto& tensor() { return *_tensor; }

  /**
   * @return the Eigen TensorDevice (for assignment)
   */
  inline auto device() { return tensor().device(get_eigen_device()); }

  inline const T* ptr() const { return _host_ptr; }

  // No copy, only move
  sycl_to_eigen_t(const Self&) = delete;
  sycl_to_eigen_t(Self&&) = default;
  Self& operator=(const Self&) = delete;
  Self& operator=(Self&&) = default;

 private:
  T* _host_ptr;
  std::unique_ptr<tensor_map_t<T, OUT_DIM, DataLayout>> _tensor;
};

/**
 * @brief Create a Tensor of dimension 0 from a SYCL buffer.
 *
 * Only the first value of the buffer is used.
 *
 * @tparam IN_DIM
 * @tparam DataLayout
 * @tparam T
 * @param b
 * @return the \p sycl_to_eigen_t associated to \p b
 */
template <int IN_DIM, Eigen::StorageOptions DataLayout = Eigen::RowMajor,
          class T>
inline auto sycl_to_scalar_eigen(buffer_t<T, IN_DIM>& b) {
  assert_less_or_eq(1LU, b.get_kernel_size());
  return sycl_to_eigen_t<T, IN_DIM, 0, DataLayout>(b, eig_dsize_t<0>());
}

/**
 * @brief Create a Tensor of any dimensions from a SYCL buffer.
 *
 * @tparam IN_DIM dimension of the input buffer
 * @tparam OUT_DIM dimension of the output Tensor
 * @tparam R_DIM dimension of the range
 * @tparam DataLayout
 * @tparam T
 * @param b
 * @param r range defining the size of the tensor
 * @return the \p sycl_to_eigen_t associated to \p b
 */
template <int IN_DIM, int OUT_DIM = IN_DIM,
          Eigen::StorageOptions DataLayout = Eigen::RowMajor, int R_DIM,
          class T>
inline auto sycl_to_eigen(buffer_t<T, IN_DIM>& b, const range<R_DIM>& r) {
  static_assert(
      R_DIM >= IN_DIM && (R_DIM <= OUT_DIM || (R_DIM == 1 && OUT_DIM == 0)),
      "");
  assert_less_or_eq(r.size(), b.get_kernel_size());
  return sycl_to_eigen_t<T, IN_DIM, OUT_DIM, DataLayout>(
      b, detail::range_to_dsize<R_DIM, OUT_DIM>(r));
}

/// @see sycl_to_eigen(buffer_t<T, IN_DIM>&, const range<IN_DIM>&)
template <int IN_DIM, int OUT_DIM = IN_DIM,
          Eigen::StorageOptions DataLayout = Eigen::RowMajor, class T>
inline auto sycl_to_eigen(buffer_t<T, IN_DIM>& b) {
  return sycl_to_eigen<IN_DIM, OUT_DIM, DataLayout>(b, b.get_kernel_range());
}

/**
 * @brief Force a buffer of dimension 1 to be converted to a Tensor of
 * dimension 2.
 *
 * @tparam D whether to build the Tensor as a column (by default) or a row.
 * @tparam DataLayout
 * @tparam T
 * @param v
 * @return the \p sycl_to_eigen_t associated to \p b
 */
template <data_dim D = COL, Eigen::StorageOptions DataLayout = Eigen::RowMajor,
          class T>
inline auto sycl_to_eigen_2d(vector_t<T>& v) {
  return sycl_to_eigen<1, 2, DataLayout>(
      v, build_lin_or_tr<opp<D>(), range<2>>(v.get_kernel_range()[0], 1));
}

}  // namespace ml

#endif  // INCLUDE_ML_EIGEN_SYCL_TO_EIGEN_HPP
