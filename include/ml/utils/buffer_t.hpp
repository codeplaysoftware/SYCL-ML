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
#ifndef INCLUDE_ML_UTILS_BUFFER_T_HPP
#define INCLUDE_ML_UTILS_BUFFER_T_HPP

#include <limits>
#include <type_traits>

#include "ml/utils/buffer_acc.hpp"
#include "ml/utils/debug/assert.hpp"
#include "ml/utils/debug/print_utils.hpp"
#include "ml/utils/optimal_range.hpp"

namespace ml {

namespace detail {

// Specialize get_nd_range
template <data_dim D, int DIM>
inline nd_range<DIM> get_nd_range(const nd_range<DIM>& kernel_range) {
  return kernel_range;
}

template <>
inline nd_range<2> get_nd_range<TR>(const nd_range<2>& kernel_range) {
  return nd_range<2>(build_lin_or_tr<TR>(kernel_range.get_global_range()),
                     build_lin_or_tr<TR>(kernel_range.get_local_range()),
                     build_lin_or_tr<TR>(kernel_range.get_offset()));
}

}  // namespace detail

#if ML_DEBUG_BOUND_CHECK
template <class T, int DIM>
event sycl_memset(queue&, buffer_t<T, DIM>&, const nd_range<1>&, T val);
#endif  // ML_DEBUG_BOUND_CHECK

// Custom wrapper around SYCL buffer to have a distinction between allocated
// size and actual data size
/**
 * @brief Custom wrapper around a SYCL buffer.
 *
 * Allow to have a distinction between the data size (\p data_range) and the
 * padded size (\p kernel_range). Also provides a custom accessor which
 * simplifies the access to multi-dimensions buffers and can enable debug check.
 *
 * @tparam T data type
 * @tparam DIM dimension in [1, 3]. The underlying SYCL buffer is always of
 * dimension 1.
 */
template <class T, int DIM>
class buffer_t : public sycl_vec_t<T> {
 public:
  buffer_t(const sycl_vec_t<T>& b, const range<DIM>& d, const nd_range<DIM>& k)
      : sycl_vec_t<T>(b),
        sub_buffer_offset(),
        sub_buffer_range(k.get_global_linear_range()),
        data_range(d),
        kernel_range(k) {
    assert_rng_less_or_eq(data_range, kernel_range.get_global_range());
    assert_rng_size_less_or_eq(kernel_range.get_global_range(), b.get_count());
  }

  buffer_t(const sycl_vec_t<T>& b, const range<DIM>& r)
      : buffer_t(b, r, get_optimal_nd_range(r)) {}
  buffer_t(const sycl_vec_t<T>& b) : buffer_t(b, b.get_range()) {}

  buffer_t(const range<DIM>& r, const nd_range<DIM>& k)
      : buffer_t(sycl_vec_t<T>(range<1>(k.get_global_range().size())), r, k) {
#if ML_DEBUG_BOUND_CHECK
    if (k.get_global_range().size() > 0 &&
        (std::is_same<T, float>::value || std::is_same<T, double>::value)) {
      T init = std::numeric_limits<T>::quiet_NaN();
      sycl_memset(get_eigen_device().sycl_queue(), *this,
                  get_optimal_nd_range(get_kernel_size()), init);
    }
#endif
  }

  buffer_t(const nd_range<DIM>& k) : buffer_t(k.get_global_range(), k) {}
  buffer_t(const range<DIM>& r) : buffer_t(get_optimal_nd_range(r)) {}

  template <class HostPtrT>
  buffer_t(HostPtrT host_ptr, range<DIM> r)
      : buffer_t(sycl_vec_t<T>(host_ptr, r.size()), r) {}

  buffer_t()
      : sycl_vec_t<T>(),
        data_range(),
        kernel_range(range<DIM>(), range<DIM>()) {}

  buffer_t(const buffer_t&) = default;
  buffer_t(buffer_t&&) = default;
  buffer_t& operator=(const buffer_t&) = default;
  buffer_t& operator=(buffer_t&&) = default;

  /**
   * @brief Return a continuous sub-buffer.
   *
   * This fills the sub_buffer_offset and sub_buffer_range members in order
   * to get ranged accessors with the get_access_*d methods.
   * This does not affect get_access from the cl::sycl::buffer class.
   *
   * @param offset
   * @param data_range
   * @param kernel_range
   * @return a continuous sub-buffer
   */
  buffer_t<T, 1> get_sub_buffer(const id<1>& offset, const range<1>& data_range,
                                const nd_range<1>& kernel_range) {
    buffer_t<T, 1> sub_buffer(*this, data_range, kernel_range);
    sub_buffer.sub_buffer_offset =
        id<1>(sub_buffer_offset.get(0) + offset.get(0));
    sub_buffer.sub_buffer_range = kernel_range.get_global_range();

    assert_less_or_eq(data_range.get(0),
                      kernel_range.get_global_range().get(0));
    assert_less_or_eq(sub_buffer.sub_buffer_offset.get(0) +
                          sub_buffer.sub_buffer_range.get(0),
                      sub_buffer.get_count());
    return sub_buffer;
  }

  /**
   * @brief Return a continuous sub-buffer.
   *
   * @see get_sub_buffer(const id<1>&, const range<1>&, const nd_range<1>&)
   * @param offset
   * @param kernel_range
   * @return a continuous sub-buffer
   */
  inline buffer_t<T, 1> get_sub_buffer(const id<1>& offset,
                                       const nd_range<1>& kernel_range) {
    return get_sub_buffer(offset, kernel_range.get_global_range(),
                          kernel_range);
  }

  /**
   * @brief Return a specific row of a matrix as a sub-buffer.
   *
   * @see get_sub_buffer(const id<1>&, const range<1>&, const nd_range<1>&)
   * @param row_idx row to return
   * @return a continuous sub-buffer
   */
  buffer_t<T, 1> get_row(SYCLIndexT row_idx) {
    static_assert(DIM == 2, "Buffer must be of dimension 2");

    auto ker_row_size = get_kernel_range()[1];
    return get_sub_buffer(id<1>(row_idx * ker_row_size),
                          range<1>(data_range[1]),
                          get_optimal_nd_range(ker_row_size));
  }

  // Custom accessors for n dimensions allowing compile time transpose
  template <access::mode acc_mode,
            access::target acc_target = access::target::global_buffer>
  inline detail::buffer_1d_acc_t<T, DIM, acc_mode, acc_target> get_access_1d(
      handler& cgh) {
    return detail::buffer_1d_acc_t<T, DIM, acc_mode, acc_target>(cgh, this);
  }

  template <access::mode acc_mode, data_dim D = LIN,
            access::target acc_target = access::target::global_buffer>
  inline detail::buffer_2d_acc_t<T, acc_mode, D, acc_target> get_access_2d(
      handler& cgh) {
    return detail::buffer_2d_acc_t<T, acc_mode, D, acc_target>(cgh, this);
  }

  template <access::mode acc_mode,
            access::target acc_target = access::target::global_buffer>
  inline detail::buffer_3d_acc_t<T, acc_mode, acc_target> get_access_3d(
      handler& cgh) {
    return detail::buffer_3d_acc_t<T, acc_mode, acc_target>(cgh, this);
  }

  // Host accessor helper
  inline T read_to_host(SYCLIndexT i) {
    T res;
    auto event =
        sycl_copy_device_to_host(get_eigen_device().sycl_queue(), *this, &res,
                                 sub_buffer_offset.get(0) + i, 1);
    event.wait_and_throw();
    return res;
  }

  inline void write_from_host(SYCLIndexT i, T val) {
    auto event =
        sycl_copy_host_to_device(get_eigen_device().sycl_queue(), &val, *this,
                                 sub_buffer_offset.get(0) + i, 1);
    event.wait_and_throw();
  }

  // Getters
  inline SYCLIndexT data_dim_size() const { return data_range.size(); }

  template <data_dim D = LIN>
  inline range<DIM> get_kernel_range() const {
    return get_nd_range<D>().get_global_range();
  }

  template <data_dim D = LIN>
  inline nd_range<DIM> get_nd_range() const {
    return detail::get_nd_range<D>(kernel_range);
  }

  inline size_t get_kernel_size() const { return get_kernel_range().size(); }

  inline void assert_data_eq_ker() {
    assert_rng_eq(data_range, get_kernel_range());
  }
  inline void assert_square() { assert_rng_square(get_kernel_range()); }

  id<1> sub_buffer_offset;
  range<1> sub_buffer_range;
  range<DIM> data_range;
  nd_range<DIM> kernel_range;
};

template <class T>
using vector_t = buffer_t<T, 1>;

template <class T>
using matrix_t = buffer_t<T, 2>;

template <class T>
using matrices_t = buffer_t<T, 3>;

template <data_dim D = LIN, class T>
inline SYCLIndexT access_data_dim(const matrix_t<T>& m, SYCLIndexT i) {
  return access_rng<D>(m.data_range, i);
}

template <data_dim D = LIN, class T>
inline SYCLIndexT access_ker_dim(const matrix_t<T>& m, SYCLIndexT i) {
  return access_rng<D>(m.get_kernel_range(), i);
}

// Print
template <class T>
std::ostream& operator<<(std::ostream& os, vector_t<T>& v) {
  return print(os, v.template get_access<access::mode::read>(), 1,
               v.data_range[0]);
}

template <class T>
std::ostream& operator<<(std::ostream& os, matrix_t<T>& m) {
  return print(os, m.template get_access<access::mode::read>(), m.data_range[0],
               m.data_range[1]);
}

template <class T>
std::ostream& operator<<(std::ostream& os, matrices_t<T>& ms) {
  auto offset = ms.data_range[0] * ms.data_range[1];
  std::string sep(20, '-');
  os << sep << '\n';
  auto ms_host = ms.template get_access<access::mode::read>();
  for (SYCLIndexT i = 0; i < ms.data_range[2]; ++i) {
    print(os, ms_host, ms.data_range[0], ms.data_range[1], i * offset);
    os << sep << '\n';
  }
  return os;
}

}  // namespace ml

#endif  // INCLUDE_ML_UTILS_BUFFER_T_HPP
