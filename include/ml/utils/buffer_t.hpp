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

#include "ml/utils/optimal_range.hpp"
#include "ml/utils/copy.hpp"
#include "ml/utils/debug/print_utils.hpp"

#ifndef ML_DEBUG_BOUND_CHECK
/**
 * @brief Set to 1 for buffer initialization with nan and boundaries access check.
 *
 * For debug only.
 * @warning Very slow.
 */
#define ML_DEBUG_BOUND_CHECK 0
#endif  // ML_DEBUG_BOUND_CHECK

namespace ml
{

namespace detail
{

// Forward declare buffer_nd_acc_t
template <class, access::mode, access::target>
class buffer_1d_acc_t;
template <class, access::mode, data_dim, access::target>
class buffer_2d_acc_t;
template <class, access::mode, access::target>
class buffer_3d_acc_t;

// Specialize get_1d_kernel_nd_range
template <int DIM>
inline nd_range<1> get_1d_kernel_nd_range(const nd_range<DIM>& nd_rng) {
  return get_optimal_nd_range(nd_rng.get_global_range().size());
}

template <>
inline nd_range<1> get_1d_kernel_nd_range<1>(const nd_range<1>& nd_rng) { return nd_rng; }

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

} // detail

// Custom wrapper around SYCL buffer to have a distinction between allocated size and actual data size
/**
 * @brief Custom wrapper around a SYCL buffer.
 *
 * Allow to have a distinction between the data size (\p data_range) and the padded size (\p kernel_range).
 * Also provides a custom accessor which simplifies the access to multi-dimensions buffers and can enable debug check.
 *
 * @tparam T data type
 * @tparam DIM dimension in [1, 3]. The underlying SYCL buffer is always of dimension 1.
 */
template <class T, int DIM>
class buffer_t : public sycl_vec_t<T> {
public:
  buffer_t(const sycl_vec_t<T>& b, const range<DIM>& d, const nd_range<DIM>& k) :
      sycl_vec_t<T>(b), data_range(d), kernel_range(k) {
    assert_rng_less_or_eq(data_range, kernel_range.get_global_range());
    assert_rng_size_less_or_eq(kernel_range.get_global_range(), b.get_count());
  }

  buffer_t(const sycl_vec_t<T>& b, const range<DIM>& r) : buffer_t(b, r, get_optimal_nd_range(r)) {}
  buffer_t(const sycl_vec_t<T>& b) : buffer_t(b, b.get_range()) {}

  buffer_t(const range<DIM>& r, const nd_range<DIM>& k) :
      buffer_t(sycl_vec_t<T>(range<1>(k.get_global_range().size())), r, k) {
#if ML_DEBUG_BOUND_CHECK
    if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
      T init = std::numeric_limits<T>::quiet_NaN();
      sycl_memset(get_eigen_device().sycl_queue(), *this, get_1d_kernel_nd_range(), init);
    }
#endif
  }

  buffer_t(const nd_range<DIM>& k) : buffer_t(k.get_global_range(), k) {}
  buffer_t(const range<DIM>& r) : buffer_t(get_optimal_nd_range(r)) {}

  template <class HostPtrT>
  buffer_t(HostPtrT host_ptr, range<DIM> r) : buffer_t(sycl_vec_t<T>(host_ptr, r.size()), r) {}
  buffer_t(std::unique_ptr<T[]>&& host_ptr, range<DIM> r) : buffer_t(sycl_vec_t<T>(std::move(host_ptr), r.size()), r) {}

  buffer_t() : buffer_t(range<DIM>(), nd_range<DIM>(range<DIM>(), range<DIM>())) {}

  /**
   * @brief Return a continuous sub-buffer.
   *
   * No copy is made and the parent buffer must stay alive at least as long as the sub-buffer.
   *
   * @tparam OUT_DIM
   * @param offset
   * @param data_range
   * @param kernel_range its global range must meet the base address alignment requirement
   * @return a continuous sub-buffer
   */
  template <int OUT_DIM>
  buffer_t<T, OUT_DIM> get_sub_buffer(const id<1>& offset, const range<OUT_DIM>& data_range,
                                      const nd_range<OUT_DIM>& kernel_range) {
    //TODO: Use sub-buffer instead of copying
    //auto sub_buf = sycl_vec_t<T>(*this, offset, range<1>(kernel_range.get_global_linear_range()));
    auto size = cl::sycl::range<1>(kernel_range.get_global_linear_range());
    auto sub_buf = sycl_vec_t<T>(size);
    get_eigen_device().sycl_queue().submit([&](cl::sycl::handler &cgh) {
      auto src_acc = this->template get_access<cl::sycl::access::mode::read>(cgh, size, offset);
      auto dst_acc = sub_buf.template get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.copy(src_acc, dst_acc);
    });
    return buffer_t<T, OUT_DIM>(sub_buf, data_range, kernel_range);
  }

  /**
   * @brief Return a continuous sub-buffer.
   *
   * @see get_sub_buffer(const id<OUT_DIM>&, const range<OUT_DIM>&, const nd_range<OUT_DIM>&)
   * @tparam OUT_DIM
   * @param offset
   * @param kernel_range
   * @return a continuous sub-buffer
   */
  template <int OUT_DIM>
  inline buffer_t<T, OUT_DIM> get_sub_buffer(const id<OUT_DIM>& offset, const nd_range<OUT_DIM>& kernel_range) {
    return get_sub_buffer(offset, kernel_range.get_global_range(), kernel_range);
  }

  /**
   * @brief Return a specific row of a matrix as a sub-buffer.
   *
   * Note that the size (in bits) of a row must be divisible by the base address alignment.
   *
   * @see get_sub_buffer(const id<OUT_DIM>&, const range<OUT_DIM>&, const nd_range<OUT_DIM>&)
   * @param row_idx row to return
   * @return a continuous sub-buffer
   */
  buffer_t<T, 1> get_row(SYCLIndexT row_idx) {
    static_assert(DIM == 2, "Buffer must be of dimension 2");

    auto ker_row_size = get_kernel_range()[1];
    assert(ker_row_size % get_device_constants()->get_sub_buffer_range_divisor<T>() == 0);
    return get_sub_buffer<1>(id<1>(row_idx * ker_row_size), range<1>(data_range[1]),
                             get_optimal_nd_range(ker_row_size));
  }

  // Custom accessors for n dimensions allowing compile time transpose
  template <access::mode acc_mode, access::target acc_target = access::target::global_buffer>
  inline detail::buffer_1d_acc_t<T, acc_mode, acc_target> get_access_1d(handler& cgh) {
    return detail::buffer_1d_acc_t<T, acc_mode, acc_target>(cgh, *this);
  }

  template <access::mode acc_mode, data_dim D = LIN, access::target acc_target = access::target::global_buffer>
  inline detail::buffer_2d_acc_t<T, acc_mode, D, acc_target> get_access_2d(handler& cgh) {
    return detail::buffer_2d_acc_t<T, acc_mode, D, acc_target>(cgh, *this);
  }

  template <access::mode acc_mode, access::target acc_target = access::target::global_buffer>
  inline detail::buffer_3d_acc_t<T, acc_mode, acc_target> get_access_3d(handler& cgh) {
    return detail::buffer_3d_acc_t<T, acc_mode, acc_target>(cgh, *this);
  }

  // Host accessor helper
  inline T read_to_host(SYCLIndexT i) {
    return this->template get_access<access::mode::read>(range<1>(1), id<1>(i))[i];
  }

  inline void write_from_host(SYCLIndexT i, T val) {
    this->template get_access<access::mode::write>(range<1>(1), id<1>(i))[i] = val;
  }

  // Getters
  inline SYCLIndexT data_dim_size() const { return data_range.size(); }

  inline range<DIM> get_kernel_range() const { return kernel_range.get_global_range(); }

  inline nd_range<1> get_1d_kernel_nd_range() const {
    return detail::get_1d_kernel_nd_range(kernel_range);
  }

  template <data_dim D = LIN>
  inline nd_range<DIM> get_nd_range() const { return detail::get_nd_range<D>(kernel_range); }

  inline void assert_data_eq_ker() { assert_rng_eq(data_range, get_kernel_range()); }
  inline void assert_square() { assert_rng_square(get_kernel_range()); }

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
inline SYCLIndexT access_mem_dim(const matrix_t<T>& m, SYCLIndexT i) { return access_rng<D>(m.get_range(), i); }
template <data_dim D = LIN, class T>
inline SYCLIndexT access_data_dim(const matrix_t<T>& m, SYCLIndexT i) { return access_rng<D>(m.data_range, i); }
template <data_dim D = LIN, class T>
inline SYCLIndexT access_ker_dim(const matrix_t<T>& m, SYCLIndexT i) { return access_rng<D>(m.get_kernel_range(), i); }

namespace detail
{

template <data_dim D>
struct get_index_2d;

template <>
struct get_index_2d<LIN> {
  static inline SYCLIndexT compute(SYCLIndexT r, SYCLIndexT c, SYCLIndexT nb_cols) { return r * nb_cols + c; }
};

template <>
struct get_index_2d<TR> {
  static inline SYCLIndexT compute(SYCLIndexT r, SYCLIndexT c, SYCLIndexT nb_cols) { return c * nb_cols + r; }
};

template <class T, access::mode>
struct is_reference_access {
  using value = T&;
};

template <class T>
struct is_reference_access<T, access::mode::read> {
  using value = T;
};

template <class T, access::mode acc_mode, access::target acc_target>
class buffer_1d_acc_t {
public:
  buffer_1d_acc_t(handler& cgh, buffer_t<T, 1>& b) :
      _range(b.get_kernel_range()),
      _acc(b.template get_access<acc_mode>(cgh)) {}

  inline typename is_reference_access<T, acc_mode>::value operator()(SYCLIndexT x) const {
#if ML_DEBUG_BOUND_CHECK
    if (x >= _range[0])
      printf("Warning accessing at (%lu) from buffer of size (%lu)\n", x, _range[0]);
#endif
    return _acc[x];
  }

private:
  range<1> _range;
  accessor<T, 1, acc_mode, acc_target> _acc;
};

template <class T, access::mode acc_mode, data_dim D, access::target acc_target>
class buffer_2d_acc_t {
public:
  buffer_2d_acc_t(handler& cgh, buffer_t<T, 2>& b) :
      _range(b.get_kernel_range()),
      _acc(b.template get_access<acc_mode>(cgh)) {}

  inline typename is_reference_access<T, acc_mode>::value operator()(SYCLIndexT r, SYCLIndexT c) const {
#if ML_DEBUG_BOUND_CHECK
    if (r >= access_rng<D>(_range, 0) || c >= access_rng<D>(_range, 1))
      printf("Warning accessing at (%lu, %lu) from buffer of size (%lu, %lu)\n",
             r, c, access_rng<D>(_range, 0), access_rng<D>(_range, 1));
#endif
    return _acc[detail::get_index_2d<D>::compute(r, c, _range[1])];
  }

private:
  range<2> _range;
  accessor<T, 1, acc_mode, acc_target> _acc;
};

template <class T, access::mode acc_mode, access::target acc_target>
class buffer_3d_acc_t {
public:
  buffer_3d_acc_t(handler& cgh, buffer_t<T, 3>& b) :
      _range(b.get_kernel_range()),
      _acc(b.template get_access<acc_mode>(cgh)) {}

  inline typename is_reference_access<T, acc_mode>::value operator()(SYCLIndexT x, SYCLIndexT y, SYCLIndexT z) const {
#if ML_DEBUG_BOUND_CHECK
    if (x >= _range[0] || y >= _range[1] || z >= _range[2])
      printf("Warning accessing at (%lu, %lu, %lu) from buffer of size (%lu, %lu, %lu)\n",
             x, y, z, _range[0], _range[1], _range[2]);
#endif
    return _acc[x + _range[1] * (y + _range[2] * z)];
  }

private:
  range<3> _range;
  accessor<T, 1, acc_mode, acc_target> _acc;
};

template <data_dim D, access::mode write_mode, class T>
struct copy_vec_to_mat_impl {
  void operator()(queue& q, matrix_t<T>& matrix, vector_t<T>& vec,
                  const nd_range<1>& nd_rng, SYCLIndexT row) {
    auto rng = nd_rng.get_global_range();
    q.submit([&](handler& cgh) {
      auto matrix_acc = matrix.template get_access<write_mode>(cgh,
          rng, id<1>(row * matrix.get_kernel_range()[1]));
      auto vec_acc = vec.template get_access<access::mode::read>(cgh, rng, id<1>(0));
      cgh.copy(vec_acc, matrix_acc);
    });
  }
};

class ml_copy_vec_to_mat;

template <access::mode write_mode, class T>
struct copy_vec_to_mat_impl<COL, write_mode, T> {
  void operator()(queue& q, matrix_t<T>& matrix, vector_t<T>& vec,
                  const nd_range<1>& nd_rng, SYCLIndexT col) {
    q.submit([&](handler& cgh) {
      auto matrix_acc = matrix.template get_access_2d<access::mode::write>(cgh);
      auto vec_acc = vec.template get_access_1d<access::mode::read>(cgh);
      using ker_name = NameGen<static_cast<int>(write_mode), ml_copy_vec_to_mat, T>;
      cgh.parallel_for<ker_name>(nd_rng, [=](nd_item<1> item) {
        auto id = item.get_global_id(0);
        matrix_acc(id, col) = vec_acc(id);
      });
    });
  }
};

template <data_dim D, access::mode write_mode, class T>
struct copy_mat_to_vec_impl {
  void operator()(queue& q, matrix_t<T>& matrix, vector_t<T>& vec,
                  const nd_range<1>& nd_rng, SYCLIndexT row) {
    auto rng = nd_rng.get_global_range();
    q.submit([&](handler& cgh) {
      auto matrix_acc = matrix.template get_access<access::mode::read>(cgh,
          rng, id<1>(row * matrix.get_kernel_range()[1]));
      auto vec_acc = vec.template get_access<write_mode>(cgh, rng, id<1>(0));
      cgh.copy(matrix_acc, vec_acc);
    });
  }
};

class ml_copy_mat_to_vec;

template <access::mode write_mode, class T>
struct copy_mat_to_vec_impl<COL, write_mode, T> {
  void operator()(queue& q, matrix_t<T>& matrix, vector_t<T>& vec,
                  const nd_range<1>& nd_rng, SYCLIndexT col) {
    q.submit([&](handler& cgh) {
      auto matrix_acc = matrix.template get_access_2d<access::mode::read>(cgh);
      auto vec_acc = vec.template get_access_1d<write_mode>(cgh);
      using ker_name = NameGen<static_cast<int>(write_mode), ml_copy_mat_to_vec, T>;
      cgh.parallel_for<ker_name>(nd_rng, [=](nd_item<1> item) {
        auto id = item.get_global_id(0);
        vec_acc(id) = matrix_acc(id, col);
      });
    });
  }
};


} // detail


/**
 * @brief Copy a vector to a row (resp. a column) of a matrix.
 *
 * @tparam D row or col
 * @tparam T
 * @tparam write_mode write by default, can be discard_write if copying the whole row
 * @param q
 * @param matrix destination buffer
 * @param vec source buffer
 * @param nd_rng range to copy
 * @param row_col which row (resp. col) to copy
 */
template <data_dim D = ROW, access::mode write_mode = access::mode::write, class T>
void copy_vec_to_mat(queue& q, matrix_t<T>& matrix, vector_t<T>& vec,
                     const nd_range<1>& nd_rng, SYCLIndexT row_col) {
  static_assert(write_mode == access::mode::write || write_mode == access::mode::discard_write,
                "Access mode must be either write or discard_write");
  assert_less_or_eq(row_col, access_ker_dim<D>(matrix, 0));
  assert_less_or_eq(nd_rng.get_global_range()[0], access_ker_dim<D>(matrix, 1));
  assert_less_or_eq(nd_rng.get_global_range()[0], vec.get_kernel_range()[0]);
  detail::copy_vec_to_mat_impl<D, write_mode, T>()(q, matrix, vec, nd_rng, row_col);
}

class ml_copy_mat_to_vec;

/**
 * @brief Copy a row (resp. a column) of a matrix to a vector.
 *
 * @tparam D row or col
 * @tparam T
 * @tparam write_mode write by default, can be discard_write if copying the whole row
 * @param q
 * @param matrix destination buffer
 * @param vec source buffer
 * @param row_col which row (resp. col) to copy
 */
template <data_dim D = ROW, access::mode write_mode = access::mode::write, class T>
void copy_mat_to_vec(queue& q, matrix_t<T>& matrix, vector_t<T>& vec,
                     const nd_range<1>& nd_rng,  SYCLIndexT row_col) {
  static_assert(write_mode == access::mode::write || write_mode == access::mode::discard_write,
                "Access mode must be either write or discard_write");
  assert_less_or_eq(row_col, access_ker_dim<D>(matrix, 0));
  assert_less_or_eq(nd_rng.get_global_range()[0], access_ker_dim<D>(matrix, 1));
  assert_less_or_eq(nd_rng.get_global_range()[0], vec.get_kernel_range()[0]);
  detail::copy_mat_to_vec_impl<D, write_mode, T>()(q, matrix, vec, nd_rng, row_col);
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
matrix_t<DataT> split_by_index(queue& q, matrix_t<DataT>& buffer, vector_t<IndexT>& indices) {
  matrix_t<DataT> split_buffer(range<2>(indices.data_range[0], access_data_dim(buffer, 1)),
                               get_optimal_nd_range(indices.get_kernel_range()[0], access_ker_dim(buffer, 1)));

  q.submit([&](handler& cgh) {
    auto indices_acc = indices.template get_access_1d<access::mode::read>(cgh);
    auto buffer_acc = buffer.template get_access_2d<access::mode::read>(cgh);
    auto split_buffer_acc = split_buffer.template get_access_2d<access::mode::discard_write>(cgh);
    cgh.parallel_for<NameGen<2, ml_split_by_index, DataT, IndexT>>(split_buffer.get_nd_range(), [=](nd_item<2> item) {
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
vector_t<DataT> split_by_index(queue& q, vector_t<DataT>& buffer, vector_t<IndexT>& indices) {
  vector_t<DataT> split_buffer(indices.data_range, indices.kernel_range);

  q.submit([&](handler& cgh) {
    auto indices_acc = indices.template get_access_1d<access::mode::read>(cgh);
    auto buffer_acc = buffer.template get_access_1d<access::mode::read>(cgh);
    auto split_buffer_acc = split_buffer.template get_access_1d<access::mode::discard_write>(cgh);
    cgh.parallel_for<NameGen<1, ml_split_by_index, DataT, IndexT>>(split_buffer.get_nd_range(), [=](nd_item<1> item) {
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
buffer_t<DataT, DIM> split_by_index(queue& q, buffer_t<DataT, DIM>& buffer, const std::vector<IndexT>& host_indices) {
  vector_t<IndexT> indices(host_indices.data(), range<1>(host_indices.size()));
  return split_by_index(q, buffer, indices);
}

// Print
template <class T>
std::ostream& operator<<(std::ostream& os, vector_t<T>& v) {
  return print(os, v.template get_access<access::mode::read>(), 1, v.get_count());
}

template <class T>
std::ostream& operator<<(std::ostream& os, matrix_t<T>& m) {
  return print(os, m.template get_access<access::mode::read>(), m.data_range[0], m.data_range[1]);
}

template <class T>
std::ostream& operator<<(std::ostream& os, matrices_t<T>& ms) {
  auto offset = ms.data_range[0] * ms.data_range[1];
  range<1> rng(offset);
  std::string sep(20, '-');
  os << sep << '\n';
  auto ms_host = ms.template get_access<access::mode::read>();
  for (SYCLIndexT i = 0; i < ms.data_range[2]; ++i) {
    print(os, ms_host, ms.data_range[0], ms.data_range[1], i * offset);
    os << sep << '\n';
  }
  return os;
}

} // ml

#endif //INCLUDE_ML_UTILS_BUFFER_T_HPP
