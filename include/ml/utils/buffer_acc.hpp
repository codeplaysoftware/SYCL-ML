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
#ifndef INCLUDE_ML_UTILS_BUFFER_ACC_HPP
#define INCLUDE_ML_UTILS_BUFFER_ACC_HPP

#include "ml/utils/access.hpp"

#ifndef ML_DEBUG_BOUND_CHECK
/**
 * @brief Set to 1 for buffer initialization with nan and boundaries access
 * check.
 *
 * For debug only.
 * @warning Very slow.
 */
#define ML_DEBUG_BOUND_CHECK 0
#endif  // ML_DEBUG_BOUND_CHECK

namespace ml {

template <class T, int DIM>
class buffer_t;

namespace detail {

template <data_dim D>
struct get_index_2d;

template <>
struct get_index_2d<LIN> {
  static inline SYCLIndexT compute(SYCLIndexT r, SYCLIndexT c,
                                   SYCLIndexT nb_cols) {
    return r * nb_cols + c;
  }
};

template <>
struct get_index_2d<TR> {
  static inline SYCLIndexT compute(SYCLIndexT r, SYCLIndexT c,
                                   SYCLIndexT nb_cols) {
    return c * nb_cols + r;
  }
};

template <class T, access::mode>
struct is_reference_access {
  using value = T&;
};

template <class T>
struct is_reference_access<T, access::mode::read> {
  using value = T;
};

template <class T, int DIM, access::mode acc_mode, access::target acc_target>
class buffer_1d_acc_t {
 public:
  buffer_1d_acc_t(handler& cgh, buffer_t<T, DIM>* b)
      :
#if ML_DEBUG_BOUND_CHECK
        _range(b->get_kernel_size()),
#endif
        _offset(b->sub_buffer_offset),
        _acc(b->template get_access<acc_mode>(cgh, b->sub_buffer_range,
                                              b->sub_buffer_offset)) {
  }

  inline typename is_reference_access<T, acc_mode>::value operator()(
      SYCLIndexT x) const {
    x += _offset.get(0);
#if ML_DEBUG_BOUND_CHECK
    if (x >= _range[0]) {
      printf("Warning accessing at (%lu) from buffer of size (%lu)\n", x,
             _range[0]);
    }
#endif
    return _acc[x];
  }

  inline accessor<T, 1, acc_mode, acc_target> get() { return _acc; }

 private:
#if ML_DEBUG_BOUND_CHECK
  range<1> _range;
#endif
  id<1> _offset;
  accessor<T, 1, acc_mode, acc_target> _acc;
};

template <class T, access::mode acc_mode, data_dim D, access::target acc_target>
class buffer_2d_acc_t {
 public:
  buffer_2d_acc_t(handler& cgh, buffer_t<T, 2>* b)
      : _range(b->get_kernel_range()),
        _offset(b->sub_buffer_offset),
        _acc(b->template get_access<acc_mode>(cgh, b->sub_buffer_range,
                                              b->sub_buffer_offset)) {}

  inline typename is_reference_access<T, acc_mode>::value operator()(
      SYCLIndexT r, SYCLIndexT c) const {
    auto idx =
        _offset.get(0) + detail::get_index_2d<D>::compute(r, c, _range[1]);
#if ML_DEBUG_BOUND_CHECK
    if (idx >= _range.size()) {
      printf(
          "Warning accessing at (%lu, %lu)+%lu from buffer of size (%lu, "
          "%lu)\n",
          r, c, _offset.get(0), access_rng<D>(_range, 0),
          access_rng<D>(_range, 1));
    }
#endif
    return _acc[idx];
  }

  inline accessor<T, 1, acc_mode, acc_target> get() { return _acc; }

 private:
  range<2> _range;
  id<1> _offset;
  accessor<T, 1, acc_mode, acc_target> _acc;
};

template <class T, access::mode acc_mode, access::target acc_target>
class buffer_3d_acc_t {
 public:
  buffer_3d_acc_t(handler& cgh, buffer_t<T, 3>* b)
      : _range(b->get_kernel_range()),
        _acc(b->template get_access<acc_mode>(cgh, b->sub_buffer_range,
                                              b->sub_buffer_offset)) {}

  inline typename is_reference_access<T, acc_mode>::value operator()(
      SYCLIndexT x, SYCLIndexT y, SYCLIndexT z) const {
    auto idx = _offset.get(0) + x + _range[1] * (y + _range[2] * z);
#if ML_DEBUG_BOUND_CHECK
    if (idx >= _range.size()) {
      printf(
          "Warning accessing at (%lu, %lu, %lu)+%lu from buffer of size (%lu, "
          "%lu, "
          "%lu)\n",
          x, y, z, _offset.get(0), _range[0], _range[1], _range[2]);
    }
#endif
    return _acc[idx];
  }

  inline accessor<T, 1, acc_mode, acc_target> get() { return _acc; }

 private:
  range<3> _range;
  id<1> _offset;
  accessor<T, 1, acc_mode, acc_target> _acc;
};

}  // namespace detail

}  // namespace ml

#endif  // INCLUDE_ML_UTILS_BUFFER_ACC_HPP
