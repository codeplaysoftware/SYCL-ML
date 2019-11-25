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

#include "ml/utils/debug/assert.hpp"
#include "ml/utils/sycl_helper.hpp"
#include "ml/utils/sycl_types.hpp"

namespace ml {

class ml_memset;

/**
 * @brief Memset a specific range to a SYCL buffer.
 * @return A SYCL event corresponding to the submitted operation.
 */
template <class T, int DIM>
event sycl_memset(queue& q, buffer<T, DIM>& buffer, const nd_range<DIM>& r,
                  T val = T(0)) {
  assert_less_or_eq(r.get_offset()[0] + r.get_global_linear_range(),
                    buffer.get_count());
  return q.submit([&buffer, r, val](handler& cgh) {
    auto acc = buffer.template get_access<access::mode::write>(cgh);
    cgh.parallel_for<NameGen<DIM, ml_memset, T>>(
        r, [=](nd_item<DIM> item) { acc[item.get_global_linear_id()] = val; });
  });
}

/**
 * @brief Memset a whole SYCL buffer.
 * @return A SYCL event corresponding to the submitted operation.
 */
template <class T, int DIM>
event sycl_memset(queue& q, buffer<T, DIM>& buffer, T val = T(0)) {
  return q.submit([&buffer, val](handler& cgh) {
    auto acc = buffer.template get_access<access::mode::discard_write>(cgh);
    cgh.fill(acc, val);
  });
}

/**
 * @brief Copy a device buffer \p src to a device buffer \p dst.
 * @return A SYCL event corresponding to the submitted operation.
 */
template <class T, int DIM>
event sycl_copy(queue& q, buffer<T, DIM>& src, buffer<T, DIM>& dst) {
  assert_eq(src.get_count(), dst.get_count());

  return q.submit([&src, &dst](handler& cgh) {
    auto src_acc = src.template get_access<access::mode::read>(cgh);
    auto dst_acc = dst.template get_access<access::mode::discard_write>(cgh);
    cgh.copy(src_acc, dst_acc);
  });
}

/**
 * @brief Copy a device sub-buffer \p src to a device sub-buffer \p dst.
 * @return A SYCL event corresponding to the submitted operation.
 */
template <class T, int DIM>
event sycl_copy(queue& q, buffer<T, DIM>& src, buffer<T, DIM>& dst,
                size_t offset_src, size_t offset_dst, size_t count) {
  assert_less_or_eq(offset_src + count, src.get_count());
  assert_less_or_eq(offset_dst + count, dst.get_count());

  return q.submit([&src, &dst, offset_src, offset_dst, count](handler& cgh) {
    auto src_acc = src.template get_access<access::mode::read>(
        cgh, range<1>(count), id<1>(offset_src));
    auto dst_acc = dst.template get_access<access::mode::discard_write>(
        cgh, range<1>(count), id<1>(offset_dst));
    cgh.copy(src_acc, dst_acc);
  });
}

/**
 * @brief Copy a host buffer \p src to a device buffer \p dst.
 * The source pointer can only be accessed or destroyed after the returned
 * event is waited on.
 * @return A SYCL event corresponding to the submitted operation.
 */
template <class T, class SrcPtrT>
event sycl_copy_host_to_device(queue& q, SrcPtrT src, sycl_vec_t<T>& dst) {
  return q.submit([&dst, src](handler& cgh) {
    auto dst_acc = dst.template get_access<access::mode::discard_write>(cgh);
    cgh.copy(src, dst_acc);
  });
}

/**
 * @brief Copy a host buffer \p src to a device sub-buffer \p dst.
 * The source pointer can only be accessed or destroyed after the returned
 * event is waited on.
 * @return A SYCL event corresponding to the submitted operation.
 */
template <class T, class SrcPtrT>
event sycl_copy_host_to_device(queue& q, SrcPtrT src, sycl_vec_t<T>& dst,
                               size_t offset_dst, size_t count) {
  return q.submit([&dst, src, offset_dst, count](handler& cgh) {
    auto dst_acc = dst.template get_access<access::mode::discard_write>(
        cgh, range<1>(count), id<1>(offset_dst));
    cgh.copy(src, dst_acc);
  });
}

/**
 * @brief Copy a device buffer \p src to a host buffer \p dst.
 * The destination pointer can only be accessed or destroyed after the returned
 * event is waited on.
 * @return A SYCL event corresponding to the submitted operation.
 */
template <class T, class DstPtrT>
event sycl_copy_device_to_host(queue& q, sycl_vec_t<T>& src, DstPtrT dst) {
  return q.submit([&src, dst](handler& cgh) {
    auto src_acc = src.template get_access<access::mode::read>(cgh);
    cgh.copy(src_acc, dst);
  });
}

/**
 * @brief Copy a device sub-buffer \p src to a host buffer \p dst.
 * The destination pointer can only be accessed or destroyed after the returned
 * event is waited on.
 * @return A SYCL event corresponding to the submitted operation.
 */
template <class T, class DstPtrT>
event sycl_copy_device_to_host(queue& q, sycl_vec_t<T>& src, DstPtrT dst,
                               size_t offset_src, size_t count) {
  return q.submit([&src, dst, offset_src, count](handler& cgh) {
    auto src_acc = src.template get_access<access::mode::read>(
        cgh, range<1>(count), id<1>(offset_src));
    cgh.copy(src_acc, dst);
  });
}

class ml_init_func_i;

/**
 * @brief Initialize a SYCL buffer with a function depending on the index.
 * @return A SYCL event corresponding to the submitted operation.
 */
template <class Op, class T, int DIM>
event sycl_init_func_i(queue& q, buffer<T, DIM>& buffer, const nd_range<DIM>& r,
                       Op op = Op()) {
  assert_less_or_eq(r.get_offset()[0] + r.get_global_linear_range(),
                    buffer.get_count());
  return q.submit([&buffer, r, op](handler& cgh) {
    auto acc = buffer.template get_access<access::mode::discard_write>(cgh);
    cgh.parallel_for<NameGen<DIM, ml_init_func_i, T, Op>>(
        r, [=](nd_item<DIM> item) {
          auto idx = item.get_global_linear_id();
          acc[idx] = op(idx);
        });
  });
}

}  // namespace ml

#endif  // INCLUDE_ML_UTILS_COPY_HPP
