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
#ifndef INCLUDE_ML_UTILS_OPTIMAL_RANGE_HPP
#define INCLUDE_ML_UTILS_OPTIMAL_RANGE_HPP

#include <array>

#include "ml/utils/access.hpp"
#include "ml/utils/device_constants.hpp"

namespace ml {

/**
 * @tparam T
 * @param x
 * @return true if x is a power of 2
 */
template <class T>
inline bool is_pow2(T x) {
  return (x & (x - 1)) == 0;
}

/**
 * @tparam T
 * @param x
 * @return the closest power of 2 higher or equal to x
 */
template <class T>
inline T to_pow2(T x) {
  return std::pow(2, std::ceil(std::log2(x)));
}

/**
 * @brief Compute the best suitable local_range associated to global_range.
 *
 * The function is trivial if the global range is smaller or equal to the max
 * work group size.\n If not the function only tries to find divisor that are
 * power of 2. Finding all possible divisors would be too costly otherwise.
 *
 * @tparam DIM
 * @param global_range
 * @return local_range
 */
template <int DIM>
range<DIM> get_optimal_local_range(const range<DIM>& global_range) {
  auto max_work_group_size = get_device_constants()->get_max_work_group_size();
  range<DIM> local_range;
  if (global_range.size() <= max_work_group_size) {
    local_range = global_range;
  } else {
    auto max_work_group_item_sizes =
        get_device_constants()->get_max_work_item_sizes();
    for (int i = 0; i < DIM; ++i) {
      local_range[i] = max_work_group_item_sizes[i];
      while (global_range[i] % local_range[i]) {
        local_range[i] >>= 1;
      }
    }

    // Make sure the local size does not exceed the maximum
    for (int i = 0; i < DIM && local_range.size() > max_work_group_size; ++i) {
      // Try to divide the ith local size to reach a size of max_work_group_size
      auto divide_by = local_range.size() / max_work_group_size;
      local_range[i] /= std::min(local_range[i], divide_by);
    }
  }

  return local_range;
}

/**
 * @see get_optimal_local_range
 * @tparam DIM
 * @param global_range
 * @param offset
 * @return the nd_range built from \p global_range with a local range as big as
 * possible
 */
template <int DIM>
inline nd_range<DIM> get_optimal_nd_range(const range<DIM>& global_range,
                                          const id<DIM>& offset = id<DIM>()) {
  return nd_range<DIM>(global_range, get_optimal_local_range(global_range),
                       offset);
}

/**
 * @see get_optimal_nd_range(const range<DIM>&, const id<DIM>&)
 * @tparam Args
 * @param args
 * @return the nd_range built from \p args with a local range as big as possible
 */
template <class... Args>
inline nd_range<sizeof...(Args)> get_optimal_nd_range(Args... args) {
  return get_optimal_nd_range(range<sizeof...(Args)>(args...));
}

}  // namespace ml

#endif  // INCLUDE_ML_UTILS_OPTIMAL_RANGE_HPP
