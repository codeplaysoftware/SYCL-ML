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
#ifndef TEST_SRC_UTILS_UTILS_HPP
#define TEST_SRC_UTILS_UTILS_HPP

#include <random>

#include "assert_utils.hpp"
#include "sycl_utils.hpp"

template <class T, class Array>
void fill_random(Array& a, T min, T max) {
  std::generate(begin(a), end(a), [=]() {
    return (max - min) * (static_cast<T>(rand()) / RAND_MAX) + min;
  });
}

template <class T>
T compute_det(const std::array<T, 4>& d) {
  return d[0] * d[3] - d[2] * d[1];
}

#endif  // TEST_SRC_UTILS_UTILS_HPP
