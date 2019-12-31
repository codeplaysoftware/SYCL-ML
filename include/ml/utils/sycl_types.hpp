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
/**
 * @file
 * @brief Common SYCL aliases
 */

#ifndef INCLUDE_ML_UTILS_SYCL_TYPES_HPP
#define INCLUDE_ML_UTILS_SYCL_TYPES_HPP

#include <CL/sycl.hpp>

#include "ml/eigen/eigen.hpp"

namespace ml {

using namespace cl::sycl;

using SYCLIndexT = size_t;

template <class T, class Alloc = cl::sycl::default_allocator>
using sycl_vec_t = buffer<T, 1, Alloc>;

template <int Index, typename... Details>
class NameGen {};

}  // namespace ml

#endif  // INCLUDE_ML_UTILS_SYCL_TYPES_HPP
