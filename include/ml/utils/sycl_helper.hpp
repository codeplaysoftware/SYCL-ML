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
 * @brief helper for SYCLParallelSTL
 */

#ifndef INCLUDE_ML_UTILS_SYCL_HELPER_HPP
#define INCLUDE_ML_UTILS_SYCL_HELPER_HPP

#include <experimental/algorithm>
#include <sycl/execution_policy>
#include <sycl/helpers/sycl_buffers.hpp>
#include <sycl/helpers/sycl_namegen.hpp>

namespace ml
{

using namespace std::experimental::parallel;

using cl::sycl::helpers::NameGen;
using sycl::helpers::begin;
using sycl::helpers::end;

} // ml

#endif //INCLUDE_ML_UTILS_SYCL_HELPER_HPP
