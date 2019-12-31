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
 * @brief Regroup common headers to all files that submit SYCL kernels.
 */

#ifndef INCLUDE_ML_UTILS_COMMON_HPP
#define INCLUDE_ML_UTILS_COMMON_HPP

#include <cassert>

#include "ml/eigen/sycl_to_eigen.hpp"
#include "ml/utils/save_utils.hpp"

// Debug
#include "ml/utils/debug/assert.hpp"
#include "ml/utils/debug/print_utils.hpp"
#include "ml/utils/debug/write_bmp.hpp"

#endif  // INCLUDE_ML_UTILS_COMMON_HPP
