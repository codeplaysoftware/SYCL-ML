/**
 * @file
 * @brief Regroup common headers to all files that submit SYCL kernels.
 */

#ifndef INCLUDE_ML_UTILS_COMMON_HPP
#define INCLUDE_ML_UTILS_COMMON_HPP

#include <cassert>

#include "ml/utils/sycl_helper.hpp"
#include "ml/utils/save_utils.hpp"
#include "ml/eigen/sycl_to_eigen.hpp"

// Debug
#include "ml/utils/debug/assert.hpp"
#include "ml/utils/debug/print_utils.hpp"
#include "ml/utils/debug/write_bmp.hpp"

#endif //INCLUDE_ML_UTILS_COMMON_HPP
