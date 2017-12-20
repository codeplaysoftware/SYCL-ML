/**
 * @file
 * @brief Common SYCL aliases
 */

#ifndef INCLUDE_ML_UTILS_SYCL_TYPES_HPP
#define INCLUDE_ML_UTILS_SYCL_TYPES_HPP

#include <CL/sycl.hpp>

#include "ml/eigen/eigen.hpp"

namespace ml
{

using namespace cl::sycl;

using SYCLIndexT = size_t;

template <class T, class Alloc = cl::sycl::default_allocator<T>>
using sycl_vec_t = buffer<T, 1, Alloc>;

/**
 * @brief Only buffer type that can be used with Eigen for now
 */
using buffer_data_type = cl::sycl::codeplay::buffer_data_type_t;

} // ml

#endif //INCLUDE_ML_UTILS_SYCL_TYPES_HPP
