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
