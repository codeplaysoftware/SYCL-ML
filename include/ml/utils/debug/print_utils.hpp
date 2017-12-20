/**
 * @file
 * @brief Allow to print generic array, std pair as well as sycl id, range and nd_range
 */

#ifndef INCLUDE_ML_UTILS_DEBUG_PRINT_UTILS_HPP
#define INCLUDE_ML_UTILS_DEBUG_PRINT_UTILS_HPP

#include <iostream>
#include <sstream>

#include "ml/utils/sycl_types.hpp"

namespace ml
{

/**
 * @brief Print std::pair
 *
 * @tparam T1
 * @tparam T2
 * @param os
 * @param p
 * @return os
 */
template <class T1, class T2>
std::ostream& operator<<(std::ostream& os, const std::pair<T1, T2>& p) {
  os << "(" << p.first << "," << p.second << ")";
  return os;
}

/**
 * @brief Print cl::sycl::id
 *
 * @tparam DIM
 * @param os
 * @param id_
 * @return os
 */
template <int DIM>
std::ostream& operator<<(std::ostream& os, const cl::sycl::id<DIM>& id_) {
  os << "[" << id_[0];
  for (int i = 1; i < DIM; ++i)
    os << ", " << id_[i];
  os << "]";
  return os;
}

/**
 * @brief Print cl::sycl::range
 *
 * @tparam DIM
 * @param os
 * @param r
 * @return os
 */
template <int DIM>
std::ostream& operator<<(std::ostream& os, const cl::sycl::range<DIM>& r) {
  os << "[" << r[0];
  for (int i = 1; i < DIM; ++i)
    os << ", " << r[i];
  os << "]";
  return os;
}

/**
 * @brief Print cl::sycl::nd_range
 *
 * @tparam DIM
 * @param os
 * @param r
 * @return os
 */
template <int DIM>
std::ostream& operator<<(std::ostream& os, const cl::sycl::nd_range<DIM>& r) {
  return os << r.get_global() << "@" << r.get_local() << "@" << r.get_offset();
}

/**
 * @brief Print any data array as a matrix
 *
 * @tparam T data type with a [] accessor
 * @param os
 * @param data
 * @param nrows
 * @param ncols
 * @param off
 * @return os
 */
template <class T>
std::ostream& print(std::ostream& os, const T& data, size_t nrows, size_t ncols, size_t off = 0) {
  for (size_t r = 0; r < nrows; ++r) {
    for (size_t c = 0; c < ncols; ++c) {
      os << data[r * ncols + c + off] << ' ';
    }
    os << std::endl;
  }
  return os;
}

/**
 * @brief Print any data array as a matrix
 *
 * @tparam T data type with a [] accessor
 * @param os
 * @param data
 * @param nrows
 * @param ncols
 * @param off
 * @return os
 */
template <class T>
std::ostream& print(const T& data, size_t nrows, size_t ncols, size_t off = 0) {
  return print(std::cout, data, nrows, ncols, off);
}

} // ml

#endif //INCLUDE_ML_UTILS_DEBUG_PRINT_UTILS_HPP
