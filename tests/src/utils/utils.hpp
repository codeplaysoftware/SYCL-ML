#ifndef TEST_SRC_UTILS_UTILS_HPP
#define TEST_SRC_UTILS_UTILS_HPP

#include <random>

#include "sycl_utils.hpp"
#include "assert_utils.hpp"

template <class T, class Array>
void fill_random(Array& a, T min, T max) {
  std::generate(begin(a), end(a), [=]() {
    return (max - min) * (static_cast<T>(rand()) / RAND_MAX) + min;
  });
}

template <class T>
T compute_det(const std::array<T, 4>& d) {
  return d[0]*d[3] - d[2]*d[1];
}

#endif //TEST_SRC_UTILS_UTILS_HPP
