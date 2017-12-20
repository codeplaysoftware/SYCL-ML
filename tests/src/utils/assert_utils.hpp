#ifndef TEST_SRC_UTILS_ASSERT_UTILS_HPP
#define TEST_SRC_UTILS_ASSERT_UTILS_HPP

#include <cmath>

// Should be included last
#undef NDEBUG
#include <cassert>

#define EPS 1E-5

template <class T>
void assert_eq(T actual, T expected) {
  if (actual != expected) {
    std::cerr << "Error: got " << actual << " expected " << expected << std::endl;
    assert(false);
  }
}

template <class T>
void assert_almost_eq(T actual, T expected, const T eps = EPS) {
  if (std::fabs(actual - expected) > eps) {
    std::cerr << "Error: got " << actual << " expected " << expected << std::endl;
    assert(false);
  }
}

template <class T>
void assert_vec_almost_eq(const T* actual, const T* expected, size_t size, const T eps = EPS) {
  for (size_t i = 0; i < size; ++i)
    assert_almost_eq(actual[i], expected[i], eps);
}

template <class T, size_t DIM>
void assert_vec_almost_eq(const std::array<T, DIM>& actual, const std::array<T, DIM>& expected, const T eps = EPS) {
  assert_vec_almost_eq(actual.data(), expected.data(), DIM, eps);
}

template <class T, int DIM>
void assert_vector_almost_eq_no_direction(const T* actual, const T* expected, const T eps = EPS) {
  T norm_pos = 0;
  T norm_neg = 0;
  for (unsigned i = 0; i < DIM; ++i) {
    T diff = actual[i] - expected[i];
    T sum = actual[i] + expected[i];
    norm_pos += diff * diff;
    norm_neg += sum * sum;
  }
  T norm = std::min(norm_neg, norm_pos);
  assert_almost_eq(norm, 0.0f, eps);
}

#endif //TEST_SRC_UTILS_ASSERT_UTILS_HPP
