#include <iostream>
#include <random>

#include "ml/math/mat_ops.hpp"
#include "utils/utils.hpp"

template <class T>
void test_lin_tr_inplace_mat_op() {
  std::cout << "\nTest transposed inplace matrix operation\n";

  static constexpr size_t M = 10;
  static constexpr size_t N = 2;  // Do not change

  std::array<T, M*N> m1;
  std::array<T, M*N> expected_m1;
  std::array<T, N*M> m2;

  srand(time(0));
  static constexpr T MAX = 1E2;
  fill_random(m1, -MAX, MAX);

  // Expect first column unchanged, second multiplied by 2
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      m2[j * M + i] = j + 1;
      expected_m1[i * N + j] = m1[i * N + j] * m2[j * M + i];
    }
  }

  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> m1_buffer(m1.data(), cl::sycl::range<2>(M, N));
    ml::matrix_t<T> m2_buffer(m2.data(), cl::sycl::range<2>(N, M)); // TR
    m2_buffer.set_final_data(nullptr);

    ml::mat_inplace_binary_op<ml::LIN, ml::TR>(q, m1_buffer, m2_buffer, std::multiplies<T>());
    clear_eigen_device();
  }

  assert_vec_almost_eq(m1, expected_m1);
}

template <class T>
void test_tr_lin_inplace_mat_op() {
  std::cout << "\nTest transposed inplace matrix operation\n";

  static constexpr size_t M = 2;  // Do not change
  static constexpr size_t N = 10;

  std::array<T, M*N> m1;
  std::array<T, M*N> expected_m1;
  std::array<T, N*M> m2;

  srand(time(0));
  static constexpr T MAX = 1E2;
  fill_random(m1, -MAX, MAX);

  // Expect first column unchanged, second multiplied by 2
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      m2[j * M + i] = i + 1;
      expected_m1[i * N + j] = m1[i * N + j] * m2[j * M + i];
    }
  }

  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> m1_buffer(m1.data(), cl::sycl::range<2>(M, N));  // TR
    ml::matrix_t<T> m2_buffer(m2.data(), cl::sycl::range<2>(N, M));
    m2_buffer.set_final_data(nullptr);

    ml::mat_inplace_binary_op<ml::TR, ml::LIN>(q, m1_buffer, m2_buffer, std::multiplies<T>());
    clear_eigen_device();
  }

  assert_vec_almost_eq(m1, expected_m1);
}

int main() {
  try {
    test_lin_tr_inplace_mat_op<ml::buffer_data_type>();
    test_tr_lin_inplace_mat_op<ml::buffer_data_type>();
  } catch (cl::sycl::exception e) {
    std::cerr << e.what();
  }

  return 0;
}
