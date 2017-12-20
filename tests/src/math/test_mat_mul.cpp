#include <iostream>
#include <random>

#include "ml/math/mat_mul.hpp"
#include "utils/utils.hpp"

template <class T>
void test_square() {
  std::cout << "\nTest square matrices\n";
  std::array<T, 4> m1 {1.0, 2.0,
                       3.0, 4.0};
  std::array<T, 4> m2 {-1.0, 1.0,
                       5.0, -2.0};
  std::array<T, 4> m3;

  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> m1_buffer(m1.data(), cl::sycl::range<2>(2, 2));
    m1_buffer.set_final_data(nullptr);
    ml::matrix_t<T> m2_buffer(m2.data(), cl::sycl::range<2>(2, 2));
    m2_buffer.set_final_data(nullptr);
    ml::matrix_t<T> out_buffer(cl::sycl::range<2>(2, 2));
    ml::mat_mul(q, m1_buffer, m2_buffer, out_buffer);
    out_buffer.set_final_data(m3.data());
    clear_eigen_device();
  }

  std::cout << "m1:\n";
  ml::print(m1, 2, 2);
  std::cout << "\nm2:\n";
  ml::print(m2, 2, 2);
  std::cout << "\nm3:\n";
  ml::print(m3, 2, 2);

  assert_vec_almost_eq(m3, {9.0, -3.0,
                            17.0, -5.0});
}

template <class T>
void test_general() {
  std::cout << "\nTest general matrices\n";
  std::array<T, 6> m1 {1.0, 2.0, 3.0,
                       4.0, 5.0, 6.0};
  std::array<T, 3> m2 {-1.0,
                        5.0,
                        2.0};
  std::array<T, 2> m3;

  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> m1_buffer(m1.data(), cl::sycl::range<2>(2, 3));
    m1_buffer.set_final_data(nullptr);
    ml::matrix_t<T> m2_buffer(m2.data(), cl::sycl::range<2>(3, 1));
    m2_buffer.set_final_data(nullptr);
    ml::matrix_t<T> out_buffer(cl::sycl::range<2>(2, 1));
    ml::mat_mul(q, m1_buffer, m2_buffer, out_buffer);
    out_buffer.set_final_data(m3.data());
    clear_eigen_device();
  }

  std::cout << "m1:\n";
  ml::print(m1, 2, 3);
  std::cout << "\nm2:\n";
  ml::print(m2, 3, 1);
  std::cout << "\nm3:\n";
  ml::print(m3, 2, 1);

  assert_vec_almost_eq(m3, {15.0, 33.0});
}

template <class T>
void test_simple_and_eigen() {
  std::cout << "\nTest simple and eigen mat_mul\n";

  static constexpr size_t M = 500;
  static constexpr size_t N = 100;
  static constexpr size_t K = 3;

  std::array<T, M*K> m1;
  std::array<T, K*N> m2;
  std::array<T, M*N> eigen_out;
  std::array<T, M*N> my_out;

  srand(time(0));
  static constexpr T MAX = 1E2;
  fill_random(m1, -MAX, MAX);
  fill_random(m2, -MAX, MAX);

  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> m1_buffer(m1.data(), cl::sycl::range<2>(M, K));
    m1_buffer.set_final_data(nullptr);
    ml::matrix_t<T> m2_buffer(m2.data(), cl::sycl::range<2>(N, K)); // TR
    m2_buffer.set_final_data(nullptr);

    ml::matrix_t<T> eigen_out_buffer(cl::sycl::range<2>(M, N));
    ml::mat_mul<ml::LIN, ml::TR>(q, m1_buffer, m2_buffer, eigen_out_buffer);

    ml::matrix_t<T> my_out_buffer(cl::sycl::range<2>(M, N));
    ml::simple_mat_mul<ml::LIN, ml::TR>(q, m1_buffer, m2_buffer, my_out_buffer);

    eigen_out_buffer.set_final_data(eigen_out.data());
    my_out_buffer.set_final_data(my_out.data());
    clear_eigen_device();
  }

  assert_vec_almost_eq(my_out, eigen_out);
  std::cout << "Outputs are equal." << std::endl;
}

template <class T>
void test_simple_and_eigen_vec() {
  std::cout << "\nTest simple and eigen mat_mul_vec\n";

  static constexpr size_t M = 500;
  static constexpr size_t K = 3;

  std::array<T, M*K> m1;
  std::array<T, K> m2;
  std::array<T, M> eigen_out;
  std::array<T, M> my_out;

  srand(time(0));
  fill_random(m1, -100, 100);
  fill_random(m2, -100, 100);

  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> m1_buffer(m1.data(), cl::sycl::range<2>(M, K));
    m1_buffer.set_final_data(nullptr);
    ml::vector_t<T> m2_buffer(m2.data(), cl::sycl::range<1>(K));
    m2_buffer.set_final_data(nullptr);

    ml::vector_t<T> eigen_out_buffer{cl::sycl::range<1>(M)};
    ml::mat_mul(q, m1_buffer, m2_buffer, eigen_out_buffer);

    ml::vector_t<T> my_out_buffer{cl::sycl::range<1>(M)};
    ml::simple_mat_mul_vec(q, m1_buffer, m2_buffer, my_out_buffer);

    eigen_out_buffer.set_final_data(eigen_out.data());
    my_out_buffer.set_final_data(my_out.data());
    clear_eigen_device();
  }

  assert_vec_almost_eq(my_out, eigen_out);
  std::cout << "Outputs are equal." << std::endl;
}

int main() {
  test_square<ml::buffer_data_type>();
  test_general<ml::buffer_data_type>();
  test_simple_and_eigen<ml::buffer_data_type>();
  test_simple_and_eigen_vec<ml::buffer_data_type>();

  return 0;
}

