#include <iostream>
#include <random>

#include "ml/math/mat_inv.hpp"
#include "ml/math/tri_inv.hpp"
#include "utils/utils.hpp"

template <class T>
void test_inv() {
  std::array<T, 9> host_data {1.0, 4.0, 6.0,
                              0.0, -1.0, 2.0,
                              5.0, 3.0, 4.0};

  std::array<T, 9> host_inv;
  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> sycl_data(host_data.data(), cl::sycl::range<2>(3, 3));
    sycl_data.set_final_data(nullptr);

    ml::matrix_t<T> sycl_inv{cl::sycl::range<2>(3, 3)};
    ml::mat_inv(q, sycl_data, sycl_inv);

    sycl_inv.set_final_data(host_inv.data());
    clear_eigen_device();
  }

  std::cout << "data:\n";
  ml::print(host_data, 3, 3);
  std::cout << "\ninv:\n";
  ml::print(host_inv, 3, 3);

  std::array<T, 9> expected {-0.166667, 0.0333335, 0.233333,
                             0.166667, -0.433333, -0.0333333,
                             0.0833333, 0.283333, -0.0166667};
  assert_vec_almost_eq(host_inv, expected);
}

template <class T>
void test_inv_big() {
  static constexpr unsigned SIDE = 100;
  static constexpr unsigned SIZE = SIDE * SIDE;
  static constexpr T MAX = 1E2;
  std::array<T, SIZE> host_data;
  srand(time(0));
  std::generate(begin(host_data), end(host_data), [=]() {
    return MAX * ((2 * (static_cast<T>(rand()) / RAND_MAX)) - 1);
  });

  // Make it diagonally dominant
  for (unsigned r = 0; r < SIDE; ++r) {
    T abs_max = host_data[r];
    for (unsigned c = 0; c < SIDE; ++c) {
      if (std::abs(host_data[r * SIDE + c]) > abs_max)
        abs_max = std::abs(host_data[r * SIDE + c]);
    }
    auto& x = host_data[r * SIDE + r];
    x = cl::sycl::sign(x) * (std::abs(x) + abs_max);
  }

  std::array<T, SIZE> host_diff;
  {
    cl::sycl::queue& q = create_queue();
    cl::sycl::range<2> rng(SIDE, SIDE);
    ml::matrix_t<T> sycl_data(host_data.data(), rng);
    sycl_data.set_final_data(nullptr);

    ml::matrix_t<T> sycl_inv{rng};
    ml::mat_inv(q, sycl_data, sycl_inv);

    ml::matrix_t<T> identity{rng};
    ml::eye(q, identity);
    ml::matrix_t<T> multiplication{rng};
    ml::mat_mul(q, sycl_data, sycl_inv, multiplication);
    ml::write_bmp_grayscale("inv_multiplication_" + std::to_string(SIDE), multiplication, true, true);

    ml::matrix_t<T> diff{rng};
    ml::sycl_copy(q, identity, diff);
    ml::mat_inplace_binary_op(q, diff, multiplication, std::minus<T>());
    ml::write_bmp_grayscale("inv_diff_" + std::to_string(SIDE), diff, true, true);
    diff.set_final_data(host_diff.data());
    clear_eigen_device();
  }

  for (unsigned i = 0; i < SIZE; ++i)
    assert_almost_eq(host_diff[i], T(0));
}

template <class T>
void test_tri_inv() {
  std::array<T, 16> host_data {1.0, 2.0, 3.0, 4.0,
                               0.0, 5.0, 6.0, 7.0,
                               0.0, 0.0, 8.0, 9.0,
                               0.0, 0.0, 0.0, 10.0};

  std::array<T, 16> host_inv;
  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> sycl_data(host_data.data(), cl::sycl::range<2>(4, 4));
    sycl_data.set_final_data(nullptr);
    ml::matrix_t<T> sycl_inv{cl::sycl::range<2>(4, 4)};
    ml::tri_inv(q, sycl_data, sycl_inv);
    sycl_inv.set_final_data(host_inv.data());
    clear_eigen_device();
  }

  std::cout << "data:\n";
  ml::print(host_data, 4, 4);
  std::cout << "\ninv:\n";
  ml::print(host_inv, 4, 4);

  std::array<T, 16> expected {1.0, -0.4, -0.075, -0.0525,
                              0.0, 0.2, -0.15, -0.005,
                              0.0, 0.0, 0.125, -0.1125,
                              0.0, 0.0, 0.0, 0.1};
  assert_vec_almost_eq(host_inv, expected);
}

template <class T>
void test_tri_inv_big() {
  static constexpr unsigned SIDE = 64;
  static constexpr unsigned SIZE = SIDE * SIDE;
  std::array<T, SIZE> host_data;
  for (unsigned r = 0; r < SIDE; ++r) {
    for (unsigned c = 0; c < SIDE; ++c) {
      if (r > c)
        host_data[r * SIDE + c] = 0;
      else
        host_data[r * SIDE + c] = r * SIDE + c + 1;
    }
  }

  std::array<T, SIZE> host_inv;
  std::array<T, SIZE> host_tri_inv;
  {
    cl::sycl::queue& q = create_queue();
    cl::sycl::range<2> rng(SIDE, SIDE);
    ml::matrix_t<T> sycl_data(host_data.data(), rng);
    sycl_data.set_final_data(nullptr);

    ml::matrix_t<T> sycl_inv{rng};
    ml::mat_inv(q, sycl_data, sycl_inv);
    ml::write_bmp_grayscale("inv_" + std::to_string(SIDE), sycl_inv, true, true);

    ml::matrix_t<T> sycl_tri_inv{rng};
    ml::tri_inv(q, sycl_data, sycl_tri_inv);
    ml::write_bmp_grayscale("tri_inv_" + std::to_string(SIDE), sycl_tri_inv, true, true);

    // Compare inv and tri_inv
    ml::matrix_t<T> identity{rng};
    ml::eye(q, identity);
    ml::matrix_t<T> multiplication{rng};
    ml::matrix_t<T> diff{rng};

    ml::mat_mul(q, sycl_data, sycl_inv, multiplication);
    ml::write_bmp_grayscale("inv_multiplication_" + std::to_string(SIDE), multiplication, true, true);

    ml::sycl_copy(q, identity, diff);
    ml::mat_inplace_binary_op(q, diff, multiplication, std::minus<T>());
    ml::write_bmp_grayscale("inv_diff_" + std::to_string(SIDE), diff, true, true);

    ml::mat_mul(q, sycl_data, sycl_tri_inv, multiplication);
    ml::write_bmp_grayscale("tri_inv_multiplication_" + std::to_string(SIDE), multiplication, true, true);

    ml::sycl_copy(q, identity, diff);
    ml::mat_inplace_binary_op(q, diff, multiplication, std::minus<T>());
    ml::write_bmp_grayscale("tri_inv_diff_" + std::to_string(SIDE), diff, true, true);

    sycl_inv.set_final_data(host_inv.data());
    sycl_tri_inv.set_final_data(host_tri_inv.data());
    clear_eigen_device();
  }

  assert_vec_almost_eq(host_tri_inv.data(), host_inv.data(), SIZE); // Differences grow larger as MAX and SIDE grow
}

int main(void) {
  test_inv<ml::buffer_data_type>();
  test_tri_inv<ml::buffer_data_type>();

  return 0;
}
