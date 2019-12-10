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
#include <iostream>
#include <random>

#include "ml/math/mat_inv.hpp"
#include "ml/math/tri_inv.hpp"
#include "utils/utils.hpp"

template <class T>
void test_inv() {
  std::array<T, 9> host_data{1.0, 4.0, 6.0, 0.0, -1.0, 2.0, 5.0, 3.0, 4.0};

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

  /*
  std::cout << "data:\n";
  ml::print(host_data, 3, 3);
  std::cout << "\ninv:\n";
  ml::print(host_inv, 3, 3);
  */

  std::array<T, 9> expected{-0.166667, 0.0333335, 0.233333,
                            0.166667,  -0.433333, -0.0333333,
                            0.0833333, 0.283333,  -0.0166667};
  assert_vec_almost_eq(host_inv, expected);
}

template <class T>
void test_inv_big() {
  static constexpr unsigned SIDE = 100;
  static constexpr unsigned SIZE = SIDE * SIDE;
  static constexpr T MAX = 1E2;
  std::array<T, SIZE> host_data;
  srand(time(0));
  std::generate(std::begin(host_data), std::end(host_data), [=]() {
    return MAX * ((2 * (static_cast<T>(rand()) / RAND_MAX)) - 1);
  });

  // Make the input matrix diagonally dominant to ensure that it is invertible
  for (unsigned r = 0; r < SIDE; ++r) {
    T abs_max = host_data[r];
    for (unsigned c = 0; c < SIDE; ++c) {
      T abs_rc = std::abs(host_data[r * SIDE + c]);
      if (abs_rc > abs_max) {
        abs_max = abs_rc;
      }
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
    // ml::write_bmp_grayscale("inv_" + std::to_string(SIDE), sycl_inv, true,
    // true);

    ml::matrix_t<T> multiplication{rng};
    ml::mat_mul(q, sycl_data, sycl_inv, multiplication);
    // ml::write_bmp_grayscale("inv_multiplication_" + std::to_string(SIDE),
    // multiplication, true, true);

    ml::matrix_t<T> identity{rng};
    ml::eye(q, identity);
    ml::matrix_t<T> diff{rng};
    ml::sycl_copy(q, identity, diff);
    ml::mat_inplace_binary_op(q, diff, multiplication, std::minus<T>());
    // ml::write_bmp_grayscale("inv_diff_" + std::to_string(SIDE), diff, true,
    // true);
    diff.set_final_data(host_diff.data());
    clear_eigen_device();
  }

  for (unsigned i = 0; i < SIZE; ++i) {
    assert_almost_eq(host_diff[i], T(0), T(1E-3));
  }
}

template <class T>
void test_tri_inv() {
  std::array<T, 16> host_data{1.0, 2.0, 3.0, 4.0, 0.0, 5.0, 6.0, 7.0,
                              0.0, 0.0, 8.0, 9.0, 0.0, 0.0, 0.0, 10.0};

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

  /*
  std::cout << "data:\n";
  ml::print(host_data, 4, 4);
  std::cout << "\ninv:\n";
  ml::print(host_inv, 4, 4);
  */

  std::array<T, 16> expected{1.0,   -0.4,   -0.075, -0.0525, 0.0,   0.2,
                             -0.15, -0.005, 0.0,    0.0,     0.125, -0.1125,
                             0.0,   0.0,    0.0,    0.1};
  assert_vec_almost_eq(host_inv, expected);
}

template <class T>
void test_tri_inv_big() {
  static constexpr unsigned SIDE = 64;
  static constexpr unsigned SIZE = SIDE * SIDE;
  std::array<T, SIZE> host_data;
  for (unsigned r = 0; r < SIDE; ++r) {
    for (unsigned c = 0; c < SIDE; ++c) {
      host_data[r * SIDE + c] = r > c ? 0 : r * SIDE + c + 1;
    }
  }

  std::array<T, SIZE> host_diff;
  {
    cl::sycl::queue& q = create_queue();
    cl::sycl::range<2> rng(SIDE, SIDE);
    ml::matrix_t<T> sycl_data(host_data.data(), rng);
    sycl_data.set_final_data(nullptr);

    ml::matrix_t<T> sycl_tri_inv{rng};
    ml::tri_inv(q, sycl_data, sycl_tri_inv);
    // ml::write_bmp_grayscale("tri_inv_" + std::to_string(SIDE), sycl_tri_inv,
    // true, true);

    ml::matrix_t<T> multiplication{rng};
    ml::mat_mul(q, sycl_data, sycl_tri_inv, multiplication);
    // ml::write_bmp_grayscale("tri_inv_multiplication_" + std::to_string(SIDE),
    // multiplication, true, true);

    ml::matrix_t<T> identity{rng};
    ml::eye(q, identity);
    ml::matrix_t<T> diff{rng};
    ml::sycl_copy(q, identity, diff);
    ml::mat_inplace_binary_op(q, diff, multiplication, std::minus<T>());
    // ml::write_bmp_grayscale("tri_inv_diff_" + std::to_string(SIDE), diff,
    // true, true);

    diff.set_final_data(host_diff.data());
    clear_eigen_device();
  }

  for (unsigned i = 0; i < SIZE; ++i) {
    assert_almost_eq(host_diff[i], T(0), T(1E-2));
  }
}

template <class T>
void test_all() {
  test_inv<T>();
  // test_inv_big<T>();
  test_tri_inv<T>();
  // test_tri_inv_big<T>();
}

int main(void) {
  try {
    test_all<float>();
#ifdef SYCLML_TEST_DOUBLE
    test_all<double>();
#endif
  } catch (cl::sycl::exception e) {
    std::cerr << e.what();
  }

  return 0;
}
