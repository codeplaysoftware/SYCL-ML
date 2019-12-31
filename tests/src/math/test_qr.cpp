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

#include "ml/math/cov.hpp"
#include "ml/math/mat_mul.hpp"
#include "ml/math/mat_ops.hpp"
#include "ml/math/qr.hpp"
#include "utils/utils.hpp"

template <class T>
void test_small_qr() {
  static constexpr unsigned NB_OBS = 5;
  static constexpr unsigned DATA_DIM = 3;
  std::array<T, NB_OBS * DATA_DIM> host_data{1.0,  4.0,  7.0, 2.0,  0.0,
                                             -8.0, 1.0,  2.0, 1.0,  -3.0,
                                             -1.0, -1.0, 0.0, -9.0, 6.0};

  std::array<T, NB_OBS * DATA_DIM> host_qr;
  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> sycl_data(host_data.data(),
                              cl::sycl::range<2>(NB_OBS, DATA_DIM));
    qr(q, sycl_data);
    sycl_data.set_final_data(host_qr.data());
    clear_eigen_device();
  }

  /*
  std::cout << "host data:\n";
  ml::print(host_data, NB_OBS, DATA_DIM);
  std::cout << "\nhost R:\n";
  ml::print(host_qr, NB_OBS, DATA_DIM);
  */

  // Multiple correct results are possible. Each row can be multiplied by -1.
  // In the current implementation all values on the diagonal are positive.
  // Only test the upper triangle matrix as the rest can have any value.
  assert_almost_eq(host_qr[0], T(3.87298));
  assert_almost_eq(host_qr[1], T(2.32379));
  assert_almost_eq(host_qr[2], T(-1.29099));
  assert_almost_eq(host_qr[4], T(9.82853));
  assert_almost_eq(host_qr[5], T(-2.03489));
  assert_almost_eq(host_qr[8], T(12.04959));
}

template <class T>
void test_qr_square() {
  static constexpr unsigned N = 2;
  static constexpr T DET_SIGN = -((N % 2) * 2) + 1;
  std::array<T, N * N> host_data;

  // Generate a random matrix with determinant 1
  fill_random(host_data, T(0.0), T(1.0));
  T det_data = compute_det(host_data);
  if (det_data < 0) {
    det_data *= -1;
    std::transform(begin(host_data), begin(host_data) + N, begin(host_data),
                   [](T x) { return -x; });
  }
  T factor = std::pow(det_data, -T(1.0) / N);
  std::transform(begin(host_data), end(host_data), begin(host_data),
                 [factor](T x) { return factor * x; });
  det_data = compute_det(host_data);
  assert_almost_eq(det_data, T(1));

  std::array<T, N * N> host_qr;
  T det_r;
  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> sycl_data(host_data.data(), cl::sycl::range<2>(N, N));
    ml::qr(q, sycl_data);
    det_r = DET_SIGN * reduce_diag(q, sycl_data, 0, T(1), std::multiplies<T>());
    sycl_data.set_final_data(host_qr.data());
    clear_eigen_device();
  }

  /*
  std::cout << "host data:\n";
  ml::print(host_data, N, N);
  std::cout << "\nhost R:\n";
  ml::print(host_qr, N, N);
  std::cout << "\ndeterminant: " << det_r << std::endl;
  */

  assert_almost_eq(det_r, DET_SIGN * host_qr[0] * host_qr[3]);
  assert_almost_eq(det_r, det_data);
}

class MLNormalizeR;
template <class T>
void test_qr() {
  static constexpr unsigned NB_OBS = 103;
  static constexpr unsigned DATA_DIM = 64;
  std::array<T, NB_OBS * DATA_DIM> host_data;
  fill_random(host_data, T(-10), T(10));

  std::array<T, DATA_DIM * DATA_DIM> host_cov;
  std::array<T, DATA_DIM * DATA_DIM> host_r2;
  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> sycl_data(host_data.data(),
                              cl::sycl::range<2>(NB_OBS, DATA_DIM));
    sycl_data.set_final_data(nullptr);

    // Center data
    ml::vector_t<T> sycl_data_avg((cl::sycl::range<1>(DATA_DIM)));
    ml::avg(q, sycl_data, sycl_data_avg);
    ml::center_data<ml::COL>(q, sycl_data, sycl_data_avg);

    // Expected cov
    ml::matrix_t<T> sycl_cov(cl::sycl::range<2>(DATA_DIM, DATA_DIM));
    ml::cov(q, sycl_data, sycl_cov);

    // QR
    ml::qr(q, sycl_data);
    ml::matrix_t<T> sycl_r(cl::sycl::range<2>(DATA_DIM, DATA_DIM));
    q.submit([&sycl_data, &sycl_r](cl::sycl::handler& cgh) {
      auto old_r_acc =
          sycl_data.template get_access_2d<cl::sycl::access::mode::read>(cgh);
      auto new_r_acc =
          sycl_r.template get_access_2d<cl::sycl::access::mode::discard_write>(
              cgh);
      cgh.parallel_for<ml::NameGen<0, MLNormalizeR, T>>(
          sycl_r.get_nd_range(), [=](cl::sycl::nd_item<2> item) {
            auto row = item.get_global_id(0);
            auto col = item.get_global_id(1);
            new_r_acc(row, col) =
                col >= row ? old_r_acc(row, col) / cl::sycl::sqrt(T(NB_OBS))
                           : 0;
          });
    });

    // Reconstructed cov
    ml::matrix_t<T> sycl_r2(cl::sycl::range<2>(DATA_DIM, DATA_DIM));
    ml::mat_mul<ml::TR, ml::LIN>(q, sycl_r, sycl_r, sycl_r2);

    sycl_cov.set_final_data(host_cov.data());
    sycl_r2.set_final_data(host_r2.data());
    clear_eigen_device();
  }

  assert_vec_almost_eq(host_r2, host_cov, T(1E-3));
}

template <class T>
void test_all() {
  test_small_qr<T>();
  test_qr_square<T>();
  test_qr<T>();
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
