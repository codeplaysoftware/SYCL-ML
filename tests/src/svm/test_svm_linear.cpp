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

#include "ml/classifiers/svm/svm.hpp"
#include "utils/utils.hpp"

template <class DataT, class LabelT>
void test_svm_linear() {
  /*
   * Solves the OR problem, kernel can be linear.
   *   y  0  1
   * x
   * 0    0  1
   * 1    1  1
   */
  std::array<DataT, 8> host_data{0, 0, 0, 1, 1, 0, 1, 1};
  std::vector<LabelT> host_labels{0, 1, 1, 1};
  std::vector<DataT> host_alphas;
  DataT host_rho;

  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<DataT> sycl_data(host_data.data(), cl::sycl::range<2>(4, 2));

    ml::svm<ml::svm_linear_kernel<DataT>, LabelT> svm(10);
    svm.train_binary(q, sycl_data, host_labels);

    auto smo_out = svm.get_smo_outs().front();
    assert_eq(smo_out.alphas.data_range[0], 3LU);
    host_alphas.resize(smo_out.alphas.get_kernel_size());
    auto event =
        ml::sycl_copy_device_to_host(q, smo_out.alphas, host_alphas.data());
    event.wait_and_throw();
    host_rho = smo_out.rho;

    sycl_data.set_final_data(nullptr);
    clear_eigen_device();
  }

  /*
  std::cout << "alphas:\n";
  ml::print(host_alphas.data(), 1, 3);
  std::cout << "\nrho: " << host_rho << std::endl;
  */

  std::array<DataT, 3> expected_alphas{-4, 2, 2};
  assert_vec_almost_eq(host_alphas.data(), expected_alphas.data(),
                       expected_alphas.size());
  assert_almost_eq(host_rho, DataT(-1));
}

int main() {
  try {
    test_svm_linear<float, uint8_t>();
#ifdef SYCLML_TEST_DOUBLE
    test_svm_linear<double, uint8_t>();
#endif
  } catch (cl::sycl::exception e) {
    std::cerr << e.what();
  }

  return 0;
}
