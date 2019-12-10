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
void test_svm_poly() {
  /*
   * Solves the XOR problem, kernel has to be at least polynomial or more
   * complex. y  0  1
   * x
   * 0    0  1
   * 1    1  0
   */
  std::array<DataT, 8> host_data{0, 0, 0, 1, 1, 0, 1, 1};
  std::vector<LabelT> host_labels{0, 1, 1, 0};
  std::vector<DataT> host_alphas;
  DataT host_rho;

  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<DataT> sycl_data(host_data.data(), cl::sycl::range<2>(4, 2));

    using KernelType = ml::svm_polynomial_kernel<DataT>;
    ml::svm<KernelType, LabelT> svm(1000, KernelType(1, 1, 2), 2, 1E-6);
    svm.train_binary(q, sycl_data, host_labels);

    auto smo_out = svm.get_smo_outs().front();
    assert_eq(smo_out.alphas.data_range[0], 4LU);
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
  ml::print(host_alphas.data(), 1, 4);
  std::cout << "\nrho: " << host_rho << std::endl;
  */

  std::array<DataT, 4> expected_alphas{-3.332425, 2.665940, 2.665940,
                                       -1.999455};
  assert_vec_almost_eq(host_alphas.data(), expected_alphas.data(),
                       expected_alphas.size(), DataT(1E-3));
  assert_almost_eq(host_rho, DataT(-0.999728), DataT(1E-3));
}

int main() {
  try {
    test_svm_poly<float, uint8_t>();
#ifdef SYCLML_TEST_DOUBLE
    test_svm_poly<double, uint8_t>();
#endif
  } catch (cl::sycl::exception e) {
    std::cerr << e.what();
  }

  return 0;
}
