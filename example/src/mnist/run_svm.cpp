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
#include "ml/classifiers/svm/svm.hpp"
#include "run_classifier.hpp"

int main(int argc, char** argv) {
  std::string mnist_path = "data/mnist";
  if (argc >= 2) {
    mnist_path = argv[1];
  }

  // Runs the SVM with the RBF kernel on MNIST with a PCA.
  // The SVM will store 2 rows of the kernel matrix and has a tolerance of 0.1
  using data_t = float;
  using label_t = uint8_t;
  using svm_kernel_t = ml::svm_rbf_kernel<data_t>;

  const data_t C = 5;            // Parameter of a C-SVM
  const svm_kernel_t ker(0.05);  // Parameter of the RBF kernel

  ml::pca_args<data_t> pca_args;
  pca_args.min_nb_vecs = 64;    // Keep at least 64 basis vector
  pca_args.keep_percent = 0.8;  // Keep at least 80% of information
  pca_args.scale_factor = 1E2;  // More accurate but slower PCA

  try {
    run_classifier(mnist_path, pca_args,
                   ml::svm<svm_kernel_t, label_t>(C, ker, 2, 0.1, 0.1));
  } catch (cl::sycl::exception e) {
    std::cerr << e.what();
  }

  return 0;
}
