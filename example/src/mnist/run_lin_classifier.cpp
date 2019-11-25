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
#include "ml/classifiers/bayes/linear_classifier.hpp"
#include "run_classifier.hpp"

int main(int argc, char** argv) {
  std::string mnist_path = "data/mnist";
  if (argc >= 2) {
    mnist_path = argv[1];
  }
  // Runs the linear classifier on MNIST with a PCA
  using data_t = float;
  ml::pca_args<data_t> pca_args;
  pca_args.min_nb_vecs = 128;   // Keep at least 128 basis vector
  pca_args.keep_percent = 0.8;  // Keep at least 80% of information
  pca_args.scale_factor = 1E2;  // More accurate but slower PCA
  try {
    run_classifier<ml::linear_classifier<data_t, uint8_t>>(mnist_path,
                                                           pca_args);
  } catch (cl::sycl::exception e) {
    std::cerr << e.what();
  }

  return 0;
}
