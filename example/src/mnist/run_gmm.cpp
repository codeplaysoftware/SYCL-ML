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
#include "ml/classifiers/bayes/distributions/log_gaussian_distribution.hpp"
#include "ml/classifiers/em/em_classifier.hpp"
#include "ml/classifiers/em/log_model_per_label.hpp"
#include "run_classifier.hpp"

int main(int argc, char** argv) {
  std::string mnist_path = "data/mnist";
  if (argc >= 2) {
    mnist_path = argv[1];
  }

  // Runs the GMM with 8 models per label on MNIST with a PCA
  using data_t = float;
  using label_t = uint8_t;
  using distribution_t = ml::buffered_log_gaussian_distribution<data_t>;
  static constexpr unsigned M = 8;
  ml::pca_args<data_t> pca_args;
  pca_args.min_nb_vecs = 64;    // Keep at least 64 basis vector
  pca_args.keep_percent = 0.8;  // Keep at least 80% of information
  pca_args.scale_factor = 1E2;  // More accurate but slower PCA
  try {
    run_classifier<
        ml::em_classifier<label_t, ml::log_model_per_label<M, distribution_t>>>(
        mnist_path, pca_args);
  } catch (cl::sycl::exception e) {
    std::cerr << e.what();
  }

  return 0;
}
