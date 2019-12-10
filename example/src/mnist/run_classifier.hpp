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
#ifndef EXAMPLE_SRC_MNIST_RUN_CLASSIFIER_HPP
#define EXAMPLE_SRC_MNIST_RUN_CLASSIFIER_HPP

#include <iostream>

#include "ml/preprocess/apply_pca.hpp"

#include "read_mnist.hpp"
#include "utils/scoped_timer.hpp"
#include "utils/sycl_utils.hpp"

/**
 * @brief Train and test any given classifier.
 *
 * @tparam ClassifierT
 * @param mnist_path
 * @param pca_args arguments given to the PCA
 * @param classifier
 */
template <class ClassifierT>
void run_classifier(
    const std::string& mnist_path,
    const ml::pca_args<typename ClassifierT::DataType>& pca_args,
    ClassifierT classifier = ClassifierT()) {
  // The TIME macro creates an object that will print the time elapsed between
  // its construction and destruction
  TIME(run_classifier);

  using DataType = typename ClassifierT::DataType;
  using LabelType = typename ClassifierT::LabelType;

  // MNIST specific
  std::vector<LabelType> label_set(10);
  // Create the set of labels here instead of computing it during the training
  std::iota(label_set.begin(), label_set.end(), 0);
  // Data will be shifted in the range [0, 1]
  const DataType normalize_factor = 255;

  // Load and save options
  const bool load_classifier = false;
  const bool save_classifier = false;

  // What the classifier will compute
  std::vector<LabelType> host_predicted_test_labels;

  {  // Scope with a SYCL queue
    cl::sycl::queue& q = create_queue();

    ml::apply_pca<DataType> apply_pca;

    // Load the train data, perform PCA and train the classifier
    {
      unsigned obs_size, padded_obs_size, nb_train_obs;
      // Load train data
      ml::matrix_t<DataType> sycl_train_data;
      {
        auto host_train_data = read_mnist_images<DataType>(
            mnist_get_train_images_path(mnist_path), obs_size, padded_obs_size,
            nb_train_obs, false, true, normalize_factor);
        if (host_train_data.empty()) {
          return;
        }
        ml::matrix_t<DataType> sycl_train_data_raw(
            host_train_data.data(),
            cl::sycl::range<2>(nb_train_obs, padded_obs_size));
        // Specify the real size of an observation
        sycl_train_data_raw.data_range[1] = obs_size;
        sycl_train_data_raw.set_final_data(nullptr);

        sycl_train_data =
            apply_pca.compute_and_apply(q, sycl_train_data_raw, pca_args);
      }

      // Load labels
      auto host_train_labels = read_mnist_labels<LabelType>(
          mnist_get_train_labels_path(mnist_path), nb_train_obs);
      if (host_train_labels.empty()) {
        return;
      }

      if (load_classifier) {
        classifier.load_from_disk(q);
      } else {
        {  // Create a scope to time only the training
          TIME(train_classifier);
          // Give the sets of labels to avoid computing it during the training
          classifier.set_label_set(label_set);
          classifier.train(q, sycl_train_data, host_train_labels);
          // Wait to measure the correct training time
          q.wait_and_throw();
        }
        if (save_classifier) {
          classifier.save_to_disk(q);
        }
      }
    }  // End of train

    // Load the test data, apply the PCA using the eigenvectors from the
    // training and test the classifier
    {
      unsigned obs_size, padded_obs_size, nb_test_obs;
      ml::matrix_t<DataType> sycl_test_data;
      {  // Load test data
        auto host_test_data = read_mnist_images<DataType>(
            mnist_get_test_images_path(mnist_path), obs_size, padded_obs_size,
            nb_test_obs, false, true, normalize_factor);
        if (host_test_data.empty()) {
          return;
        }
        ml::matrix_t<DataType> sycl_test_data_raw(
            host_test_data.data(),
            cl::sycl::range<2>(nb_test_obs, padded_obs_size));
        // Specify the real size of an observation
        sycl_test_data_raw.data_range[1] = obs_size;
        sycl_test_data_raw.set_final_data(nullptr);

        sycl_test_data = apply_pca.apply(q, sycl_test_data_raw);
      }

      // Inference
      TIME(predict_classifier);
      auto sycl_predicted_test_labels = classifier.predict(q, sycl_test_data);
      // Can be rounded up to a power of 2
      auto nb_labels_predicted = sycl_predicted_test_labels.get_kernel_size();
      host_predicted_test_labels.resize(nb_labels_predicted);
      sycl_predicted_test_labels.set_final_data(
          host_predicted_test_labels.data());
      // Wait to measure the correct prediction time
      q.wait_and_throw();
    }  // End of tests

    clear_eigen_device();
  }  // SYCL queue is destroyed

  // Compare predicted labels and expected labels
  unsigned nb_test_obs;
  auto host_expected_test_labels = read_mnist_labels<LabelType>(
      mnist_get_test_labels_path(mnist_path), nb_test_obs);
  if (host_expected_test_labels.empty()) {
    return;
  }
  classifier.print_score(host_predicted_test_labels.data(),
                         host_expected_test_labels.data(), nb_test_obs);
}

#endif  // EXAMPLE_SRC_MNIST_RUN_CLASSIFIER_HPP
