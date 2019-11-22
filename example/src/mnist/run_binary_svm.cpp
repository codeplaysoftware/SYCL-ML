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
#include "ml/preprocess/apply_pca.hpp"
#include "read_mnist.hpp"
#include "utils/scoped_timer.hpp"
#include "utils/sycl_utils.hpp"

template <class Cond, class LabelT, class IndexT>
void select_indices(LabelT* data, unsigned data_size,
                    std::vector<IndexT>& indices, Cond cond = Cond()) {
  for (unsigned i = 0; i < data_size; ++i) {
    if (cond(data[i]))
      indices.push_back(i);
  }
}

template <class DataT, class LabelT>
void run_binary_svm(const std::string& mnist_path) {
  // MNIST specific
  std::vector<LabelT> label_set(2);  // Classify only labels 0 and 1
  std::iota(label_set.begin(), label_set.end(), 0);
  const DataT normalize_factor = 255;
  auto select_labels = [](LabelT l) { return l <= 1; };

  // Run the polynomial kernel by default
  using KernelType = ml::svm_polynomial_kernel<DataT>;
  ml::svm<KernelType, LabelT> svm(1000, KernelType(1, 1, 2));
  /*
  using KernelType = ml::svm_rbf_kernel<DataT>;
  ml::svm<KernelType, LabelT> svm(5, KernelType(0.05), 100);
  */

  std::shared_ptr<LabelT> host_split_expected_test_labels;
  std::shared_ptr<LabelT> host_split_predicted_test_labels;
  unsigned nb_test_obs;
  {
    cl::sycl::queue& q = create_queue();

    ml::apply_pca<DataT> apply_pca;
    ml::pca_args<DataT> pca_args;
    pca_args.keep_percent = 0.f;  // PCA disabled

    // Load and train on 0s and 1s
    {
      unsigned obs_size, padded_obs_size, nb_train_obs;
      // Load train dataset and labels on host. The dataset will not be
      // transposed and the data dimension will be padded  to a power of 2.
      auto host_train_data = read_mnist_images<DataT>(
          mnist_get_train_images_path(mnist_path), obs_size, padded_obs_size,
          nb_train_obs, false, true, normalize_factor);
      auto host_train_labels = read_mnist_labels<LabelT>(
          mnist_get_train_labels_path(mnist_path), nb_train_obs);

      // Copy dataset and labels to device
      ml::matrix_t<DataT> sycl_train_data(
          host_train_data, cl::sycl::range<2>(nb_train_obs, padded_obs_size));
      sycl_train_data.data_range[1] =
          obs_size;  // Specify the real size of an observation
      sycl_train_data.set_final_data(nullptr);
      ml::vector_t<LabelT> sycl_train_labels(host_train_labels,
                                             cl::sycl::range<1>(nb_train_obs));
      sycl_train_labels.set_final_data(nullptr);

      // Select 0s and 1s
      std::vector<unsigned> indices;
      select_indices(host_train_labels.get(), nb_train_obs, indices,
                     select_labels);
      auto split_train_data = split_by_index(q, sycl_train_data, indices);
      auto split_train_labels = split_by_index(q, sycl_train_labels, indices);

      // Compute a new basis and apply it. The basis will be automatically
      // loaded if already saved before, otherise it will be computed and saved.
      split_train_data =
          apply_pca.compute_and_apply(q, split_train_data, pca_args);

      // Train
      TIME(train_binary_svm);
      svm.train_binary(q, split_train_data, split_train_labels);
      q.wait_and_throw();  // wait to measure the correct training time
    }

    auto& smo_out = svm.get_smo_outs().front();
    std::cout << "\nNumber of iterations: " << smo_out.nb_iter << std::endl;
    std::cout << "Number support vectors: " << smo_out.alphas.data_range[0]
              << "\n"
              << std::endl;

    // Load and test on 0s and 1s
    {
      unsigned obs_size, padded_obs_size;
      // Load test dataset and labels on host. The dataset will not be
      // transposed and the data dimension will be padded to a power of 2.
      auto host_test_data = read_mnist_images<DataT>(
          mnist_get_test_images_path(mnist_path), obs_size, padded_obs_size,
          nb_test_obs, false, true, normalize_factor);
      auto host_expected_test_labels = read_mnist_labels<LabelT>(
          mnist_get_test_labels_path(mnist_path), nb_test_obs);

      // Copy labels on device
      ml::matrix_t<DataT> sycl_test_data(
          host_test_data, cl::sycl::range<2>(nb_test_obs, padded_obs_size));
      sycl_test_data.data_range[1] =
          obs_size;  // Specify the real size of an observation
      sycl_test_data.set_final_data(nullptr);

      // Select 0s and 1s
      std::vector<unsigned> indices;
      select_indices(host_expected_test_labels.get(), nb_test_obs, indices,
                     select_labels);
      auto split_test_data = split_by_index(q, sycl_test_data, indices);
      nb_test_obs = indices.size();
      host_split_expected_test_labels =
          ml::make_shared_array(new LabelT[nb_test_obs]);
      for (unsigned i = 0; i < nb_test_obs; ++i)
        host_split_expected_test_labels.get()[i] =
            host_expected_test_labels.get()[indices[i]];

      // Apply the same basis than the PCA during the training
      split_test_data = apply_pca.apply(q, split_test_data);

      // Inference
      TIME(predict_binary_svm);
      auto sycl_predicted_test_labels = svm.predict(q, split_test_data);
      auto prediction_size =
          sycl_predicted_test_labels
              .get_count();  // Can be rounded up to a power of 2
      host_split_predicted_test_labels =
          ml::make_shared_array(new LabelT[prediction_size]);
      sycl_predicted_test_labels.set_final_data(
          host_split_predicted_test_labels);
      q.wait_and_throw();  // wait to measure the correct prediction time
    }

    clear_eigen_device();
  }

  svm.print_score(host_split_predicted_test_labels.get(),
                  host_split_expected_test_labels.get(), nb_test_obs);
}

int main(int argc, char** argv) {
  std::string mnist_path = "data/mnist";
  if (argc >= 2)
    mnist_path = argv[1];
  try {
    run_binary_svm<float, uint8_t>(mnist_path);
  } catch (cl::sycl::exception e) {
    std::cerr << e.what();
  }

  return 0;
}
