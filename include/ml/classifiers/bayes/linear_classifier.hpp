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
#ifndef INCLUDE_ML_CLASSIFIERS_BAYES_LINEAR_CLASSIFIER_HPP
#define INCLUDE_ML_CLASSIFIERS_BAYES_LINEAR_CLASSIFIER_HPP

#include "ml/classifiers/data_splitter_extremum_dist.hpp"

namespace ml {

/**
 * @brief Naive Bayes Classifier with a linear function.
 *
 * During the training, compute the average for each label.
 * The distance is the Euclidean distance between the learned average and the
 * given sample. The index of the smallest distance then gives the predicted
 * label.
 *
 * The linear_classifier could be written using the bayes_classifier but is
 * simpler and faster this way.
 *
 * @tparam DataT
 * @tparam LabelT
 */
template <class DataT, class LabelT>
class linear_classifier : public data_splitter_extremum_dist<DataT, LabelT> {
 protected:
  vector_t<DataT> _act_data_avg;
  matrix_t<DataT> _data_avg_per_label;

  virtual void train_setup_for_each_label(queue& q) override {
    data_splitter_extremum_dist<DataT, LabelT>::train_setup_for_each_label(q);

    auto nb_labels = this->get_nb_labels();
    _act_data_avg = vector_t<DataT>(range<1>(this->_data_dim),
                                    get_optimal_nd_range(this->_data_dim_pow2));
    _data_avg_per_label =
        matrix_t<DataT>(range<2>(nb_labels, this->_data_dim),
                        get_optimal_nd_range(nb_labels, this->_data_dim_pow2));
  }

  virtual void train_for_each_label(queue& q, unsigned label_idx,
                                    matrix_t<DataT>& act_data) override {
    avg(q, act_data, _act_data_avg);
    copy_vec_to_mat<ROW, access::mode::discard_write>(
        q, _data_avg_per_label, _act_data_avg, _act_data_avg.kernel_range,
        static_cast<SYCLIndexT>(label_idx));
  }

  virtual void compute_dist(queue&, matrix_t<DataT>& dataset,
                            matrix_t<DataT>& dist) override {
    // Sum squared each pixel
    eig_index_t nb_labels = static_cast<eig_index_t>(access_data_dim(dist, 0));
    eig_index_t nb_obs = static_cast<eig_index_t>(access_data_dim(dataset, 0));
    eig_index_t data_dim_pow2 = static_cast<eig_index_t>(this->_data_dim_pow2);

    auto eig_dataset = sycl_to_eigen(dataset);
    auto eig_data_avg_per_label = sycl_to_eigen(_data_avg_per_label);

    auto eig_dist = sycl_to_eigen(dist);
    auto dataset_3d = eig_dataset.tensor()
                          .reshape(eig_dims_t<3>{nb_obs, 1, data_dim_pow2})
                          .broadcast(eig_dims_t<3>{1, nb_labels, 1});
    auto data_avg_per_label_3d =
        eig_data_avg_per_label.tensor()
            .reshape(eig_dims_t<3>{1, nb_labels, data_dim_pow2})
            .broadcast(eig_dims_t<3>{nb_obs, 1, 1});
    auto sliced_dist = eig_dist.tensor().slice(
        eig_dsize_t<2>{0, 0}, eig_dsize_t<2>(nb_labels, nb_obs));
    sliced_dist.device(get_eigen_device()) =
        (dataset_3d - data_avg_per_label_3d)
            .square()
            .sum(eig_dims_t<1>{2})
            .shuffle(eig_dims_t<2>{1, 0});
  }
};

}  // namespace ml

#endif  // INCLUDE_ML_CLASSIFIERS_BAYES_LINEAR_CLASSIFIER_HPP
