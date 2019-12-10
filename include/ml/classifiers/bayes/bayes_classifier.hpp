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
#ifndef INCLUDE_ML_CLASSIFIERS_BAYES_BAYES_CLASSIFIER_HPP
#define INCLUDE_ML_CLASSIFIERS_BAYES_BAYES_CLASSIFIER_HPP

#include "ml/classifiers/data_splitter_extremum_dist.hpp"

namespace ml {

/**
 * @brief Naive Bayes Classifier
 *
 * Compute the parameters of a distribution during the training.
 * Use the parameters during the inference.
 *
 * @tparam DistributionT
 * @tparam LabelT
 */
template <class DistributionT, class LabelT>
class bayes_classifier
    : public data_splitter_extremum_dist<typename DistributionT::DataType,
                                         LabelT, GREATER> {
 public:
  using DataType = typename DistributionT::DataType;

 protected:
  std::vector<DistributionT> _distributions;

  virtual void train_setup_for_each_label(queue& q) override {
    data_splitter_extremum_dist<DataType, LabelT,
                                GREATER>::train_setup_for_each_label(q);

    range<1> data_dim_rng(this->_data_dim);
    auto data_dim_pow2_rng = get_optimal_nd_range(this->_data_dim_pow2);
    range<2> data_dim_rng_d2(this->_data_dim, this->_data_dim);
    auto data_dim_pow2_rng_d2 =
        get_optimal_nd_range(this->_data_dim_pow2, this->_data_dim_pow2);

    auto nb_labels = this->get_nb_labels();
    for (unsigned l = 0; l < nb_labels; ++l) {
      _distributions.emplace_back();
      _distributions.back().init(data_dim_rng, data_dim_pow2_rng,
                                 data_dim_rng_d2, data_dim_pow2_rng_d2);
    }
  }

  virtual void train_for_each_label(queue& q, unsigned label_idx,
                                    matrix_t<DataType>& act_data) override {
    _distributions[label_idx].compute(q, act_data);
  }

  virtual void compute_dist(queue& q, matrix_t<DataType>& dataset,
                            matrix_t<DataType>& dist) override {
    auto nb_labels = this->get_nb_labels();
    for (SYCLIndexT l = 0; l < nb_labels; ++l) {
      auto dist_row = dist.get_row(l);
      _distributions[l].compute_dist(q, dataset, dist_row);
    }
  }
};

}  // namespace ml

#endif  // INCLUDE_ML_CLASSIFIERS_BAYES_BAYES_CLASSIFIER_HPP
