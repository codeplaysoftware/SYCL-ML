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
#ifndef INCLUDE_ML_CLASSIFIERS_EM_EM_CLASSIFIER_HPP
#define INCLUDE_ML_CLASSIFIERS_EM_EM_CLASSIFIER_HPP

#include <vector>

#include "ml/classifiers/data_splitter_extremum_dist.hpp"

namespace ml {

/**
 * @brief Classifier using the EM algorithm.
 *
 * If used with log_model_per_label and log_gaussian_distribution this
 * implements a GMM. The GMM learn M models per label (instead of M=1 with a
 * GaussClassifier).
 *
 * @see log_model_per_label
 * @tparam LabelT
 * @tparam ModelPerLabelT type of the model to use
 */
template <class LabelT, class ModelPerLabelT>
class em_classifier
    : public data_splitter_extremum_dist<typename ModelPerLabelT::DataType,
                                         LabelT, GREATER> {
 public:
  using DataType = typename ModelPerLabelT::DataType;

  em_classifier(ModelPerLabelT model_impl = ModelPerLabelT())
      : data_splitter_extremum_dist<typename ModelPerLabelT::DataType, LabelT,
                                    GREATER>(),
        _model_impl(model_impl) {}

  virtual void load_from_disk(queue& q) override {
    for (unsigned i = 0; i < this->get_nb_labels(); ++i) {
      _ems[i].load_from_disk(q);
    }
  }

  virtual void save_to_disk(queue& q) override {
    for (unsigned i = 0; i < this->get_nb_labels(); ++i) {
      _ems[i].save_to_disk(q);
    }
  }

 protected:
  std::vector<ModelPerLabelT> _ems;

  virtual void train_setup_for_each_label(queue& q) override {
    data_splitter_extremum_dist<DataType, LabelT,
                                GREATER>::train_setup_for_each_label(q);

    for (unsigned i = 0; i < this->get_nb_labels(); ++i) {
      _ems.push_back(_model_impl);  // Copy model parameters
      _ems.back().set_idx(i);
    }
  }

  virtual inline void train_for_each_label(
      queue& q, unsigned label_idx, matrix_t<DataType>& act_data) override {
    _ems[label_idx].train(q, act_data);
  }

  virtual void compute_dist(queue& q, matrix_t<DataType>& dataset,
                            matrix_t<DataType>& dist) override {
    for (unsigned label_idx = 0; label_idx < this->get_nb_labels();
         ++label_idx) {
      auto dist_row = dist.get_row(label_idx);
      _ems[label_idx].compute_llk(q, dataset, dist_row);
    }
  }

 private:
  ModelPerLabelT _model_impl;
};

}  // namespace ml

#endif  // INCLUDE_ML_CLASSIFIERS_EM_EM_CLASSIFIER_HPP
