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
#ifndef INCLUDE_ML_CLASSIFIERS_DATA_SPLITTER_HPP
#define INCLUDE_ML_CLASSIFIERS_DATA_SPLITTER_HPP

#include <algorithm>
#include <vector>

#include "ml/classifiers/classifier.hpp"
#include "ml/math/mat_ops.hpp"

namespace ml {

/**
 * @brief Abstract class of all classifiers needing to split the data for each
 * label.
 *
 * @tparam DataT
 * @tparam LabelT
 */
template <class DataT, class LabelT>
class data_splitter : public virtual classifier<DataT, LabelT> {
  template <int Index, typename... Details>
  using NameGenDS = NameGen<Index, data_splitter, Details..., DataT, LabelT>;

 public:
  /**
   * @brief Call train_for_each_label with a sub-dataset.
   *
   * Assumes labels are integers in [min(labels), max(labels)]
   *
   * @param q
   * @param dataset
   * @param labels
   * @param nb_labels number of different labels
   */
  virtual void train(queue& q, matrix_t<DataT>& dataset,
                     std::vector<LabelT>& host_labels,
                     unsigned nb_labels = 0) override {
    if (!this->check_nb_labels(nb_labels)) {
      return;
    }

    auto nb_obs = access_data_dim(dataset, 0);
    assert_eq(nb_obs, host_labels.size());

    _data_dim = access_data_dim(dataset, 1);
    _data_dim_pow2 = access_ker_dim(dataset, 1);

    this->process_labels(host_labels, nb_labels);

    // Compute indices for each labels
    auto labels_indices =
        this->get_labels_indices(host_labels, nb_labels, nb_obs);

    // Train for each label
    train_setup_for_each_label(q);
    for (unsigned i = 0; i < nb_labels; ++i) {
      const auto& act_labels_indices = labels_indices[i];
      auto act_data = split_by_index(q, dataset, act_labels_indices);
      train_for_each_label(q, i, act_data);
    }
  }

 protected:
  SYCLIndexT _data_dim;
  SYCLIndexT _data_dim_pow2;

  virtual void train_setup_for_each_label(queue&) {}
  virtual void train_for_each_label(queue& q, unsigned label_idx,
                                    matrix_t<DataT>& act_data) = 0;
};

}  // namespace ml

#endif  // INCLUDE_ML_CLASSIFIERS_DATA_SPLITTER_HPP
