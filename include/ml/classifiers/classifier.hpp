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
#ifndef INCLUDE_ML_CLASSIFIERS_CLASSIFIER_HPP
#define INCLUDE_ML_CLASSIFIERS_CLASSIFIER_HPP

#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "ml/utils/common.hpp"

namespace ml {

/**
 * @brief Abstract class for all classifiers.
 *
 * @tparam DataT type of the dataset
 * @tparam LabelT type of the labels
 */
template <class DataT, class LabelT>
class classifier {
 public:
  using DataType = DataT;
  using LabelType = LabelT;

  /**
   * @brief Train the classifier.
   *
   * @param q
   * @param dataset
   * @param labels
   * @param nb_labels number of different labels, must be set if set_label_set
   * has not been called
   */
  virtual void train(queue& q, matrix_t<DataT>& dataset,
                     std::vector<LabelT>& labels, unsigned nb_labels = 0) = 0;

  /**
   * @brief Predict labels with the given observations.
   *
   * @param q
   * @param dataset
   * @return the predicted labels
   */
  virtual vector_t<LabelT> predict(queue& q, matrix_t<DataT>& dataset) = 0;

  /**
   * @brief Print statistics about the predicted labels.
   *
   * Compute and print the confusion matrix as well as the success rate,
   * precision, recall and F1-score.
   *
   * @param[in] predicted
   * @param[in] expected
   * @param nb_obs
   */
  static void print_score(
      const LabelT* predicted, const LabelT* expected, unsigned nb_obs,
      unsigned nb_labels,
      const std::unordered_map<LabelT, unsigned>& label_user_to_label_idx) {
    std::vector<unsigned> cm(nb_labels * nb_labels, 0);
    for (unsigned i = 0; i < nb_obs; ++i) {
      cm[label_user_to_label_idx.at(expected[i]) * nb_labels +
         label_user_to_label_idx.at(predicted[i])] += 1;
    }

    double success_rate = 0;
    double precision = 0;
    double recall = 0;
    double sum_row;
    double sum_col;
    double diag_val;
    for (unsigned i = 0; i < nb_labels; ++i) {
      sum_row = 0;
      sum_col = 0;
      for (unsigned j = 0; j < nb_labels; ++j) {
        sum_row += cm[i * nb_labels + j];
        sum_col += cm[j * nb_labels + i];
      }

      diag_val = cm[i * nb_labels + i];
      success_rate += diag_val;
      precision += diag_val / sum_row;
      recall += diag_val / sum_col;
    }

    success_rate /= nb_obs;
    precision /= nb_labels;
    recall /= nb_labels;

    double f1_score = 2 * (precision * recall) / (precision + recall);

    std::cout << "\nSuccess rate: " << success_rate * 100 << "%\n";
    std::cout << "Precision: " << precision * 100 << "%\n";
    std::cout << "Recall: " << recall * 100 << "%\n";
    std::cout << "F1-score: " << f1_score << "\n\n";

    std::cout << "Confusion matrix:\n";
    char prev_fill = std::cout.fill(' ');
    for (unsigned i = 0; i < nb_labels; ++i) {
      for (unsigned j = 0; j < nb_labels; ++j) {
        if (j < nb_labels - 1) {
          std::cout << std::left << std::setw(5) << cm[i * nb_labels + j]
                    << ' ';
        } else {
          std::cout << std::left << cm[i * nb_labels + j] << '\n';
        }
      }
    }
    std::cout.fill(prev_fill);
  }

  inline void print_score(const LabelT* predicted, const LabelT* expected,
                          unsigned nb_obs) {
    classifier<DataT, LabelT>::print_score(
        predicted, expected, nb_obs, get_nb_labels(), _label_user_to_label_idx);
  }

  virtual void load_from_disk(queue&) { assert(false); }
  virtual void save_to_disk(queue&) { assert(false); }

  inline unsigned get_nb_labels() const {
    return _host_label_idx_to_label_user.size();
  }

  /**
   * @brief Give the set of labels instead of computing it during the training.
   *
   * Optional function called before the training.
   *
   * @tparam LabelSet any type with a begin and end method used for copy
   * @param[in] label_set
   */
  template <class LabelSet>
  void set_label_set(const LabelSet& label_set) {
    std::copy(label_set.begin(), label_set.end(),
              std::back_inserter(_host_label_idx_to_label_user));
    assert(_host_label_idx_to_label_user.size() > 0);
    setup_host_label_idx_to_label_user();
  }

  /**
   * @brief Compute the list of indexes of each labels.
   *
   * @tparam HostLabelsT any type with a squared bracket accessor
   * @param host_labels
   * @param nb_labels number of different labels
   * @param nb_obs number of element in host_labels
   * @return labels_indices
   */
  template <class HostLabelsT>
  std::vector<std::vector<SYCLIndexT>> get_labels_indices(
      const HostLabelsT& host_labels, unsigned nb_labels, unsigned nb_obs) {
    std::vector<std::vector<SYCLIndexT>> labels_indices(nb_labels);
    for (unsigned i = 0; i < nb_obs; ++i) {
      labels_indices[this->_label_user_to_label_idx[host_labels[i]]].push_back(
          i);
    }
    return labels_indices;
  }

 protected:
  std::vector<LabelT> _host_label_idx_to_label_user;
  vector_t<LabelT> _label_idx_to_label_user;
  std::unordered_map<LabelT, unsigned> _label_user_to_label_idx;

  /**
   * @brief Fill _label_idx_to_label_user and _label_user_to_label_idx
   *
   * @tparam HostLabelT any type accessible with square brackets
   * @param[in] labels
   * @param nb_labels
   */
  template <class HostLabelT>
  void process_labels(const HostLabelT& host_labels, unsigned nb_labels) {
    // Labels have been set by the user beforehand
    if (!_label_user_to_label_idx.empty()) {
      return;
    }

    // Find all different labels
    _host_label_idx_to_label_user.reserve(nb_labels);
    for (unsigned i = 0; _host_label_idx_to_label_user.size() < nb_labels;
         ++i) {
      auto user_label = host_labels[i];
      auto it = std::find(_host_label_idx_to_label_user.begin(),
                          _host_label_idx_to_label_user.end(), user_label);
      if (it == _host_label_idx_to_label_user.end()) {
        _host_label_idx_to_label_user.push_back(user_label);
      }
    }
    std::sort(_host_label_idx_to_label_user.begin(),
              _host_label_idx_to_label_user.end());
    setup_host_label_idx_to_label_user();
  }

  /**
   * @brief Copy _host_label_idx_to_label_user to the device and to
   * _label_user_to_label_idx.
   */
  void setup_host_label_idx_to_label_user() {
    auto nb_labels = _host_label_idx_to_label_user.size();
    _label_idx_to_label_user = vector_t<LabelT>(
        const_cast<const LabelT*>(_host_label_idx_to_label_user.data()),
        range<1>(nb_labels));

    // Map label user back to label idx
    for (unsigned i = 0; i < nb_labels; ++i) {
      _label_user_to_label_idx[_host_label_idx_to_label_user[i]] = i;
    }
  }

  /**
   * @brief Check that nb_labels was given or set_label_set has been called.
   *
   * @param[in, out] nb_labels
   * @return true if nb_labels was given or set_label_set has been called
   */
  bool check_nb_labels(unsigned& nb_labels) {
    if (nb_labels == 0) {
      nb_labels = get_nb_labels();
      if (nb_labels == 0) {
        std::cerr << "Error: set_label_set must be called before training if "
                     "nb_labels is 0."
                  << std::endl;
        return false;
      }
    }
    if (nb_labels == 1) {
      std::cerr << "Error: must have more than one label." << std::endl;
      return false;
    }
    return true;
  }
};

}  // namespace ml

#endif  // INCLUDE_ML_CLASSIFIERS_CLASSIFIER_HPP
