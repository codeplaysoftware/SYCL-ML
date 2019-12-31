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
#ifndef INCLUDE_ML_CLASSIFIERS_MIN_DIST_HPP
#define INCLUDE_ML_CLASSIFIERS_MIN_DIST_HPP

#include "ml/classifiers/classifier.hpp"
#include "ml/math/mat_ops.hpp"

namespace ml {

/**
 * @brief Determine whether the classifier should minimize or maximize the
 * computed distance.
 */
enum extremum_dist_compare { LESS, GREATER };

namespace detail {

template <extremum_dist_compare Compare>
struct compare_detail;

template <>
struct compare_detail<LESS> {
  template <class T>
  using Op = std::less<T>;
  static constexpr int SIGN = -1;
};

template <>
struct compare_detail<GREATER> {
  template <class T>
  using Op = std::greater<T>;
  static constexpr int SIGN = 1;
};

}  // namespace detail

/**
 * @brief Abstract class of all classifiers minimizing or maximizing a distance.
 *
 * @tparam DataT
 * @tparam LabelT
 * @tparam Compare minimize or maximize the computed distance
 */
template <class DataT, class LabelT, extremum_dist_compare Compare = LESS>
class extremum_dist : public virtual classifier<DataT, LabelT> {
 protected:
  using Op = typename detail::compare_detail<Compare>::template Op<DataT>;

  static constexpr DataT SIGN =
      static_cast<DataT>(detail::compare_detail<Compare>::SIGN);
  SYCLIndexT _predict_data_dim_assert;

  virtual void compute_dist(queue& q, matrix_t<DataT>& dataset,
                            matrix_t<DataT>& dist) = 0;

  template <int Index, typename... Details>
  using NameGenED = NameGen<Index, Details..., DataT, LabelT, Op>;

 public:
  virtual vector_t<LabelT> predict(queue& q,
                                   matrix_t<DataT>& dataset) override {
    assert_eq(access_data_dim(dataset, 1), this->_predict_data_dim_assert);

    auto nb_labels = this->get_nb_labels();
    auto nb_obs = access_data_dim(dataset, 0);
    auto padded_nb_obs =
        get_device_constants()->pad_sub_buffer_size<DataT>(nb_obs);
    // The pad between nb_obs and padded_nb_obs can be left uninitialized.
    // It will produce random values in predicted_labels which shouldn't be
    // read.
    matrix_t<DataT> dist(range<2>(nb_labels, nb_obs),
                         get_optimal_nd_range(nb_labels, padded_nb_obs));
    compute_dist(q, dataset, dist);

    // Find extremum dist for each column
    vector_t<LabelT> predicted_labels(range<1>(nb_obs),
                                      get_optimal_nd_range(padded_nb_obs));
    q.submit([this, &dist, &predicted_labels, nb_labels](handler& cgh) {
      auto dist_acc = dist.template get_access_2d<access::mode::read>(cgh);
      auto label_idx_to_user_acc =
          this->_label_idx_to_label_user
              .template get_access_1d<access::mode::read>(cgh);
      auto predicted_labels_acc =
          predicted_labels.template get_access_1d<access::mode::discard_write>(
              cgh);
      cgh.parallel_for<NameGenED<0>>(
          predicted_labels.get_nd_range(), [=](nd_item<1> item) {
            auto col = item.get_global_id(0);
            auto extremum_index = 0;
            auto extremum_dist = dist_acc(extremum_index, col);
            for (unsigned i = 1; i < nb_labels; ++i) {  // Loop is small enough
              if (Op()(dist_acc(i, col), extremum_dist)) {
                extremum_dist = dist_acc(i, col);
                extremum_index = i;
              }
            }
            predicted_labels_acc(col) = label_idx_to_user_acc(extremum_index);
          });
    });

    return predicted_labels;
  }
};

}  // namespace ml

#endif  // INCLUDE_ML_CLASSIFIERS_MIN_DIST_HPP
