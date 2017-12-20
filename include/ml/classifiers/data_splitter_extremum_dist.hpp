#ifndef INCLUDE_ML_CLASSIFIERS_DATA_SPLITTER_MIN_DIST_HPP
#define INCLUDE_ML_CLASSIFIERS_DATA_SPLITTER_MIN_DIST_HPP

#include "ml/classifiers/data_splitter.hpp"
#include "ml/classifiers/extremum_dist.hpp"

namespace ml
{

/**
 * @brief Abstract class regrouping the data_splitter and extremum_dist classes.
 *
 * @tparam DataT
 * @tparam LabelT
 * @tparam Compare minimize or maximize the computed distance
 */
template <class DataT, class LabelT, extremum_dist_compare Compare = LESS>
class data_splitter_extremum_dist : public data_splitter<DataT, LabelT>, public extremum_dist<DataT, LabelT, Compare> {
protected:
  inline virtual void train_setup_for_each_label(queue&) override {
    this->_predict_data_dim_assert = this->_data_dim;
  }
};

} // ml

#endif //INCLUDE_ML_CLASSIFIERS_DATA_SPLITTER_MIN_DIST_HPP
