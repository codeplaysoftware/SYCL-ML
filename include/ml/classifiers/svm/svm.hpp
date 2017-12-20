#ifndef INCLUDE_ML_CLASSIFIERS_SVM_SVM_HPP
#define INCLUDE_ML_CLASSIFIERS_SVM_SVM_HPP

#include <vector>
#include <algorithm>
#include <memory>

#include "ml/classifiers/classifier.hpp"
#include "ml/classifiers/svm/kernel_cache.hpp"
#include "ml/classifiers/svm/smo.hpp"

namespace ml
{

class ml_svm_pad_data;
class ml_svm_predict_binary;
class ml_svm_predict;
class ml_svm_get_internal_labels;
class ml_svm_create_internal_labels;

/**
 * @brief SVM classifier.
 *
 * @tparam KernelType
 * @tparam LabelType
 */
template <class KernelType, class LabelType>
class svm : public classifier<typename KernelType::DataType, LabelType> {
public:
  using DataType = typename KernelType::DataType;

private:
  template <int Index, typename... Details>
  using NameGenSVM = NameGen<Index, svm, Details..., DataType, LabelType>;

  /**
   * @brief Internal labels are -1 and 1.
   * They could fit in type char but having them in DataType avoid the need to cast at kernel execution time.
   */
  using InternalLabelType = DataType;

public:
  /**
   * @brief Construct SMV with the parameters for smo.
   *
   * @param c
   * @param ker
   * @param nb_cache_line set to 0 to use the kernel_cache_matrix, must be 2 or greater to use the kernel_cache_row
   * @param tol
   * @param eps
   * @param max_nb_iter
   */
  explicit svm(DataType c, KernelType ker = KernelType(), SYCLIndexT nb_cache_line = 2,
               DataType tol = 1E-2, DataType eps = 1E-8, SYCLIndexT max_nb_iter = 0) :
      _c(c), _ker(ker), _tol(tol), _eps(eps), _nb_cache_line(nb_cache_line), _max_nb_iter(max_nb_iter),
      _data_dim(0), _data_dim_pow2(0), _nb_labels(0), _smo_outs()
  {
    assert(nb_cache_line != 1);
  }

  /**
   * @brief Train a binary SVM.
   *
   * Same result than the train method but avoid unnecessary copies.
   *
   * @see train
   * @param q
   * @param dataset size mxn
   * @param labels size m, assumed to have only 2 different values
   */
  void train_binary(queue& q, matrix_t<DataType>& dataset, vector_t<LabelType>& labels) {
    if (_nb_labels != 2) {  // skip setup_train if _nb_labels is already 2
      _nb_labels = 2;
      if (!setup_train(dataset, labels, _nb_labels))
        return;
    }

    assert(this->get_nb_labels() == 2);
    auto internal_labels = get_internal_labels(q, labels, this->_host_label_idx_to_label_user[0]);

    auto nb_obs_pow_2 = internal_labels.get_kernel_range()[0];
    if (access_ker_dim(dataset, 0) < nb_obs_pow_2 || access_ker_dim(dataset, 1) != _data_dim_pow2)
      dataset = pad_data(q, dataset, nb_obs_pow_2, _data_dim_pow2);

    _smo_outs.clear();
    push_back_smo(q, dataset, internal_labels);
  }

  /**
   * @brief Predict labels for a binary SVM.
   *
   * Same result than the predict method but optimized for 2 labels.
   *
   * @param q
   * @param dataset size mxn
   * @return m predicted labels
   */
  vector_t<LabelType> predict_binary(queue& q, matrix_t<DataType>& dataset) {
    assert_eq(access_data_dim(dataset, 1), _data_dim);
    assert(this->get_nb_labels() == 2);
    assert(_smo_outs.size() == 1);

    // dataset needs to be padded the same way it was during the training
    auto old_ker_nb_obs = access_ker_dim(dataset, 0);
    if (!is_pow2(old_ker_nb_obs) || access_ker_dim(dataset, 1) != _data_dim_pow2)
      dataset = pad_data(q, dataset, to_pow2(old_ker_nb_obs), _data_dim_pow2);

    auto& smo_out = _smo_outs.front();
    auto nb_obs = access_data_dim(dataset, 0);
    auto ker_nb_obs = access_ker_dim(dataset, 0);
    auto nb_sv = access_data_dim(smo_out.svs, 0);

    vector_t<LabelType> predictions(range<1>(nb_obs), get_optimal_nd_range(ker_nb_obs));
    if (nb_sv <= 1) {
      std::cerr << "Error: training has not been called or failed." << std::endl;
      return predictions;
    }

    predict_binary(q, dataset, predictions, smo_out);
    return predictions;
  }

  void train(queue& q, matrix_t<DataType>& dataset, vector_t<LabelType>& labels, unsigned nb_labels = 0) override {
    if (!setup_train(dataset, labels, nb_labels))
      return;

    _nb_labels = nb_labels;
    if (_nb_labels == 2) {
      train_binary(q, dataset, labels);
      return;
    }
    _smo_outs.clear();

    // Compute indices for each labels
    SYCLIndexT nb_obs = access_data_dim(dataset, 0);
    // Labels have already been copied to host by setup_train
    auto host_labels = labels.template get_access<access::mode::read>(id<1>(0), range<1>(0));
    auto indices_per_label = this->get_labels_indices(host_labels, nb_labels, nb_obs);

    unsigned max_act_data_nb_obs;
    std::vector<nd_range<1>> data_nd_ranges_per_label;
    {
      std::vector<unsigned> nb_obs_per_label;
      for (auto& vec : indices_per_label) {
        auto nb_obs_i = vec.size();
        nb_obs_per_label.push_back(nb_obs_i);
        data_nd_ranges_per_label.push_back(get_optimal_nd_range(nb_obs_i * _data_dim_pow2));
      }
      std::sort(nb_obs_per_label.begin(), nb_obs_per_label.end());
      max_act_data_nb_obs = nb_obs_per_label.back() + nb_obs_per_label[nb_obs_per_label.size() - 2];
    }

    // Pad data_dim dataset
    if (access_ker_dim(dataset, 1) != _data_dim_pow2)
      dataset = pad_data(q, dataset, nb_obs, _data_dim_pow2);

    // Split data
    std::vector<matrix_t<DataType>> data_per_label;
    for (auto& label_indices : indices_per_label)
      data_per_label.push_back(split_by_index(q, dataset, label_indices));

    // One vs One training
    unsigned nb_svms = nb_labels * (nb_labels - 1) / 2;
    _smo_outs.reserve(nb_svms);
    unsigned max_act_data_nb_obs_pow2 = to_pow2(max_act_data_nb_obs);
    matrix_t<DataType> act_data(get_optimal_nd_range(max_act_data_nb_obs_pow2, _data_dim_pow2));
    act_data.data_range[1] = _data_dim;
    vector_t<DataType> act_internal_labels(get_optimal_nd_range(max_act_data_nb_obs_pow2));

    for (unsigned i = 0; i < nb_labels - 1; ++i) {
      unsigned act_nb_obs_i = indices_per_label[i].size();
      const auto& data_nd_ranges_i = data_nd_ranges_per_label[i];

      // Copy data labeled i
      {
        auto act_data_sub_i = act_data.get_sub_buffer(id<1>(0), data_nd_ranges_i);
        sycl_copy(q, data_per_label[i], act_data_sub_i);
      }
      id<1> data_offset_i(data_nd_ranges_i.get_global_linear_range());

      for (unsigned j = i + 1; j < nb_labels; ++j) {
        unsigned act_nb_obs_j = indices_per_label[j].size();
        const auto& data_nd_ranges_j = data_nd_ranges_per_label[j];

        // Copy data labeled j
        {
          auto act_data_sub_j = act_data.get_sub_buffer(data_offset_i, data_nd_ranges_j);
          sycl_copy(q, data_per_label[j], act_data_sub_j);
        }

        create_internal_labels(q, act_nb_obs_i, act_nb_obs_j, act_internal_labels);

        act_internal_labels.data_range[0] = act_nb_obs_i + act_nb_obs_j;
        act_data.data_range[0] = act_internal_labels.data_range[0];

        std::cout << "Training (" << i << ", " << j << ") ..." << std::endl;
        // The padded nb_obs of act_data don't need to be initialized
        push_back_smo(q, act_data, act_internal_labels);
        std::cout << "Number support vectors: " << _smo_outs.back().alphas.data_range[0] << "\n" << std::endl;
      }
    }
  }

  vector_t<LabelType> predict(queue& q, matrix_t<DataType>& dataset) override {
    if (_nb_labels == 2)
      return predict_binary(q, dataset);
    assert_eq(access_data_dim(dataset, 1), _data_dim);

    // dataset needs to be padded the same way it was during the training
    auto old_ker_nb_obs = access_ker_dim(dataset, 0);
    if (!is_pow2(old_ker_nb_obs) || access_ker_dim(dataset, 1) != _data_dim_pow2)
      dataset = pad_data(q, dataset, to_pow2(old_ker_nb_obs), _data_dim_pow2);

    auto nb_obs = access_data_dim(dataset, 0);
    auto ker_nb_obs = access_ker_dim(dataset, 0);

    vector_t<LabelType> predictions(range<1>(nb_obs), get_optimal_nd_range(ker_nb_obs));
    if (access_data_dim(_smo_outs.front().svs, 0) <= 1) {
      std::cerr << "Error: training has not been called or failed." << std::endl;
      return predictions;
    }

    matrix_t<DataType> labels_counter(range<2>(_nb_labels, ker_nb_obs));
    sycl_memset(q, labels_counter);
    unsigned act_smo = 0;
    for (unsigned i = 0; i < _nb_labels - 1; ++i)
      for (unsigned j = i + 1; j < _nb_labels; ++j)
        predict_increment_counter(q, dataset, labels_counter, _smo_outs[act_smo++], i, j, nb_obs, ker_nb_obs,
                                  predictions.kernel_range);

    predict_reduce_counter(q, labels_counter, predictions);

    return predictions;
  }

  inline std::vector<smo_out<DataType>>& get_smo_outs() { return _smo_outs; }

private:
  const DataType _c;
  const KernelType _ker;
  const DataType _tol;
  const DataType _eps;
  const SYCLIndexT _nb_cache_line;
  const SYCLIndexT _max_nb_iter;

  SYCLIndexT _data_dim;
  SYCLIndexT _data_dim_pow2;

  unsigned _nb_labels;
  std::vector<smo_out<DataType>> _smo_outs;

  /**
   * @brief Common setup for train and train_binary.
   *
   * @see check_nb_labels
   * @param[in] dataset
   * @param[in] labels
   * @param[in, out] nb_labels
   * @return true if training can begin
   */
  bool setup_train(matrix_t<DataType>& dataset, vector_t<LabelType>& labels, unsigned& nb_labels) {
    if (!this->check_nb_labels(nb_labels))
      return false;

    SYCLIndexT nb_obs = access_data_dim(dataset, 0);
    assert_eq(nb_obs, labels.get_count());

    auto host_labels = labels.template get_access<access::mode::read>();
    this->process_labels(host_labels, nb_labels);

    _data_dim = access_data_dim(dataset, 1);
    _data_dim_pow2 = get_device_constants()->pad_sub_buffer_size<DataType>(access_ker_dim(dataset, 1));
    return true;
  }

  /**
   * @brief Pad the 2 dimensions of dataset to a power of 2.
   *
   * The first dimension is padded to match the size of the labels (speed up the computation of svm kernels).
   * Because the OvO method is used, the number of observation is relatively small so the pad is acceptable.
   * The second dimension is padded to be able to create a sub-buffer from a row and to speed up the
   * computation of svm kernels as well.
   *
   * @param q
   * @param[in] old_data
   * @param padded_obs_size new_obs_size
   * @return new_data
   */
  matrix_t<DataType> pad_data(queue& q, matrix_t<DataType>& old_data,
                              SYCLIndexT padded_nb_obs, SYCLIndexT padded_obs_size) {
    auto nb_obs = access_data_dim(old_data, 0);
    auto obs_size = access_data_dim(old_data, 1);
    matrix_t<DataType> new_data(range<2>(nb_obs, obs_size),
                                get_optimal_nd_range(padded_nb_obs, padded_obs_size));

    q.submit([&](handler& cgh) {
      auto old_acc = old_data.template get_access_2d<access::mode::read>(cgh);
      auto new_acc = new_data.template get_access_2d<access::mode::discard_write>(cgh);
      cgh.parallel_for<NameGenSVM<0, ml_svm_pad_data>>(new_data.get_nd_range(), [=](nd_item<2> item) {
        auto row = item.get_global(0);
        auto col = item.get_global(1);
        new_acc(row, col) = (row < nb_obs && col < obs_size) ? old_acc(row, col) : DataType(0);
      });
    });
    return new_data;
  }

  /**
   * @brief Cast the user labels to -1 or 1 label.
   *
   * @param q
   * @param[in] user_labels assume there are only 2 different labels
   * @param minus_one_label label to cast to -1, the other one becomes 1
   * @return internal_labels
   */
  vector_t<InternalLabelType> get_internal_labels(queue& q, vector_t<LabelType>& user_labels,
                                                  LabelType minus_one_label) {
    auto nb_user_labels = user_labels.data_range[0];
    vector_t<InternalLabelType> internal_labels(user_labels.data_range,
                                                get_optimal_nd_range(to_pow2(nb_user_labels)));
    q.submit([&](handler& cgh) {
      auto user_acc = user_labels.template get_access_1d<access::mode::read>(cgh);
      auto internal_acc = internal_labels.template get_access_1d<access::mode::discard_write>(cgh);
      cgh.parallel_for<NameGenSVM<0, ml_svm_get_internal_labels>>(internal_labels.get_nd_range(), [=](nd_item<1> item) {
        auto row = item.get_global(0);
        internal_acc(row) = (row < nb_user_labels) ? (user_acc(row) != minus_one_label) * 2 - 1 : InternalLabelType(0);
      });
    });
    return internal_labels;
  }

  /**
   * @brief Create vector with nb_obs_i -1s, nb_obs_j 1s and pad the rest with 0s.
   *
   * @param q
   * @param nb_obs_i
   * @param nb_obs_j
   * @param[out] internal_labels
   */
  void create_internal_labels(queue& q, SYCLIndexT nb_obs_i, SYCLIndexT nb_obs_j,
                              vector_t<InternalLabelType>& internal_labels) {
    q.submit([&](handler& cgh) {
      auto internal_acc = internal_labels.template get_access_1d<access::mode::discard_write>(cgh);
      cgh.parallel_for<NameGenSVM<0, ml_svm_create_internal_labels>>(internal_labels.get_nd_range(), [=](nd_item<1> item) {
        auto row = item.get_global(0);
        internal_acc(row) = row < nb_obs_i ? InternalLabelType(-1) :
                           (row < nb_obs_i + nb_obs_j ? InternalLabelType(1) : InternalLabelType(0));
      });
    });
  }

  /**
   * @brief Compute the internal label using smo_out and convert it to user label.
   *
   * @param q
   * @param[in] dataset
   * @param[out] predictions
   * @param[in] smo_out
   */
  void predict_binary(queue& q, matrix_t<DataType>& dataset, vector_t<LabelType>& predictions,
                      smo_out<DataType>& smo_out) {
    auto nb_obs = access_data_dim(dataset, 0);
    auto ker_nb_obs = access_ker_dim(dataset, 0);
    auto nb_sv = access_data_dim(smo_out.svs, 0);
    matrix_t<DataType> ker_values(range<2>(nb_sv, nb_obs), get_optimal_nd_range(nb_sv, ker_nb_obs));
    _ker(q, smo_out.svs, dataset, ker_values);

    auto minus_one_label = this->_host_label_idx_to_label_user[0];
    auto one_label = this->_host_label_idx_to_label_user[1];
    auto rho = smo_out.rho;
    q.submit([&](handler& cgh) {
      auto ker_values_acc = ker_values.template get_access_2d<access::mode::read>(cgh);
      auto a_acc = smo_out.alphas.template get_access_1d<access::mode::read>(cgh);
      auto pred_acc = predictions.template get_access_1d<access::mode::discard_write>(cgh);
      cgh.parallel_for<NameGenSVM<0, ml_svm_predict_binary>>(predictions.get_nd_range(), [=](nd_item<1> item) {
        auto col = item.get_global(0);
        DataType sum = rho;
        for (SYCLIndexT i = 0; i < nb_sv; ++i)  // Loop is (usually) small enough
          sum += a_acc(i) * ker_values_acc(i, col);
        pred_acc(col) = (sum < 0) ? minus_one_label : one_label;
      });
    });
  }

  /**
   * @brief For each observation, count how many times each label was predicted.
   *
   * @param q
   * @param[in] dataset
   * @param[in, out] labels_counter
   * @param[in] smo_out
   * @param i
   * @param j
   * @param nb_obs
   * @param ker_nb_obs
   * @param kernel_range
   */
  void predict_increment_counter(queue& q, matrix_t<DataType>& dataset, matrix_t<DataType>& labels_counter,
                                 smo_out<DataType>& smo_out, unsigned i, unsigned j, SYCLIndexT nb_obs,
                                 SYCLIndexT ker_nb_obs, const nd_range<1>& kernel_range) {
    auto nb_sv = access_data_dim(smo_out.svs, 0);
    matrix_t<DataType> ker_values(range<2>(nb_sv, nb_obs), get_optimal_nd_range(nb_sv, ker_nb_obs));
    _ker(q, smo_out.svs, dataset, ker_values);
    auto rho = smo_out.rho;

    q.submit([&](handler& cgh) {
      auto ker_values_acc = ker_values.template get_access_2d<access::mode::read>(cgh);
      auto a_acc = smo_out.alphas.template get_access_1d<access::mode::read>(cgh);
      auto counter_acc = labels_counter.template get_access_2d<access::mode::read_write>(cgh);
      cgh.parallel_for<NameGenSVM<0, ml_svm_predict>>(kernel_range, [=](nd_item<1> item) {
        auto col = item.get_global(0);
        DataType sum = rho;
        for (SYCLIndexT k = 0; k < nb_sv; ++k)  // Loop is (usually) small enough
          sum += a_acc(k) * ker_values_acc(k, col);
        auto prediction = (sum < 0) ? i : j;
        counter_acc(prediction, col) += 1;
      });
    });
  }

  /**
   * @brief For each observation, select the label predicted the most.
   *
   * @param q
   * @param[in] labels_counter
   * @param[out] predictions
   */
  void predict_reduce_counter(queue& q, matrix_t<DataType>& labels_counter, vector_t<LabelType>& predictions) {
    auto nb_labels = _nb_labels;
    q.submit([&](handler& cgh) {
      auto counter_acc = labels_counter.template get_access_2d<access::mode::read>(cgh);
      auto label_idx_to_label_user_acc = this->_label_idx_to_label_user.template get_access_1d<access::mode::read>(cgh);
      auto pred_acc = predictions.template get_access_1d<access::mode::discard_write>(cgh);
      cgh.parallel_for<NameGenSVM<1, ml_svm_predict>>(predictions.get_nd_range(), [=](nd_item<1> item) {
        auto col = item.get_global(0);
        SYCLIndexT max_idx = 0;
        for (SYCLIndexT i = 1; i < nb_labels; ++i) {  // Loop is small enough
          if (counter_acc(i, col) > counter_acc(max_idx, col))
            max_idx = i;
        }
        pred_acc(col) = label_idx_to_label_user_acc(max_idx);
      });
    });
  }

  void push_back_smo(queue& q, matrix_t<DataType>& dataset, vector_t<InternalLabelType>& internal_labels) {
    if (_nb_cache_line == 0) {
      _smo_outs.push_back(smo(q, dataset, internal_labels, _c, _tol, _eps, _max_nb_iter,
        detail::kernel_cache_matrix<KernelType, DataType>(q, _ker, dataset,
                                                        internal_labels.data_range, internal_labels.kernel_range)));
    }
    else {
      _smo_outs.push_back(smo(q, dataset, internal_labels, _c, _tol, _eps, _max_nb_iter,
        detail::kernel_cache_row<KernelType, DataType>(q, _ker, dataset,
                                                     internal_labels.data_range, internal_labels.kernel_range,
                                                     _nb_cache_line)));
    }
  }
};

} // ml

#endif //INCLUDE_ML_CLASSIFIERS_SVM_SVM_HPP
