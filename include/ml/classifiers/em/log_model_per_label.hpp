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
#ifndef INCLUDE_ML_CLASSIFIERS_EM_LOG_MODEL_PER_LABEL_HPP
#define INCLUDE_ML_CLASSIFIERS_EM_LOG_MODEL_PER_LABEL_HPP

#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <utility>

#ifdef __unix__
#include <cstdlib>  // Used to call system
#endif

#include "ml/math/mat_ops.hpp"
#include "ml/math/vec_ops.hpp"

namespace ml {

class ml_add_weights_tmp;

namespace detail {

class ml_em_add_weights;

template <class T>
event add_weights(queue& q, matrix_t<T>& plx, vector_t<T>& weights) {
  return q.submit([&plx, &weights](handler& cgh) {
    auto w_acc = weights.template get_access_1d<access::mode::read>(cgh);
    auto plx_acc = plx.template get_access_2d<access::mode::read_write>(cgh);
    cgh.parallel_for<NameGen<0, ml_em_add_weights, T>>(
        plx.get_nd_range(), [=](nd_item<2> item) {
          auto row = item.get_global_id(0);
          auto col = item.get_global_id(1);
          plx_acc(row, col) += cl::sycl::log(w_acc(row));
        });
  });
}

}  // namespace detail

/**
 * @brief Expectation-Maximisation algorithm.
 *
 * @todo the algorithm would be more stable if the distribution were initialized
 * with a K-means instead of random.
 *
 * @see log_gaussian_distribution
 * @tparam M number of instance of Distribution per label
 * @tparam Distribution must be the log of any probability distribution
 */
template <unsigned M, class Distribution>
class log_model_per_label {
 public:
  using DataType = typename Distribution::DataType;

  /**
   * @param max_nb_iter maximum number of expectation and maximization
   * iterations
   * @param diff_llk_eps threshold below which the training stops
   * @param weight_eps threshold below which the model gets re-initialized if
   * its weight drops below it
   * @param percent_rnd value in [0, 1] used to initialize the models
   */
  log_model_per_label(unsigned max_nb_iter = 20, DataType diff_llk_eps = 5E-3,
                      DataType weight_eps = 1E-5, DataType percent_rnd = 0.2)
      : _idx(-1),
        _max_nb_iter(max_nb_iter),
        _diff_llk_eps(diff_llk_eps),
        _weight_eps(weight_eps),
        _percent_rnd(percent_rnd),
        _nb_obs(-1),
        _data_dim(-1),
        _offset_noise(),
        _range_noise(),
        _weights(),
        _host_weights(),
        _distributions() {}

  /**
   * @brief The llk is the equivalent of the distance using all the models.
   *
   * @param q
   * @param[in] data
   * @param[out] llk
   */
  void compute_llk(queue& q, matrix_t<DataType>& data,
                   vector_t<DataType>& llk) {
    // nb_obs here can be different than from the training because this is
    // called to get the distance for the tests.
    auto nb_obs = access_data_dim(data, 0);
    auto padded_nb_obs = llk.get_kernel_range()[0];
    assert_less_or_eq(nb_obs, padded_nb_obs);

    matrix_t<DataType> plx = matrix_t<DataType>(range<2>(M, padded_nb_obs));
    vector_t<DataType> plx_max = vector_t<DataType>(range<1>(padded_nb_obs));

    compute_s_hat(q, data, llk, plx, plx_max);
    vec_binary_op(q, llk, plx_max, std::plus<DataType>());
  }

  /**
   * @brief Train the M models associated to a specific label.
   *
   * @param q
   * @param data train data of a same label
   * @param max_nb_iter maximum number of expectation and maximization
   * iterations
   * @param diff_llk_eps threshold below which the training stops
   * @param weight_eps threshold below which the model gets re-initialized if
   * its weight drops below it
   * @param percent_rnd value in [0, 1] used to initialize the models
   */
  void train(queue& q, matrix_t<DataType>& data) {
    _nb_obs = access_data_dim(data, 0);
    _data_dim = access_data_dim(data, 1);
    auto padded_nb_obs =
        get_device_constants()->pad_sub_buffer_size<DataType>(to_pow2(_nb_obs));
    auto padded_data_dim = to_pow2(_data_dim);

    range<1> nb_obs_rng(_nb_obs);
    auto padded_nb_obs_rng = get_optimal_nd_range(padded_nb_obs);
    range<1> data_dim_rng(_data_dim);
    auto padded_data_dim_rng = get_optimal_nd_range(padded_data_dim);
    range<2> data_dim_rng_d2(_data_dim, _data_dim);
    auto padded_data_dim_rng_d2 =
        get_optimal_nd_range(padded_data_dim, padded_data_dim);

    _offset_noise = vector_t<DataType>(data_dim_rng, padded_data_dim_rng);
    _range_noise = vector_t<DataType>(data_dim_rng, padded_data_dim_rng);

    _weights = vector_t<DataType>(range<1>(M));
    sycl_memset(q, _weights, DataType(1) / M);

    for (unsigned k = 0; k < M; ++k) {
      _distributions[k].init(data_dim_rng, padded_data_dim_rng, data_dim_rng_d2,
                             padded_data_dim_rng_d2);
    }

    vector_t<DataType> old_llk =
        vector_t<DataType>(nb_obs_rng, padded_nb_obs_rng);
    vector_t<DataType> new_llk =
        vector_t<DataType>(nb_obs_rng, padded_nb_obs_rng);
    vector_t<DataType> plx_max =
        vector_t<DataType>(nb_obs_rng, padded_nb_obs_rng);
    matrix_t<DataType> plx = matrix_t<DataType>(
        range<2>(M, _nb_obs), get_optimal_nd_range(M, padded_nb_obs));

    auto llk_padded_rng =
        get_optimal_nd_range(range<1>(padded_nb_obs - _nb_obs), id<1>(_nb_obs));
    std::function<void()> memset_llk = []() {};
    if (llk_padded_rng.get_global_linear_range() > 0) {
      memset_llk = [&]() { sycl_memset(q, new_llk, llk_padded_rng); };
    }

    {
      auto eig_data = sycl_to_eigen(data);
      auto eig_offset_noise = sycl_to_eigen(_offset_noise);
      auto eig_range_noise = sycl_to_eigen(_range_noise);
      eig_offset_noise.device() = eig_data.tensor().minimum(eig_dims_t<1>{0});
      eig_range_noise.device() = eig_data.tensor().maximum(eig_dims_t<1>{0}) -
                                 eig_offset_noise.tensor();
    }

    srand(time(NULL));
    for (unsigned k = 0; k < M; ++k) {
      SYCLIndexT rand_obs_idx = rand() % _nb_obs;
      auto data_sample = data.get_row(rand_obs_idx);
      _distributions[k].randomize(q, data_sample, _percent_rnd, _offset_noise,
                                  _range_noise);
    }

    DataType diff_llk;
    unsigned act_iter = 0;
    while (act_iter < _max_nb_iter) {
      step_e(q, data, plx, plx_max, new_llk);
      memset_llk();
      step_m(q, data, plx);

      if (act_iter > 0) {
        diff_llk = sycl_dist(q, old_llk, new_llk) / _nb_obs;
        assert_real(diff_llk);
        std::cout << "#" << _idx << " iter " << act_iter << " diff_llk "
                  << diff_llk << std::endl;
        if (diff_llk < _diff_llk_eps) {
          break;
        }
      }
      sycl_copy(q, new_llk, old_llk);
      ++act_iter;
    }
    std::cout << "\n";
  }

  void load_from_disk(queue& q) {
    auto prefix = get_file_prefix().second;
    load_array(q, _offset_noise, prefix + "_offset_noise");
    load_array(q, _range_noise, prefix + "_range_noise");
    load_array(q, _weights, prefix + "_weights");
    for (unsigned k = 0; k < M; ++k) {
      _distributions[k].load_from_disk(
          q, prefix + "_distrib_" + std::to_string(k));
    }
  }

  void save_to_disk(queue& q) {
#ifdef __unix__
    auto folder_prefix = get_file_prefix();
    int ret = system(("mkdir -p " + folder_prefix.first).c_str());
    if (ret == 0) {
      const auto& prefix = folder_prefix.second;
      save_array(q, _offset_noise, prefix + "_offset_noise");
      save_array(q, _range_noise, prefix + "_range_noise");
      save_array(q, _weights, prefix + "_weights");
      for (unsigned k = 0; k < M; ++k) {
        _distributions[k].save_to_disk(
            q, prefix + "_distrib_" + std::to_string(k));
      }
    }
#else
    std::cerr << "Error: Saving to disk is only supported on Unix for now."
              << std::endl;
    assert(false);
#endif
  }

  /**
   * @brief Set index to print, for debug
   *
   * @param idx
   */
  inline void set_idx(unsigned idx) { _idx = idx; }

 private:
  unsigned _idx;
  unsigned _max_nb_iter;
  DataType _diff_llk_eps;
  DataType _weight_eps;
  DataType _percent_rnd;
  SYCLIndexT _nb_obs;
  SYCLIndexT _data_dim;

  vector_t<DataType> _offset_noise;
  vector_t<DataType> _range_noise;
  vector_t<DataType> _weights;
  std::array<DataType, M> _host_weights;
  std::array<Distribution, M> _distributions;

  /**
   * @brief Compute s_hat, plx and plx_mat.
   *
   * s_hat is an intermediate variable used to compute both plx and llk.
   *
   * @param q
   * @param[in] data
   * @param[out] s_hat
   * @param[out] plx
   * @param[out] plx_max
   */
  void compute_s_hat(queue& q, matrix_t<DataType>& data,
                     vector_t<DataType>& s_hat, matrix_t<DataType>& plx,
                     vector_t<DataType>& plx_max) {
    for (unsigned k = 0; k < M; ++k) {
      auto plx_k = plx.get_row(k);
      _distributions[k].compute_dist(q, data, plx_k);
    }

    detail::add_weights(q, plx, _weights);

    // Divide plx by a constant so that its max becomes 1
    // Because we work in log space, substract so that its max become 0
    auto eig_plx = sycl_to_eigen(plx);
    {
      auto eig_plx_max = sycl_to_eigen(plx_max);
      eig_plx_max.device() = eig_plx.tensor().maximum(eig_dims_t<1>{0});
    }
    mat_vec_apply_op<COL>(q, plx, plx_max, std::minus<DataType>());

    auto eig_s_hat = sycl_to_eigen(s_hat);
    eig_s_hat.device() = eig_plx.tensor().exp().sum(eig_dims_t<1>{0}).log();
  }

  /**
   * @brief Expectation step.
   *
   * @param q
   * @param[in] data
   * @param[out] plx
   * @param[out] plx_max
   * @param[out] new_llk
   */
  void step_e(queue& q, matrix_t<DataType>& data, matrix_t<DataType>& plx,
              vector_t<DataType>& plx_max, vector_t<DataType>& new_llk) {
    compute_s_hat(q, data, new_llk, plx, plx_max);
    mat_vec_apply_op_data_rng<COL>(q, plx, new_llk,
                                   functors::exp_diff<DataType>());
    vec_binary_op(q, new_llk, plx_max, std::plus<DataType>());
  }

  /**
   * @brief Maximization step.
   *
   * @param q
   * @param[in] data
   * @param[in] plx
   */
  void step_m(queue& q, matrix_t<DataType>& data, matrix_t<DataType>& plx) {
    {
      auto eig_plx = sycl_to_eigen(plx);
      auto eig_weights = sycl_to_eigen(_weights);
      eig_weights.device() =
          eig_plx.tensor()
              .slice(eig_dsize_t<2>{0, 0},
                     eig_dsize_t<2>{
                         M, static_cast<eig_index_t>(plx.data_range[1])})
              .sum(eig_dims_t<1>{1});
    }

    auto copy_event =
        sycl_copy_device_to_host(q, _weights, _host_weights.data());
    copy_event.wait_and_throw();
    DataType act_w;
    for (unsigned k = 0; k < M; ++k) {
      act_w = _host_weights[k];
      if (act_w < _weight_eps) {
        // Re-randomize the k-th distribution (should not happen often)
        std::cout << "Warning: EM(" << _idx << ") weight(" << k
                  << ") is too small: " << act_w << "\n"
                  << "This may indicate that too many distributions (M) are "
                     "used for this problem."
                  << std::endl;
        unsigned max_idx = 0;
        for (unsigned i = 0; i < M; ++i) {
          if (_host_weights[i] > _host_weights[max_idx]) {
            max_idx = i;
          }
        }
        assert(max_idx != k);
        act_w = static_cast<DataType>(_nb_obs) / M;
        _weights.write_from_host(k, act_w);
        _distributions[k].randomize(q, _distributions[max_idx], _percent_rnd,
                                    _offset_noise, _range_noise);
      } else {
        auto plx_k = plx.get_row(k);
        _distributions[k].compute(q, data, act_w, plx_k);
      }
    }

    sycl_normalize(q, _weights, static_cast<DataType>(_nb_obs));
  }

  /**
   * @return the folder and th e prefix to use for saving and loading
   */
  std::pair<std::string, std::string> get_file_prefix() const {
    std::string folder =
        "em_" + std::to_string(M) + "_" + typeid(DataType).name();
    return std::make_pair(folder, folder + "/" + std::to_string(_idx));
  }
};

}  // namespace ml

#endif  // INCLUDE_ML_CLASSIFIERS_EM_LOG_MODEL_PER_LABEL_HPP
