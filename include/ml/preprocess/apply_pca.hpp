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
#ifndef INCLUDE_ML_PREPROCESS_APPLY_PCA_HPP
#define INCLUDE_ML_PREPROCESS_APPLY_PCA_HPP

#include <cassert>
#include <fstream>

#include "pca.hpp"

namespace ml {

/**
 * @brief Helper to compute and apply the PCA from a training set and applying
 * it on a test set.
 *
 * @see pca_svd
 * @tparam T
 */
template <class T>
class apply_pca {
 public:
  apply_pca()
      : _enable_pca(true),
        _nb_vec_computed(0),
        _data_avg(range<1>()),
        _eigenvectors(range<2>()) {}

  /**
   * @brief Either load the eigenvectors or compute them and apply the PCA to
   * the given data.
   *
   * @param q
   * @param[in, out] data this matrix has been centered after this call
   * @param pca_args @see struct pca_args
   * @return the new data
   */
  matrix_t<T> compute_and_apply(queue& q, matrix_t<T>& data,
                                const pca_args<T>& pca_args) {
    _enable_pca = pca_args.keep_percent > 0;
    if (!_enable_pca) {
      return data;
    }

    auto data_dim = access_data_dim(data, 1);
    auto data_dim_pow2 = access_ker_dim(data, 1);

    _data_avg =
        vector_t<T>(range<1>(data_dim), get_optimal_nd_range(data_dim_pow2));

    std::string load_filename = get_filename(
        data_dim_pow2, pca_args.min_nb_vecs, pca_args.scale_factor);
    if (pca_args.auto_load && file_exists(load_filename)) {
      // avg and center_data would have been called by pca_svd otherwise
      avg(q, data, _data_avg);
      center_data<COL>(q, data, _data_avg);

      _nb_vec_computed = pca_args.min_nb_vecs;
      _eigenvectors = matrix_t<T>(
          range<2>(pca_args.min_nb_vecs, data_dim),
          get_optimal_nd_range(pca_args.min_nb_vecs, data_dim_pow2));
      load_array(q, _eigenvectors, load_filename);
    } else {
      std::cout << "Computing PCA..." << std::endl;
      _eigenvectors = pca_svd(q, data, _data_avg, pca_args);
      _nb_vec_computed = access_data_dim(_eigenvectors, 0);
      if (pca_args.save) {
        save_array(q, _eigenvectors,
                   get_filename(data_dim_pow2, _nb_vec_computed,
                                pca_args.scale_factor));
      }
    }

    matrix_t<T> new_data =
        matrix_t<T>(range<2>(access_data_dim(data, 0), _nb_vec_computed));
    mat_mul<LIN, TR>(q, data, _eigenvectors, new_data);
    return new_data;
  }

  /**
   * @brief Apply the PCA to a dataset from previously computed eigenvectors and
   * data_avg.
   *
   * @param q
   * @param[in, out] data this matrix has been centered after this call
   * @return the new data
   */
  matrix_t<T> apply(queue& q, matrix_t<T>& data) {
    if (!_enable_pca) {
      return data;
    }

    assert(_nb_vec_computed != 0);
    matrix_t<T> new_data(range<2>(access_data_dim(data, 0), _nb_vec_computed));
    center_data<COL>(q, data, _data_avg);
    mat_mul<LIN, TR>(q, data, _eigenvectors, new_data);
    return new_data;
  }

 private:
  bool _enable_pca;
  SYCLIndexT _nb_vec_computed;
  vector_t<T> _data_avg;
  matrix_t<T> _eigenvectors;

  /**
   * @brief Get the filename used for saving and loading eigenvectors.
   *
   * @param data_dim_pow2
   * @param nb_vec
   * @return the filename
   */
  inline std::string get_filename(SYCLIndexT data_dim_pow2, SYCLIndexT nb_vec,
                                  T svd_factor) {
    std::stringstream ss;
    ss << "pca_" << nb_vec << "_" << data_dim_pow2 << "_" << svd_factor << "_"
       << typeid(T).name();
    return ss.str();
  }

  /**
   * @param filename
   * @return true if filename exists (and is not locked)
   */
  inline bool file_exists(const std::string& filename) {
    std::ifstream ifs(filename);
    return ifs.good();
  }
};

}  // namespace ml

#endif  // INCLUDE_ML_PREPROCESS_APPLY_PCA_HPP
