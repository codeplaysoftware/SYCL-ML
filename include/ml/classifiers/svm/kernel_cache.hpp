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
#ifndef INCLUDE_ML_CLASSIFIERS_SVM_KERNEL_CACHE_HPP
#define INCLUDE_ML_CLASSIFIERS_SVM_KERNEL_CACHE_HPP

#include <list>
#include <set>
#include <unordered_map>
#include <vector>

#include "ml/classifiers/svm/svm_kernels.hpp"

namespace ml {

namespace detail {

/**
 * @brief Cache either the whole kernel matrix or only the last row used.
 *
 * @tparam KerFun
 * @tparam T
 */
template <class KerFun, class T>
class kernel_cache {
 public:
  kernel_cache(queue& q, const KerFun& ker, matrix_t<T>& x,
               const range<1>& data_rng, const nd_range<1>& ker_rng)
      : _q(q), _ker(ker), _x(x), _ker_diag_buf(data_rng, ker_rng) {
    // Compute the diagonal values of ker only once
    ker(q, x, _ker_diag_buf);
    auto m = access_ker_dim(x, 0);
    auto padded_m = to_pow2(m);
    auto pad_size_rng = get_optimal_nd_range(range<1>(padded_m - m), id<1>(m));
    if (pad_size_rng.get_global_linear_range() > 0) {
      sycl_memset(q, _ker_diag_buf, pad_size_rng);
    }
  }

  virtual vector_t<T> get_ker_row(SYCLIndexT row) = 0;

  inline vector_t<T>& get_ker_diag() { return _ker_diag_buf; }
  inline T get_ker_diag(SYCLIndexT row) {
    return _ker_diag_buf.read_to_host(row);
  }

 protected:
  queue& _q;
  const KerFun& _ker;

  matrix_t<T>& _x;
  vector_t<T> _ker_diag_buf;  // diagonal of kernel matrix
};

/**
 * @brief Compute the whole kernel matrix once
 *
 * If resulting matrix is too big, use kernel_cache_row instead.
 *
 * @tparam KerFun
 * @tparam T
 */
template <class KerFun, class T>
class kernel_cache_matrix : public kernel_cache<KerFun, T> {
 public:
  kernel_cache_matrix(queue& q, const KerFun& ker, matrix_t<T>& x,
                      const range<1>& data_rng, const nd_range<1>& ker_rng)
      : kernel_cache<KerFun, T>(q, ker, x, data_rng, ker_rng), _ker_mat() {
    auto nb_obs = access_ker_dim(x, 0);
    auto padded_nb_obs = get_device_constants()->pad_sub_buffer_size<T>(nb_obs);
    _ker_mat = matrix_t<T>(range<2>(nb_obs, nb_obs),
                           get_optimal_nd_range(nb_obs, padded_nb_obs));
    ker(q, x, x, _ker_mat);
  }

  inline virtual vector_t<T> get_ker_row(SYCLIndexT row) override {
    return _ker_mat.get_row(row);
  }

 private:
  matrix_t<T> _ker_mat;
};

/**
 * @brief Map a row index with its corresponding row in the kernel matrix.
 *
 * Should be used if the kernel matrix is too large.
 *
 * nb_cache_line is the maximum number of kernel line to cache.
 * It should be 2 for simple kernel (linear or polynomial) and grow bigger for
 * more complex kernels. The maximum size of the cache in byte is sizeof(T) * n
 * * nb_cache_line.
 *
 * @tparam KerFun
 * @tparam T
 */
template <class KerFun, class T>
class kernel_cache_row : public kernel_cache<KerFun, T> {
 public:
  kernel_cache_row(queue& q, const KerFun& ker, matrix_t<T>& x,
                   const range<1>& data_rng, const nd_range<1>& ker_rng,
                   SYCLIndexT nb_cache_line)
      : kernel_cache<KerFun, T>(q, ker, x, data_rng, ker_rng),
        _nb_cache_line(nb_cache_line),
        _ker_cache(),
        _cache_last_access() {}

  virtual vector_t<T> get_ker_row(SYCLIndexT row) override {
    auto it = _ker_cache.find(row);
    if (it != _ker_cache.end()) {
      // Move element row to the end
      auto row_it =
          std::find(_cache_last_access.begin(), _cache_last_access.end(), row);
      _cache_last_access.splice(_cache_last_access.end(), _cache_last_access,
                                row_it);
      return it->second;
    }

    _cache_last_access.push_back(row);
    if (_ker_cache.size() >= _nb_cache_line) {
      auto replace_row = _cache_last_access.front();
      _cache_last_access.pop_front();
      auto ker_row = std::move(_ker_cache[replace_row]);
      _ker_cache.erase(replace_row);
      this->_ker(this->_q, this->_x, row, ker_row);
      auto inserted_it = _ker_cache.insert(std::make_pair(row, ker_row));
      return inserted_it.first->second;
    }

    auto inserted_it = _ker_cache.emplace(
        std::piecewise_construct, std::forward_as_tuple(row),
        std::forward_as_tuple(this->_ker_diag_buf.data_range,
                              this->_ker_diag_buf.kernel_range));
    auto& ker_row = inserted_it.first->second;
    this->_ker(this->_q, this->_x, row, ker_row);
    return ker_row;
  }

 private:
  SYCLIndexT _nb_cache_line;
  std::unordered_map<SYCLIndexT, ml::vector_t<T>>
      _ker_cache;                            // Cached rows of kernel matrix
  std::list<SYCLIndexT> _cache_last_access;  // Indices of last used rows
};

}  // namespace detail

}  // namespace ml

#endif  // INCLUDE_ML_CLASSIFIERS_SVM_KERNEL_CACHE_HPP
