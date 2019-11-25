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
#ifndef INCLUDE_ML_PREPROCESS_PCA_HPP
#define INCLUDE_ML_PREPROCESS_PCA_HPP

#include "ml/math/cov.hpp"
#include "ml/math/mat_ops.hpp"
#include "ml/math/svd.hpp"

namespace ml {

namespace detail {

class ml_pca_svd_copy_v;

template <class T>
event copy_eigenvectors(queue& q, vector_t<SYCLIndexT>& indices,
                        matrix_t<T>& in_v, matrix_t<T>& out_v) {
  return q.submit([&indices, &in_v, &out_v](handler& cgh) {
    auto in_acc = in_v.template get_access_2d<access::mode::read>(cgh);
    auto indices_acc = indices.template get_access_1d<access::mode::read>(cgh);
    auto out_acc =
        out_v.template get_access_2d<access::mode::discard_write>(cgh);
    cgh.parallel_for<NameGen<0, ml_pca_svd_copy_v, T>>(
        out_v.get_nd_range(), [=](nd_item<2> item) {
          auto row = item.get_global_id(0);
          auto col = item.get_global_id(1);
          out_acc(row, col) = in_acc(indices_acc(row), col);
        });
  });
}

}  // namespace detail

/**
 * @brief Arguments given to PCA
 *
 * auto_load: whether to load the basis vectors from the disk if the expected
 * file is present, defaults to true save: whether to save the basis vectors to
 * the disk if the PCA was not loaded, defaults to true min_nb_vecs: minimum
 * number of vectors to use, defaults to 0 which disable this constraint
 * keep_percent: minimum "amount of information" to keep in range [0; 1]. 0
 * disables the PCA and 1 keeps as many vectors as possible. Defaults to 1
 * scale_factor: factor applied when computing the PCA, a higher value yields
 * more precision but is slower. Defaults to 1
 *
 */
template <class T>
struct pca_args {
  pca_args()
      : auto_load(true),
        save(true),
        min_nb_vecs(0),
        keep_percent(1.f),
        scale_factor(T(1)) {}

  bool auto_load;
  bool save;
  SYCLIndexT min_nb_vecs;
  float keep_percent;
  T scale_factor;
};

/**
 * @brief Center the data and compute the principal components.
 *
 * Assumes the number of rows is the number of observations and the size of an
 * observation is a power of 2. Uses the svd to compute the eigenpairs. V =
 * pca(X) gives the eigenvectors so that Y = cX * V' where cX is the data
 * centered and Y is the new data with a smaller size of observation.
 *
 * @see apply_pca_svd
 * @tparam T
 * @param q
 * @param[in] data
 * @param[out] data_avg
 * @param pca_args @see struct pca_args
 * @return the eigenvectors V
 */
template <class T>
matrix_t<T> pca_svd(queue& q, matrix_t<T>& data, vector_t<T>& data_avg,
                    const pca_args<T>& pca_args) {
  avg(q, data, data_avg);
  center_data<COL>(q, data, data_avg);
  auto data_dim = access_data_dim(data, 1);
  auto data_dim_pow2 = access_ker_dim(data, 1);

  // For precision, scale data to change the eigenvalues but not the
  // eigenvectors
  auto scaled_data = matrix_t<T>(data.data_range, data.kernel_range);
  if (pca_args.scale_factor != T(1)) {
    vec_unary_op(q, data, scaled_data,
                 functors::partial_binary_op<T, std::multiplies<T>>(
                     pca_args.scale_factor));
  }

  matrix_t<T> cov_matrix(range<2>(data_dim, data_dim),
                         get_optimal_nd_range(data_dim_pow2, data_dim_pow2));
  cov(q, scaled_data, cov_matrix);
  SYCLIndexT estimated_nb_vecs = data_dim;
  auto svd_out = svd<false, true, true>(q, cov_matrix, estimated_nb_vecs);

  if (pca_args.keep_percent >= 1) {
    return svd_out.V;
  }

  // Sort indices of l in descending order
  std::vector<SYCLIndexT> host_indices(estimated_nb_vecs);
  std::iota(begin(host_indices), end(host_indices), 0);
  auto& host_l = svd_out.L;
  std::sort(
      begin(host_indices), end(host_indices),
      [&](SYCLIndexT i1, SYCLIndexT i2) { return host_l[i1] > host_l[i2]; });

  // Compute nb_vecs needed to reach keep_percent
  SYCLIndexT nb_vecs = 0;
  float act_percent = 0;
  for (; nb_vecs < estimated_nb_vecs && act_percent < pca_args.keep_percent;
       ++nb_vecs) {
    act_percent += host_l[host_indices[nb_vecs]] / svd_out.eig_vals_sum;
  }
  nb_vecs = std::max(nb_vecs, pca_args.min_nb_vecs);
  std::cout << "Keeping " << nb_vecs << " vectors" << std::endl;
  assert(nb_vecs > 0);

  // Copy the eigenvectors with the highest eigenvalue
  vector_t<SYCLIndexT> sycl_indices(host_indices.data(), range<1>(nb_vecs));
  matrix_t<T> V(range<2>(nb_vecs, data_dim),
                get_optimal_nd_range(nb_vecs, data_dim_pow2));
  detail::copy_eigenvectors(q, sycl_indices, svd_out.V, V);

  return V;
}

}  // namespace ml

#endif  // INCLUDE_ML_PREPROCESS_PCA_HPP
