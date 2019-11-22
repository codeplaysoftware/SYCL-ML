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
/**
 * @file
 * @brief Include the Tensor module of Eigen using SYCL and define useful
 * aliases.
 */

#ifndef INCLUDE_ML_EIGEN_MY_EIGEN_HPP
#define INCLUDE_ML_EIGEN_MY_EIGEN_HPP

#include <unsupported/Eigen/CXX11/Tensor>

namespace ml {

using Eigen::Dynamic;

template <class T, int DIM, Eigen::StorageOptions DataLayout = Eigen::RowMajor>
using tensor_map_t = Eigen::TensorMap<Eigen::Tensor<T, DIM, DataLayout>>;

#define DEFINE_EIGEN_ALIAS(NAME, DIM)                       \
  template <class T, int DataLayout = Eigen::RowMajor>      \
  using eig_##NAME##_t = Eigen::Tensor<T, DIM, DataLayout>; \
  template <class T, int DataLayout = Eigen::RowMajor>      \
  using eig_##NAME##_map_t = Eigen::TensorMap<eig_##NAME##_t<T, DataLayout>>

/// @brief Generate \p eig_scalar_t and \p eig_scalar_map_t
DEFINE_EIGEN_ALIAS(scalar, 0);
/// @brief Generate \p eig_vec_t and \p eig_vec_map_t
DEFINE_EIGEN_ALIAS(vec, 1);
/// @brief Generate \p eig_mat_t and \p eig_mat_map_t
DEFINE_EIGEN_ALIAS(mat, 2);
/// @brief Generate \p eig_mats_t and \p eig_mats_map_t
DEFINE_EIGEN_ALIAS(mats, 3);

using eig_index_t = typename eig_mat_t<float>::Index;
using eig_dim_pair_t = typename eig_mat_t<float>::DimensionPair;
template <int DIM>
using eig_dsize_t = Eigen::DSizes<eig_index_t, DIM>;
template <int DIM>
using eig_dims_t = Eigen::array<eig_index_t, DIM>;

}  // namespace ml

#endif  // INCLUDE_ML_EIGEN_MY_EIGEN_HPP
