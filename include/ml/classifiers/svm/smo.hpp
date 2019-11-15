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
#ifndef INCLUDE_ML_CLASSIFIERS_SVM_SMO_HPP
#define INCLUDE_ML_CLASSIFIERS_SVM_SMO_HPP

#include "ml/classifiers/svm/kernel_cache.hpp"

namespace ml
{

namespace detail
{

class ml_smo_compute_obj_values;

/**
 * @brief Compute obj_values
 *
 * @tparam T
 * @param q
 * @param[in] i
 * @param[in] g_max
 * @param[in] eps
 * @param[in] gradient
 * @param[in] ker_diag_buffer
 * @param[in] ker_i_t
 * @param[out] obj_values
 */
template <class T>
void compute_obj_values(queue& q, SYCLIndexT i, T g_max, T eps, vector_t<T>& gradient,
                        vector_t<T>& ker_diag_buffer, vector_t<T>& ker_i_t, vector_t<T>& obj_values) {
  q.submit([&](handler& cgh) {
    auto g_acc = gradient.template get_access_1d<access::mode::read>(cgh);
    auto ker_diag_acc = ker_diag_buffer.template get_access_1d<access::mode::read>(cgh);
    auto ker_i_t_acc = ker_i_t.template get_access_1d<access::mode::read>(cgh);
    auto obj_values_acc = obj_values.template get_access_1d<access::mode::discard_write>(cgh);
    cgh.parallel_for<NameGen<0, ml_smo_compute_obj_values, T>>(obj_values.get_nd_range(), [=](nd_item<1> item) {
      auto t = item.get_global_id(0);
      auto b = g_max - g_acc(t);
      auto a = ker_diag_acc(i) + ker_diag_acc(t) - 2 * ker_i_t_acc(t);
      obj_values_acc(t) = (b > 0 && a > eps) ? -b * b / a : 0;
    });
  });
}

class ml_smo_find_extremum_idx;

/**
 * @brief Return the index of the min or max element of gradient validating cond.
 *
 * If multiple elements satisfying the condition are equal to the max or the min, it must return the last element.
 *
 * @tparam Compare should be std::less or std::greater
 * @tparam T
 * @param q
 * @param[in] cond whether to take into account the ith element
 * @param[in] gradient values to minimize or maximize
 * @param[in, out] in_indices the first iteration indices from 0 to m otherwise result from previous iteration
 * @param[out] out_indices resulting indices
 * @param search_rng
 * @param size_threshold_host threshold below which the search is done on the host
 * @param comp
 * @param[out] extremum_idx
 * @return true if an extremum could be found (i.e. if cond has at least one true)
 */
template <class Compare, class T>
bool find_extremum_idx(queue& q, vector_t<T>& cond, vector_t<T>& gradient, vector_t<uint32_t>& in_indices,
                       vector_t<uint32_t>& out_indices, const nd_range<1>& search_rng,
                       SYCLIndexT size_threshold_host, Compare comp, SYCLIndexT& extremum_idx) {
  auto search_size = search_rng.get_global_linear_range();
  if (search_size <= size_threshold_host) { // Search on the host starting from the end
    auto host_in_indices = in_indices.template get_access<access::mode::read>(range<1>(2 * search_size), id<1>(0));
    long k = 2 * search_size - 1;
    extremum_idx = -1;
    T extremum_gradient;
    while (k >= 0) { // Find last gradient which index holds the condition
      auto i = host_in_indices[k];
      if (cond.read_to_host(i)) {
        extremum_idx = i;
        extremum_gradient = gradient.read_to_host(i);
        break;
      }
      --k;
    }
    if (k < 0)
      return false;

    --k;
    while (k >= 0) { // Find extremum gradient which index holds the condition
      auto i = host_in_indices[k--];
      T grad_i = gradient.read_to_host(i);
      if (cond.read_to_host(i) && comp(grad_i, extremum_gradient)) {
        extremum_idx = i;
        extremum_gradient = grad_i;
      }
    }
    return cond.read_to_host(extremum_idx);
  }

  q.submit([&](handler& cgh) {
    auto cond_acc = cond.template get_access_1d<access::mode::read>(cgh);
    auto g_acc = gradient.template get_access_1d<access::mode::read>(cgh);
    auto in_indices_acc = in_indices.template get_access_1d<access::mode::read>(cgh);
    auto out_indices_acc = out_indices.template get_access_1d<access::mode::discard_write>(cgh);
    cgh.parallel_for<NameGen<0, ml_smo_find_extremum_idx, T, Compare>>(search_rng, [=](nd_item<1> item) {
      auto idx = item.get_global_id(0);
      auto i = in_indices_acc(2 * idx);
      auto j = in_indices_acc(2 * idx + 1);
      // If both conditions are true, use comp to select which index to return
      // else if one is true, return the corresponding index
      // if both are false return 0 (this index will be ignored at some point)
      out_indices_acc(idx) = (cond_acc(i) && cond_acc(j)) ? (comp(g_acc(i), g_acc(j)) ? i : j) :
                                                            (cond_acc(i) * i + cond_acc(j) * j);
    });
  });

  sycl_copy(q, out_indices, in_indices, 0, 0, search_size);
  return find_extremum_idx(q, cond, gradient, in_indices, out_indices, get_optimal_nd_range(search_size / 2),
                           size_threshold_host, comp, extremum_idx);
}

/**
 * @brief Select Working Set i.e. the pair (i,j) to optimize.
 *
 * @see smo
 * @tparam KerFun kernel function
 * @tparam T
 * @param q
 * @param[in] y
 * @param[in] gradient
 * @param[in] vec_cond_greater
 * @param[in] vec_cond_less
 * @param[in] tol
 * @param[in] eps
 * @param[in] start_search_indices
 * @param[in] start_search_rng
 * @param[in] find_size_threshold_host
 * @param[in, out] kernel_cache
 * @param[out] i
 * @param[out] j
 * @param[out] diff
 * @return whether a pair was successfully selected
 */
template <class KerFun, class T>
bool select_wss(queue& q, vector_t<T>& y, vector_t<T>& gradient, vector_t<T>& vec_cond_greater,
                vector_t<T>& vec_cond_less, T tol, T eps, vector_t<uint32_t>& start_search_indices,
                const nd_range<1>& start_search_rng, SYCLIndexT find_size_threshold_host,
                kernel_cache<KerFun, T>& kernel_cache, SYCLIndexT& i, SYCLIndexT& j, T& diff) {
  vector_t<uint32_t> tmp_in_search_indices(start_search_indices.data_range, start_search_indices.kernel_range);
  vector_t<uint32_t> buff_search_indices(start_search_rng);

  // Compute max(gradient) and its index i
  sycl_copy(q, start_search_indices, tmp_in_search_indices);
  if (!find_extremum_idx(q, vec_cond_greater, gradient, tmp_in_search_indices, buff_search_indices, start_search_rng,
                         find_size_threshold_host, std::greater<T>(), i)) {
    return false;
  }
  auto ker_i_t = kernel_cache.get_ker_row(i);
  T g_max = gradient.read_to_host(i);

  // Compute min(gradient)
  sycl_copy(q, start_search_indices, tmp_in_search_indices);
  SYCLIndexT g_min_idx;
  if (!find_extremum_idx(q, vec_cond_less, gradient, tmp_in_search_indices, buff_search_indices, start_search_rng,
                         find_size_threshold_host, std::less<T>(), g_min_idx)) {
    return false;
  }
  T g_min = gradient.read_to_host(g_min_idx);

  diff = g_max - g_min;
  if (diff < tol)
    return false;

  // Compute the index j of min(obj_values)
  vector_t<T> obj_values(y.data_range, y.kernel_range);
  compute_obj_values(q, i, g_max, eps, gradient, kernel_cache.get_ker_diag(), ker_i_t, obj_values);

  sycl_copy(q, start_search_indices, tmp_in_search_indices);
  return find_extremum_idx(q, vec_cond_less, obj_values, tmp_in_search_indices, buff_search_indices, start_search_rng,
                           find_size_threshold_host, std::less<T>(), j);
}

class ml_smo_update_gradient;

/**
 * @brief Update gradient.
 *
 * Formula is different from the paper:
 * \f$
 * G(t) = G(t) + y(i)*ker(t, i)*delta_ai - y(j)*ker(t, j)*delta_aj
 * \f$
 * \n became
 * \f$
 * G(t) = G(t) + y(i)*ker(i, t)*delta_ai - y(j)*ker(j, t)*delta_aj
 * \f$
 * \n because the kernel matrix is assumed to be symmetric.
 *
 * @tparam T
 * @param q
 * @param i
 * @param j
 * @param delta_ai signed by y(i)
 * @param delta_aj signed by y(j)
 * @param[in] ker_i_t
 * @param[in] ker_j_t
 * @param[out] gradient
 */
template <class T>
void update_gradient(queue& q, T delta_ai, T delta_aj, vector_t<T>& ker_i_t, vector_t<T>& ker_j_t,
                     vector_t<T>& gradient) {
  q.submit([&](handler& cgh) {
    auto ker_i_t_acc = ker_i_t.template get_access_1d<access::mode::read>(cgh);
    auto ker_j_t_acc = ker_j_t.template get_access_1d<access::mode::read>(cgh);
    auto g_acc = gradient.template get_access_1d<access::mode::read_write>(cgh);
    cgh.parallel_for<NameGen<0, ml_smo_update_gradient, T>>(gradient.get_nd_range(), [=](nd_item<1> item) {
      auto t = item.get_global_id(0);
      g_acc(t) += -ker_i_t_acc(t) * delta_ai - ker_j_t_acc(t) * delta_aj;
    });
  });
}

class ml_smo_copy_alphas;

/**
 * @brief Copy the non zeros of y .* alphas in sv_alphas
 *
 * @tparam T
 * @tparam IndexT
 * @param q
 * @param[in] alphas
 * @param[in] y
 * @param[in] sv_indices
 * @param[out] sv_alphas
 */
template <class T, class IndexT>
void copy_alphas(queue& q, vector_t<T>& alphas, vector_t<T>& y, vector_t<IndexT>& sv_indices,
                 vector_t<T>& sv_alphas) {
  q.submit([&](handler& cgh) {
    auto y_acc = y.template get_access_1d<access::mode::read>(cgh);
    auto old_a_acc = alphas.template get_access_1d<access::mode::read>(cgh);
    auto sv_idx_acc = sv_indices.template get_access_1d<access::mode::read>(cgh);
    auto new_a_acc = sv_alphas.template get_access_1d<access::mode::discard_write>(cgh);
    cgh.parallel_for<NameGen<0, ml_smo_copy_alphas, T, IndexT>>(sv_alphas.get_nd_range(), [=](nd_item<1> item) {
      auto row = item.get_global_id(0);
      auto sv_idx = sv_idx_acc(row);
      new_a_acc(row) = y_acc(sv_idx) * old_a_acc(sv_idx);
    });
  });
}

} // detail

/**
 * @brief Output of smo.
 * @see smo
 * @tparam T
 */
template <class T>
struct smo_out {
  matrix_t<T> svs;
  vector_t<T> alphas;
  T rho;
  size_t nb_iter;
};

/**
 * @brief Sequential Minimal Optimization
 *
 * The smo solves a quadratic problem by iteratively selecting a pair of variable and optimizing them.
 * This implementation is based on the paper
 * <em>Working Set Selection Using Second Order Information for Training Support Vector Machines</em>
 * (JLMR 6 2005) which is also used by libSVM.
 * The main difference is that the gradient G stores -y(t)*G(t) instead of the G(t) in the paper.
 *
 * @see svm
 *
 * @tparam KernelCacheT cache to use for kernel matrix
 * @tparam T
 * @param q
 * @param[in] x data of size mxn where m is the number of observation
 * @param[in] y labels must be a vector of size pow2(m) with -1 or 1 the first m elements and 0 after
 * @param c parameter for C-SVM
 * @param tol criteria for stopping condition
 * @param alpha_eps threshold above which alpha needs to be to be used as a weight of a support vector
 * @param max_nb_iter maximum number of iterations
 * @param kernel_cache
 * @return smo_out containing the support vectors svs, the alphas (multiplied by their respective labels) and
 *         the offset rho
 */
template <class KernelCacheT, class T>
smo_out<T> smo(queue& q, matrix_t<T>& x, vector_t<T>& y, T c, T tol, T alpha_eps, SYCLIndexT max_nb_iter,
              KernelCacheT kernel_cache) {
  auto m = access_ker_dim(x, 0);
  assert_eq(y.kernel_range.get_global_linear_range(), to_pow2(m));

  if (max_nb_iter == 0)
    max_nb_iter = std::max(10000000LU, m > INT_MAX/100 ? INT_MAX : 100*m);

  vector_t<T> alphas(y.data_range, y.kernel_range);
  vector_t<T> gradient(y.data_range, y.kernel_range);
  vector_t<uint32_t> start_search_indices(y.data_range, y.kernel_range);
  auto start_search_rng = get_optimal_nd_range(start_search_indices.kernel_range.get_global_linear_range() >> 1);

  // cond stores boolean only but type is T to avoid multiple cast at runtime
  vector_t<T> vec_cond_greater(y.data_range, y.kernel_range);
  vector_t<T> vec_cond_less(y.data_range, y.kernel_range);

  auto cond_greater = [c, alpha_eps](T y, T a) { return T((y > 0 && a < c) || (y < 0 && a > alpha_eps)); };
  auto cond_less = [c, alpha_eps](T y, T a) { return T((y > 0 && a > alpha_eps) || (y < 0 && a < c)); };

  // Compute initial cond
  vec_unary_op(q, y, vec_cond_greater, ml::functors::positive<T>());
  vec_unary_op(q, y, vec_cond_less, ml::functors::negative<T>());

  sycl_memset(q, alphas);
  sycl_copy(q, y, gradient);
  sycl_init_func_i(q, start_search_indices, start_search_indices.get_nd_range(), functors::identity<T>());
  SYCLIndexT find_size_threshold_host = std::min(m, 8LU);

  SYCLIndexT i;
  SYCLIndexT j;
  T diff;
  SYCLIndexT nb_iter = 0;
  T eps = 1E-8;
  while (nb_iter < max_nb_iter) {
    if (!detail::select_wss(q, y, gradient, vec_cond_greater, vec_cond_less, tol, eps, start_search_indices,
                            start_search_rng, find_size_threshold_host, kernel_cache, i, j, diff)) {
      break;
    }
    //std::cout << "#" << nb_iter << " i=" << i << " j=" << j << " diff=" << diff << std::endl;
    auto ker_i_t = kernel_cache.get_ker_row(i);
    auto ker_j_t = kernel_cache.get_ker_row(j);

    // Read values from device for i and j
    T ai = alphas.read_to_host(i);
    T aj = alphas.read_to_host(j);
    T yi = y.read_to_host(i);
    T yj = y.read_to_host(j);

    T a = std::max(kernel_cache.get_ker_diag(i) + kernel_cache.get_ker_diag(j) - 2 * ker_i_t.read_to_host(j), eps);
    T b = gradient.read_to_host(i) - gradient.read_to_host(j);

    // Update alphas i and j
    T old_ai = ai;
    T old_aj = aj;
    ai += yi * b / a;

    // Project alpha back to feasible region
    T s = yi*old_ai + yj*old_aj;
    ai = clamp(ai, T(0), c);
    aj = yj * (s - yi * ai);
    aj = clamp(aj, T(0), c);
    ai = yi * (s - yj * aj);
    alphas.write_from_host(i, ai);
    alphas.write_from_host(j, aj);

    // Update gradient
    T delta_ai = yi * (ai - old_ai);
    T delta_aj = yj * (aj - old_aj);

    // Shouldn't happen in theory but can because of precision issue
    assert(std::abs(delta_ai) >= eps);
    assert(std::abs(delta_aj) >= eps);

    detail::update_gradient(q, delta_ai, delta_aj, ker_i_t, ker_j_t, gradient);
    vec_cond_greater.write_from_host(i, cond_greater(yi, ai));
    vec_cond_greater.write_from_host(j, cond_greater(yj, aj));
    vec_cond_less.write_from_host(i, cond_less(yi, ai));
    vec_cond_less.write_from_host(j, cond_less(yj, aj));
    ++nb_iter;
  }

  if (nb_iter == max_nb_iter)
    std::cout << "Warning: maximum number of iteration reached, SVM may not have converged." << std::endl;

  // Compute host_sv_indices
  auto host_alphas = alphas.template get_access<access::mode::read>();
  std::vector<uint32_t> host_sv_indices;
  for (unsigned k = 0; k < m; ++k) {
    if (host_alphas[k] > alpha_eps)
      host_sv_indices.push_back(k);
  }
  auto nb_sv = host_sv_indices.size();
  assert(nb_sv > 0);

  // Compute rho
  T rho = 0;
  for (auto sv_idx : host_sv_indices)
    rho += gradient.read_to_host(sv_idx);
  rho /= nb_sv;

  // Copy the selected support vectors and y .* alphas
  vector_t<uint32_t> sv_indices(host_sv_indices.data(), range<1>(nb_sv));
  auto svs = split_by_index(q, x, sv_indices);
  vector_t<T> sv_alphas(sv_indices.data_range, sv_indices.kernel_range);
  detail::copy_alphas(q, alphas, y, sv_indices, sv_alphas);

  return smo_out<T>{svs, sv_alphas, rho, nb_iter};
}

} // ml

#endif //INCLUDE_ML_CLASSIFIERS_SVM_SMO_HPP
