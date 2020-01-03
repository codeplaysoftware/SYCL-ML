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
#include "ml/math/functors.hpp"

namespace ml {

namespace detail {

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
 * @return A SYCL event corresponding to the submitted operation
 */
template <class T>
event compute_obj_values(queue& q, SYCLIndexT i, T g_max, T eps,
                         vector_t<T>& gradient, vector_t<T>& ker_diag_buffer,
                         vector_t<T>& ker_i_t, vector_t<T>& obj_values) {
  return q.submit([&gradient, &ker_diag_buffer, &ker_i_t, &obj_values, i, g_max,
                   eps](handler& cgh) {
    auto g_acc = gradient.template get_access_1d<access::mode::read>(cgh);
    auto ker_diag_acc =
        ker_diag_buffer.template get_access_1d<access::mode::read>(cgh);
    auto ker_i_t_acc = ker_i_t.template get_access_1d<access::mode::read>(cgh);
    auto obj_values_acc =
        obj_values.template get_access_1d<access::mode::discard_write>(cgh);
    cgh.parallel_for<NameGen<0, ml_smo_compute_obj_values, T>>(
        obj_values.get_nd_range(), [=](nd_item<1> item) {
          auto t = item.get_global_id(0);
          auto b = g_max - g_acc(t);
          auto a = ker_diag_acc(i) + ker_diag_acc(t) - 2 * ker_i_t_acc(t);
          obj_values_acc(t) = (b > 0 && a > eps) ? -b * b / a : 0;
        });
  });
}

/**
 * @brief Return the argmax of the data validating cond.
 *
 * @tparam T
 * @param q (unused)
 * @param[in] cond whether to take into account the ith element
 * @param[in] data
 * @param[out] extremum_idx
 * @return true if an extremum could be found i.e. cond has at least one
 * true which does not correspond to a -inf
 */
template <class T, class EigenScalar>
bool argmax_cond(queue&, vector_t<T>& cond, vector_t<T>& data,
                 EigenScalar& eig_scalar, SYCLIndexT& argmax) {
  T extremum = -std::numeric_limits<T>::infinity();
  auto eig_cond = sycl_to_eigen(cond);
  auto eig_data = sycl_to_eigen(data);
  auto eig_extremum = eig_data.tensor().constant(extremum);
  auto extremum_cond =
      eig_cond.tensor().select(eig_data.tensor(), eig_extremum);
  eig_scalar.device() = extremum_cond.argmax().template cast<SYCLIndexT>();
  get_eigen_device().memcpyDeviceToHost(&argmax, eig_scalar.ptr(),
                                        sizeof(SYCLIndexT));
  return data.read_to_host(argmax) != extremum;
}

/**
 * @brief Return the argmin of the data validating cond.
 *
 * @tparam T
 * @param q (unused)
 * @param[in] cond whether to take into account the ith element
 * @param[in] data
 * @param[out] extremum_idx
 * @return true if an extremum could be found i.e. cond has at least one
 * true which does not correspond to a +inf
 */
template <class T, class EigenScalar>
bool argmin_cond(queue&, vector_t<T>& cond, vector_t<T>& data,
                 EigenScalar& eig_scalar, SYCLIndexT& argmin) {
  T extremum = std::numeric_limits<T>::infinity();
  auto eig_cond = sycl_to_eigen(cond);
  auto eig_data = sycl_to_eigen(data);
  auto eig_extremum = eig_data.tensor().constant(extremum);
  auto extremum_cond =
      eig_cond.tensor().select(eig_data.tensor(), eig_extremum);
  eig_scalar.device() = extremum_cond.argmin().template cast<SYCLIndexT>();
  get_eigen_device().memcpyDeviceToHost(&argmin, eig_scalar.ptr(),
                                        sizeof(SYCLIndexT));
  return data.read_to_host(argmin) != extremum;
}

/**
 * @brief Select Working Set i.e. the pair (i,j) to optimize.
 *
 * @see smo
 * @tparam KerFun kernel function
 * @tparam T
 * @tparam EigenScalar device buffer used for argmax_cond and argmin_cond
 * @param q
 * @param obj_values temporary buffer the size of the labels
 * @param[in] gradient
 * @param[in] vec_cond_greater
 * @param[in] vec_cond_less
 * @param[in] tol
 * @param[in] eps
 * @param[in, out] kernel_cache
 * @param[out] i
 * @param[out] j
 * @param[out] diff
 * @return whether a pair was successfully selected
 */
template <class KerFun, class T, class EigenScalar>
bool select_wss(queue& q, vector_t<T>& obj_values, vector_t<T>& gradient,
                vector_t<T>& vec_cond_greater, vector_t<T>& vec_cond_less,
                T tol, T eps, EigenScalar& eig_scalar,
                kernel_cache<KerFun, T>& kernel_cache, SYCLIndexT& i,
                SYCLIndexT& j, T& diff) {
  // Compute i=argmax(gradient) validating vec_cond_greater
  if (!argmax_cond(q, vec_cond_greater, gradient, eig_scalar, i)) {
    return false;
  }
  auto ker_i_t = kernel_cache.get_ker_row(i);
  T g_max = gradient.read_to_host(i);

  // Compute min(gradient) validating vec_cond_less
  SYCLIndexT g_min_idx;
  if (!argmin_cond(q, vec_cond_less, gradient, eig_scalar, g_min_idx)) {
    return false;
  }
  T g_min = gradient.read_to_host(g_min_idx);

  diff = g_max - g_min;
  if (diff < tol) {
    return false;
  }

  // Compute j=argmin(obj_values) validating vec_cond_less
  compute_obj_values(q, i, g_max, eps, gradient, kernel_cache.get_ker_diag(),
                     ker_i_t, obj_values);

  return argmin_cond(q, vec_cond_less, obj_values, eig_scalar, j);
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
 * @return A SYCL event corresponding to the submitted operation
 */
template <class T>
event update_gradient(queue& q, T delta_ai, T delta_aj, vector_t<T>& ker_i_t,
                      vector_t<T>& ker_j_t, vector_t<T>& gradient) {
  return q.submit([&ker_i_t, &ker_j_t, &gradient, delta_ai,
                   delta_aj](handler& cgh) {
    auto ker_i_t_acc = ker_i_t.template get_access_1d<access::mode::read>(cgh);
    auto ker_j_t_acc = ker_j_t.template get_access_1d<access::mode::read>(cgh);
    auto g_acc = gradient.template get_access_1d<access::mode::read_write>(cgh);
    cgh.parallel_for<NameGen<0, ml_smo_update_gradient, T>>(
        gradient.get_nd_range(), [=](nd_item<1> item) {
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
 * @return A SYCL event corresponding to the submitted operation
 */
template <class T, class IndexT>
event copy_alphas(queue& q, vector_t<T>& alphas, vector_t<T>& y,
                  vector_t<IndexT>& sv_indices, vector_t<T>& sv_alphas) {
  return q.submit([&](handler& cgh) {
    auto y_acc = y.template get_access_1d<access::mode::read>(cgh);
    auto old_a_acc = alphas.template get_access_1d<access::mode::read>(cgh);
    auto sv_idx_acc =
        sv_indices.template get_access_1d<access::mode::read>(cgh);
    auto new_a_acc =
        sv_alphas.template get_access_1d<access::mode::discard_write>(cgh);
    cgh.parallel_for<NameGen<0, ml_smo_copy_alphas, T, IndexT>>(
        sv_alphas.get_nd_range(), [=](nd_item<1> item) {
          auto row = item.get_global_id(0);
          auto sv_idx = sv_idx_acc(row);
          new_a_acc(row) = y_acc(sv_idx) * old_a_acc(sv_idx);
        });
  });
}

}  // namespace detail

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
 * The smo solves a quadratic problem by iteratively selecting a pair of
 * variable and optimizing them. This implementation is based on the paper
 * <em>Working Set Selection Using Second Order Information for Training Support
 * Vector Machines</em> (JLMR 6 2005) which is also used by libSVM. The main
 * difference is that the gradient G stores -y(t)*G(t) instead of the G(t) in
 * the paper.
 *
 * @see svm
 *
 * @tparam KernelCacheT cache to use for kernel matrix
 * @tparam T
 * @param q
 * @param[in] x data of size mxn where m is the number of observation
 * @param[in] y labels must be a vector of size pow2(m) with -1 or 1 the first m
 * elements and 0 after
 * @param c parameter for C-SVM
 * @param tol criteria for stopping condition
 * @param alpha_eps threshold above which alpha needs to be to be used as a
 * weight of a support vector
 * @param max_nb_iter maximum number of iterations
 * @param kernel_cache
 * @return smo_out containing the support vectors svs, the alphas (multiplied by
 * their respective labels) and the offset rho
 */
template <class KernelCacheT, class T>
smo_out<T> smo(queue& q, matrix_t<T>& x, vector_t<T>& y, T c, T tol,
               T alpha_eps, SYCLIndexT max_nb_iter, KernelCacheT kernel_cache) {
  auto m = access_ker_dim(x, 0);
  assert_eq(y.get_kernel_size(), to_pow2(m));

  std::vector<T> host_alphas(m);
  std::vector<T> host_y(m);
  auto copy_event = sycl_copy_device_to_host(q, y, host_y.data(), 0, m);

  if (max_nb_iter == 0) {
    max_nb_iter = std::max(10000000LU, m > INT_MAX / 100 ? INT_MAX : 100 * m);
  }

  vector_t<SYCLIndexT> device_scalar_index(range<1>(1));
  auto eig_scalar_index = sycl_to_eigen<1, 0>(device_scalar_index);
  vector_t<T> gradient(y.data_range, y.kernel_range);

  // vec_cond_greater and vec_cond_less are boolean vectors with type T for
  // convenience
  vector_t<T> vec_cond_greater(y.data_range, y.kernel_range);
  vector_t<T> vec_cond_less(y.data_range, y.kernel_range);
  vector_t<T> obj_values(y.data_range, y.kernel_range);

  auto cond_greater = [c, alpha_eps](T y, T a) {
    return T((y > 0 && a < c) || (y < 0 && a > alpha_eps));
  };
  auto cond_less = [c, alpha_eps](T y, T a) {
    return T((y > 0 && a > alpha_eps) || (y < 0 && a < c));
  };

  // Compute initial cond
  vec_unary_op(q, y, vec_cond_greater, ml::functors::positive<T>());
  vec_unary_op(q, y, vec_cond_less, ml::functors::negative<T>());
  sycl_copy(q, y, gradient);

  SYCLIndexT i;
  SYCLIndexT j;
  T diff;
  SYCLIndexT nb_iter = 0;
  T eps = 1E-8;
  copy_event.wait_and_throw();
  while (nb_iter < max_nb_iter) {
    if (!detail::select_wss(q, obj_values, gradient, vec_cond_greater,
                            vec_cond_less, tol, eps, eig_scalar_index,
                            kernel_cache, i, j, diff)) {
      break;
    }
    /*
    std::cout << "#" << nb_iter << " i=" << i << " j=" << j
              << " diff=" << diff << std::endl;
    */
    auto ker_i_t = kernel_cache.get_ker_row(i);
    auto ker_j_t = kernel_cache.get_ker_row(j);

    T& ai = host_alphas[i];
    T& aj = host_alphas[j];
    T yi = host_y[i];
    T yj = host_y[j];

    T a = std::max(kernel_cache.get_ker_diag(i) + kernel_cache.get_ker_diag(j) -
                       2 * ker_i_t.read_to_host(j),
                   eps);
    T b = gradient.read_to_host(i) - gradient.read_to_host(j);

    // Update alphas i and j
    T old_ai = ai;
    T old_aj = aj;
    ai += yi * b / a;

    // Project alpha back to feasible region
    T s = yi * old_ai + yj * old_aj;
    ai = clamp(ai, T(0), c);
    aj = yj * (s - yi * ai);
    aj = clamp(aj, T(0), c);
    ai = yi * (s - yj * aj);

    // Update gradient
    T delta_ai = yi * (ai - old_ai);
    T delta_aj = yj * (aj - old_aj);

    // Shouldn't happen in theory but can because of precision issue
    if (std::abs(delta_ai) < eps || std::abs(delta_aj) < eps) {
      // Try again with a different working set
      vec_cond_less.write_from_host(j, false);
      continue;
    }

    auto update_event = detail::update_gradient(q, delta_ai, delta_aj, ker_i_t,
                                                ker_j_t, gradient);
    vec_cond_greater.write_from_host(i, cond_greater(yi, ai));
    vec_cond_greater.write_from_host(j, cond_greater(yj, aj));
    vec_cond_less.write_from_host(i, cond_less(yi, ai));
    vec_cond_less.write_from_host(j, cond_less(yj, aj));
    ++nb_iter;
    update_event.wait_and_throw();
  }

  if (nb_iter == max_nb_iter) {
    std::cout << "Warning: maximum number of iteration reached, SVM may not "
                 "have converged."
              << std::endl;
  }

  // Compute host_sv_indices
  std::vector<SYCLIndexT> host_sv_indices;
  for (unsigned k = 0; k < m; ++k) {
    if (host_alphas[k] > alpha_eps) {
      host_sv_indices.push_back(k);
    }
  }
  auto nb_sv = host_sv_indices.size();
  if (nb_sv == 0) {
    std::cerr << "Error: no support vectors could be found with the given "
                 "parameters. Try lowering alpha_eps, increasing C or using a "
                 "different kernel."
              << std::endl;
    assert(false);
    return smo_out<T>();
  }
  vector_t<SYCLIndexT> sv_indices(host_sv_indices.data(), range<1>(nb_sv));

  // Compute rho
  T rho = 0;
  {
    auto host_gradient = gradient.template get_access<access::mode::read>();
    for (auto sv_idx : host_sv_indices) {
      rho += host_gradient[sv_idx];
    }
    rho /= nb_sv;
  }

  // Copy the selected support vectors and y .* alphas
  vector_t<T> alphas(y.data_range, y.kernel_range);
  copy_event = sycl_copy_host_to_device(q, host_alphas.data(), alphas, 0, m);
  auto svs = split_by_index(q, x, sv_indices);
  vector_t<T> sv_alphas(sv_indices.data_range, sv_indices.kernel_range);
  detail::copy_alphas(q, alphas, y, sv_indices, sv_alphas);

  copy_event.wait_and_throw();
  return smo_out<T>{svs, sv_alphas, rho, nb_iter};
}

}  // namespace ml

#endif  // INCLUDE_ML_CLASSIFIERS_SVM_SMO_HPP
