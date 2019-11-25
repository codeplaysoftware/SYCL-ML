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
#ifndef INCLUDE_ML_CLASSIFIERS_SVM_SVM_KERNELS_HPP
#define INCLUDE_ML_CLASSIFIERS_SVM_SVM_KERNELS_HPP

#include "ml/math/vec_ops.hpp"

namespace ml {

/**
 * @brief A kernel function usable by an SVM.
 *
 * @see svm
 * @tparam T
 */
template <class T>
struct svm_kernel {
  using DataType = T;

  /**
   * @brief Compute the ith row of the kernel matrix.
   *
   * This method is optional if kernel_cache_row is not used
   * (i.e. if the SVM using this kernel is called with the option
   * nb_cache_line=0).
   *
   * \f$
   * out(t) = ker(i, t)
   * \f$
   *
   * @param q
   * @param[in] x size mxn
   * @param[in] i
   * @param[out] out size m
   */
  virtual void operator()(queue& q, matrix_t<T>& x, SYCLIndexT i,
                          vector_t<T>& out) const = 0;

  /**
   * @brief Compute the diagonal of the kernel matrix.
   *
   * \f$
   * out(t) = ker(t, t)
   * \f$
   *
   * @param q
   * @param[in] x size mxn
   * @param[out] out size m
   */
  virtual void operator()(queue& q, matrix_t<T>& x, vector_t<T>& out) const = 0;

  /**
   * @brief Compute m1xm2 kernel values.
   *
   * \f$
   * out(i, j) = ker(i, j)
   * \f$
   *
   * @param q
   * @param[in] x1 size m1xn
   * @param[in] x2 size m2xn
   * @param[out] out size m1xm2
   */
  virtual void operator()(queue& q, matrix_t<T>& x1, matrix_t<T>& x2,
                          matrix_t<T>& out) const = 0;
};

/**
 * @brief Linear SVM kernel
 *
 * \f$
 * ker = u * v'
 * \f$
 *
 * @tparam T
 */
template <class T>
struct svm_linear_kernel : public svm_kernel<T> {
  void operator()(queue&, matrix_t<T>& x, SYCLIndexT i,
                  vector_t<T>& out) const override {
    auto n = access_ker_dim(x, 1);
    auto eig_x = sycl_to_eigen(x);
    auto eig_out = sycl_to_eigen_2d<ROW>(out);
    auto sliced_x =
        eig_x.tensor().slice(eig_dsize_t<2>{static_cast<eig_index_t>(i), 0},
                             eig_dsize_t<2>{1, static_cast<eig_index_t>(n)});

    eig_out.device() =
        sliced_x.contract(eig_x.tensor(), get_contract_dim<COL, COL>());
  }

  void operator()(queue&, matrix_t<T>& x, vector_t<T>& out) const override {
    auto eig_x = sycl_to_eigen(x);
    auto eig_out = sycl_to_eigen(out);

    eig_out.device() = eig_x.tensor().square().sum(eig_dims_t<1>{1});
  }

  void operator()(queue&, matrix_t<T>& x1, matrix_t<T>& x2,
                  matrix_t<T>& out) const override {
    auto m1 = access_ker_dim(x1, 0);
    auto m2 = access_ker_dim(x2, 0);
    auto eig_x1 = sycl_to_eigen(x1);
    auto eig_x2 = sycl_to_eigen(x2);
    auto eig_out = sycl_to_eigen(out);

    auto ker_mat =
        eig_x1.tensor().contract(eig_x2.tensor(), get_contract_dim<COL, COL>());
    if (access_ker_dim(out, 0) == m1 && access_ker_dim(out, 1) == m2) {
      eig_out.device() = ker_mat;
    } else {
      auto sliced_out = eig_out.tensor().slice(
          eig_dsize_t<2>{0, 0}, detail::range_to_dsize(range<2>(m1, m2)));
      sliced_out.device(get_eigen_device()) = ker_mat;
    }
  }
};

/**
 * @brief Polynomial SVM kernel
 *
 * \f$
 * ker = (g .* (u * v') .+ c) .^ d
 * \f$
 *
 * @tparam T
 */
template <class T>
struct svm_polynomial_kernel : public svm_kernel<T> {
  svm_polynomial_kernel(T g, T c, T d) : _g(g), _c(c), _d(d) {}

  void operator()(queue&, matrix_t<T>& x, SYCLIndexT i,
                  vector_t<T>& out) const override {
    auto n = access_ker_dim(x, 1);
    auto eig_x = sycl_to_eigen(x);
    auto eig_out = sycl_to_eigen_2d<ROW>(out);
    auto sliced_x =
        eig_x.tensor().slice(eig_dsize_t<2>{static_cast<eig_index_t>(i), 0},
                             eig_dsize_t<2>{1, static_cast<eig_index_t>(n)});

    eig_out.device() =
        (_g * sliced_x.contract(eig_x.tensor(), get_contract_dim<COL, COL>()) +
         _c)
            .pow(_d);
  }

  void operator()(queue&, matrix_t<T>& x, vector_t<T>& out) const override {
    auto eig_x = sycl_to_eigen(x);
    auto eig_out = sycl_to_eigen(out);

    eig_out.device() =
        (_g * eig_x.tensor().square().sum(eig_dims_t<1>{1}) + _c).pow(_d);
  }

  void operator()(queue&, matrix_t<T>& x1, matrix_t<T>& x2,
                  matrix_t<T>& out) const override {
    auto m1 = access_ker_dim(x1, 0);
    auto m2 = access_ker_dim(x2, 0);
    auto eig_x1 = sycl_to_eigen(x1);
    auto eig_x2 = sycl_to_eigen(x2);
    auto eig_out = sycl_to_eigen(out);

    auto ker_mat = (_g * eig_x1.tensor().contract(
                             eig_x2.tensor(), get_contract_dim<COL, COL>()) +
                    _c)
                       .pow(_d);
    if (access_ker_dim(out, 0) == m1 && access_ker_dim(out, 1) == m2) {
      eig_out.device() = ker_mat;
    } else {
      auto sliced_out = eig_out.tensor().slice(
          eig_dsize_t<2>{0, 0}, detail::range_to_dsize(range<2>(m1, m2)));
      sliced_out.device(get_eigen_device()) = ker_mat;
    }
  }

 private:
  T _g;
  T _c;
  T _d;
};

/**
 * @brief (Gaussian) Radial Basis Function SVM kernel
 *
 * \f$
 * ker = exp(-g .* |u - v|^2)
 * \f$
 *
 * @tparam T
 */
template <class T>
struct svm_rbf_kernel : public svm_kernel<T> {
  svm_rbf_kernel(T g) : _g(g) {}

  void operator()(queue&, matrix_t<T>& x, SYCLIndexT i,
                  vector_t<T>& out) const override {
    auto m = access_ker_dim(x, 0);
    auto n = access_ker_dim(x, 1);
    auto eig_x = sycl_to_eigen(x);
    auto eig_out = sycl_to_eigen(out);

    auto sliced_x =
        eig_x.tensor().slice(eig_dsize_t<2>{static_cast<eig_index_t>(i), 0},
                             eig_dsize_t<2>{1, static_cast<eig_index_t>(n)});
    auto rep_sliced_x =
        sliced_x.broadcast(eig_dims_t<2>{static_cast<eig_index_t>(m), 1});

    eig_out.device() =
        ((eig_x.tensor() - rep_sliced_x).square().sum(eig_dims_t<1>{1}) * (-_g))
            .exp();
  }

  void operator()(queue& q, matrix_t<T>&, vector_t<T>& out) const override {
    sycl_memset(q, out, T(1));
  }

  void operator()(queue&, matrix_t<T>& x1, matrix_t<T>& x2,
                  matrix_t<T>& out) const override {
    auto m1 = access_ker_dim(x1, 0);
    auto m2 = access_ker_dim(x2, 0);
    auto n = static_cast<eig_index_t>(access_ker_dim(x1, 1));
    assert_eq(n, static_cast<eig_index_t>(access_ker_dim(x2, 1)));

    auto eig_x1 = sycl_to_eigen(x1);
    auto eig_x2 = sycl_to_eigen(x2);
    auto eig_out = sycl_to_eigen(out);

    auto rep_x1 =
        eig_x1.tensor()
            .reshape(eig_dims_t<3>{static_cast<eig_index_t>(m1), 1, n})
            .broadcast(eig_dims_t<3>{1, static_cast<eig_index_t>(m2), 1});
    auto rep_x2 =
        eig_x2.tensor()
            .reshape(eig_dims_t<3>{1, static_cast<eig_index_t>(m2), n})
            .broadcast(eig_dims_t<3>{static_cast<eig_index_t>(m1), 1, 1});

    auto ker_mat =
        ((rep_x1 - rep_x2).square().sum(eig_dims_t<1>{2}) * (-_g)).exp();
    if (access_ker_dim(out, 0) == m1 && access_ker_dim(out, 1) == m2) {
      eig_out.device() = ker_mat;
    } else {
      auto sliced_out = eig_out.tensor().slice(
          eig_dsize_t<2>{0, 0}, detail::range_to_dsize(range<2>(m1, m2)));
      sliced_out.device(get_eigen_device()) = ker_mat;
    }
  }

 private:
  T _g;
};

/**
 * @brief Sigmoid SVM kernel
 *
 * \f$
 * ker = tanh(g .* (u * v') .+ c)
 * \f$
 *
 * @tparam T
 */
template <class T>
struct svm_sigmoid_kernel : public svm_kernel<T> {
  svm_sigmoid_kernel(T g, T c) : _g(g), _c(c) {}

  void operator()(queue&, matrix_t<T>& x, SYCLIndexT i,
                  vector_t<T>& out) const override {
    auto n = access_ker_dim(x, 1);
    auto eig_x = sycl_to_eigen(x);
    auto eig_out = sycl_to_eigen_2d<ROW>(out);
    auto sliced_x =
        eig_x.tensor().slice(eig_dsize_t<2>{static_cast<eig_index_t>(i), 0},
                             eig_dsize_t<2>{1, static_cast<eig_index_t>(n)});

    eig_out.device() =
        (_g * sliced_x.contract(eig_x.tensor(), get_contract_dim<COL, COL>()) +
         _c)
            .tanh();
  }

  void operator()(queue&, matrix_t<T>& x, vector_t<T>& out) const override {
    auto eig_x = sycl_to_eigen(x);
    auto eig_out = sycl_to_eigen(out);

    eig_out.device() =
        (_g * eig_x.tensor().square().sum(eig_dims_t<1>{1}) + _c).tanh();
  }

  void operator()(queue&, matrix_t<T>& x1, matrix_t<T>& x2,
                  matrix_t<T>& out) const override {
    auto m1 = access_ker_dim(x1, 0);
    auto m2 = access_ker_dim(x2, 0);
    auto eig_x1 = sycl_to_eigen(x1);
    auto eig_x2 = sycl_to_eigen(x2);
    auto eig_out = sycl_to_eigen(out);

    auto ker_mat = (_g * eig_x1.tensor().contract(
                             eig_x2.tensor(), get_contract_dim<COL, COL>()) +
                    _c)
                       .tanh();
    if (access_ker_dim(out, 0) == m1 && access_ker_dim(out, 1) == m2) {
      eig_out.device() = ker_mat;
    } else {
      auto sliced_out = eig_out.tensor().slice(
          eig_dsize_t<2>{0, 0}, detail::range_to_dsize(range<2>(m1, m2)));
      sliced_out.device(get_eigen_device()) = ker_mat;
    }
  }

 private:
  T _c;
  T _g;
};

}  // namespace ml

#endif  // INCLUDE_ML_CLASSIFIERS_SVM_SVM_KERNELS_HPP
