#ifndef INCLUDE_ML_MATH_CG_SOLVE_HPP
#define INCLUDE_ML_MATH_CG_SOLVE_HPP

#include "ml/math/mat_mul.hpp"
#include "ml/math/mat_ops.hpp"
#include "ml/math/vec_ops.hpp"

namespace ml
{

namespace detail
{

class ml_cg_update_x_r;
class ml_cg_update_p;

template <class T>
void update_x_r(queue& q, vector_t<T>& v1, vector_t<T>& v2, T alpha) {
  q.submit([&](handler& cgh) {
    auto v1_acc = v1.template get_access_1d<access::mode::read_write>(cgh);
    auto v2_acc = v2.template get_access_1d<access::mode::read>(cgh);
    cgh.parallel_for<NameGen<0, ml_cg_update_x_r, T>>(v1.get_nd_range(), [=](nd_item<1> item) {
      auto row = item.get_global(0);
      v1_acc(row) += alpha * v2_acc(row);
    });
  });
}

template <class T>
void update_p(queue& q, vector_t<T>& p, vector_t<T>& r, T factor) {
  q.submit([&](handler& cgh) {
    auto p_acc = p.template get_access_1d<access::mode::read_write>(cgh);
    auto r_acc = r.template get_access_1d<access::mode::read>(cgh);
    cgh.parallel_for<NameGen<0, ml_cg_update_p, T>>(p.get_nd_range(), [=](nd_item<1> item) {
      auto row = item.get_global(0);
      p_acc(row) = r_acc(row) + factor * p_acc(row);
    });
  });
}

} // detail

/**
 * @brief Solve the system Ax = b where A is a symmetric SPD matrix of size nxn.
 *
 * Uses the Conjugate Gradient method.
 *
 * @tparam T
 * @param q
 * @param[in] a of size nxn
 * @param[in] b of size n
 * @param[in,out] x of size n, it is used as an initial guess, if you have none, it must be set to 0
 * @param r temporary buffer must be at least of size n
 * @param p temporary buffer must be at least of size n
 * @param Ap temporary buffer must be at least of size n
 */
template <class T>
void cg_solve(queue& q, matrix_t<T>& a, vector_t<T>& b, vector_t<T>& x,
              vector_t<T>& r, vector_t<T>& p, vector_t<T>& Ap, T epsilon = 1E-4) {
  auto n = a.data_range[0];
  a.assert_square();
  assert_less_or_eq(n, x.data_range[0]);
  assert_less_or_eq(n, r.data_range[0]);
  assert_less_or_eq(n, p.data_range[0]);
  assert_less_or_eq(n, Ap.data_range[0]);

  sycl_copy(q, b, r);
  sycl_copy(q, b, p);

  T rs_old = sycl_inner_product(q, r);
  T rs_new = 0;

  for (SYCLIndexT i = 0; i < n; ++i) {
    mat_mul(q, A, p, Ap);
    alpha = rs_old / sycl_inner_product(q, p, Ap);
    detail::update_x_r(q, x, p, alpha);
    detail::update_x_r(q, r, Ap, -alpha);
    rs_new = sycl_inner_product(q, r);
    if (std::sqrt(rs_new) < epsilon)
      break;
    detail::update_p(q, p, r, rs_new / rs_old);
    rs_old = rs_new;
  }
}

/**
 * @brief Solve the system Ax = b and create any necessary temporary buffers.
 *
 * @see cg_solve(queue&, matrix_t<T>&, vector_t<T>&, vector_t<T>&, vector_t<T>&, vector_t<T>&, vector_t<T>&)
 * @tparam T
 * @param q
 * @param[in] a
 * @param[in] b
 * @param[out] x
 */
template <class T>
void cg_solve(queue& q, matrix_t<T>& a, vector_t<T>& b, vector_t<T>& x, T epsilon = 1E-4) {
  auto n = a.data_range[0];
  vector_t<T> r(b.data_range, b.kernel_range);
  vector_t<T> p(b.data_range, b.kernel_range);
  vector_t<T> Ap(b.data_range, b.kernel_range);
  cg_solve(q, a, b, x, r, p, Ap, epsilon);
}

} // ml

#endif //INCLUDE_ML_MATH_CG_SOLVE_HPP
