#include <iostream>

#include "ml/math/mat_ops.hpp"
#include "ml/math/cov.hpp"
#include "utils/utils.hpp"

template <class T, ml::data_dim D>
void test_cov_square() {
  std::array<T, 9> host_data {1.0, 4.0, 7.0,
                              2.0, 0.0, -8.0,
                              1.0, 2.0, 1.0};

  std::array<T, 9> host_cov;
  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> sycl_data(host_data.data(), cl::sycl::range<2>(3, 3));
    sycl_data.set_final_data(nullptr);
    ml::vector_t<T> sycl_data_avg{cl::sycl::range<1>(3)};

    ml::avg<D>(q, sycl_data, sycl_data_avg);
    ml::center_data<ml::opp<D>()>(q, sycl_data, sycl_data_avg);

    ml::matrix_t<T> sycl_cov(cl::sycl::range<2>(3, 3));
    ml::cov<D>(q, sycl_data, sycl_cov);
    sycl_cov.set_final_data(host_cov.data());
    clear_eigen_device();
  }

  std::cout << "host data:\n";
  ml::print(host_data, 3, 3);
  std::cout << "\ncov:\n";
  ml::print(host_cov, 3, 3);

  std::array<T, 9> expected {6.0,         -10.0,       0.0,
                             host_cov[1], 56.0/3.0,    2.0/3.0,
                             host_cov[2], host_cov[5], 2.0/9.0};
  assert_vec_almost_eq(host_cov, expected);
}

template <class T, ml::data_dim D>
void test_cov_general() {
  // 3 observations that have 2 variables each
  std::array<T, 6> host_data {1.0, 2.0,
                              3.0, 2.0,
                              2.0, 11.0};

  std::array<T, 4> host_cov;
  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<T> sycl_data(host_data.data(), cl::sycl::range<2>(3, 2));
    ml::vector_t<T> sycl_data_avg(cl::sycl::range<1>(2));

    ml::avg<D>(q, sycl_data, sycl_data_avg);
    ml::center_data<ml::opp<D>()>(q, sycl_data, sycl_data_avg);

    ml::matrix_t<T> sycl_cov(cl::sycl::range<2>(2, 2));
    ml::cov<D>(q, sycl_data, sycl_cov);
    sycl_cov.set_final_data(host_cov.data());
    clear_eigen_device();
  }

  std::cout << "data:\n";
  ml::print(host_data, 3, 2);
  std::cout << "\ncov:\n";
  ml::print(host_cov, 2, 2);

  std::array<T, 4> expected {2.0/3.0,     0.0,
                             host_cov[1], 18.0};
  assert_vec_almost_eq(host_cov, expected);
}

int main(void) {
  try {
    test_cov_square<ml::buffer_data_type, ml::COL>();
    test_cov_general<ml::buffer_data_type, ml::ROW>();
  } catch (cl::sycl::exception e) {
    std::cerr << e.what();
  }

  return 0;
}
