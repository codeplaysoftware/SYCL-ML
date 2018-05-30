#include <iostream>

#include "ml/classifiers/svm/svm.hpp"
#include "utils/utils.hpp"

template <class DataT, class LabelT>
void test_svm_or() {
  std::array<DataT, 8> host_data {0, 0,
                                  0, 1,
                                  1, 0,
                                  1, 1};
  std::array<LabelT, 4> host_labels {0, 1, 1, 1};
  std::shared_ptr<DataT> host_alphas;
  DataT host_rho;

  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<DataT> sycl_data(host_data.data(), cl::sycl::range<2>(4, 2));
    ml::vector_t<LabelT> sycl_labels(host_labels.data(), cl::sycl::range<1>(4));

    ml::svm<ml::svm_linear_kernel<DataT>, LabelT> svm(10);
    svm.train_binary(q, sycl_data, sycl_labels);

    auto smo_out = svm.get_smo_outs().front();
    ::assert_eq(smo_out.alphas.data_range[0], 3LU);
    host_alphas = ml::make_shared_array(new DataT[smo_out.alphas.get_count()]);
    ml::sycl_copy_device_to_host(q, smo_out.alphas, host_alphas);
    host_rho = smo_out.rho;

    sycl_data.set_final_data(nullptr);
    sycl_labels.set_final_data(nullptr);
    clear_eigen_device();
  }

  std::cout << "alphas:\n";
  ml::print(host_alphas.get(), 1, 3);
  std::cout << "\nrho: " << host_rho << std::endl;

  std::array<DataT, 3> expected_alphas {-4, 2, 2};
  assert_vec_almost_eq(host_alphas.get(), expected_alphas.data(), expected_alphas.size());
  assert_almost_eq(host_rho, DataT(-1));
}

int main() {
  try {
    test_svm_or<ml::buffer_data_type, uint8_t>();
  } catch (cl::sycl::exception e) {
    std::cerr << e.what();
  }

  return 0;
}

