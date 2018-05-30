#include <iostream>

#include "ml/classifiers/svm/svm.hpp"
#include "utils/utils.hpp"

template <class DataT, class LabelT>
void test_svm_xor() {
  std::array<DataT, 8> host_data {0, 0,
                                  0, 1,
                                  1, 0,
                                  1, 1};
  std::array<LabelT, 4> host_labels {0, 1, 1, 0};
  std::shared_ptr<DataT> host_alphas;
  DataT host_rho;

  {
    cl::sycl::queue& q = create_queue();
    ml::matrix_t<DataT> sycl_data(host_data.data(), cl::sycl::range<2>(4, 2));
    ml::vector_t<LabelT> sycl_labels(host_labels.data(), cl::sycl::range<1>(4));

    using KernelType = ml::svm_polynomial_kernel<DataT>;
    ml::svm<KernelType, LabelT> svm(1000, KernelType(1, 1, 2), 2, 1E-6);
    svm.train_binary(q, sycl_data, sycl_labels);

    auto smo_out = svm.get_smo_outs().front();
    ::assert_eq(smo_out.alphas.data_range[0], 4LU);
    host_alphas = ml::make_shared_array(new DataT[smo_out.alphas.get_count()]);
    ml::sycl_copy_device_to_host(q, smo_out.alphas, host_alphas);
    host_rho = smo_out.rho;

    sycl_data.set_final_data(nullptr);
    sycl_labels.set_final_data(nullptr);
    clear_eigen_device();
  }

  std::cout << "alphas:\n";
  ml::print(host_alphas.get(), 1, 4);
  std::cout << "\nrho: " << host_rho << std::endl;

  std::array<DataT, 4> expected_alphas {-3.332425, 2.665940, 2.665940, -1.999455};
  assert_vec_almost_eq(host_alphas.get(), expected_alphas.data(), expected_alphas.size(), DataT(1E-3));
  assert_almost_eq(host_rho, DataT(-0.999728), DataT(1E-3));
}

int main() {
  try {
    test_svm_xor<ml::buffer_data_type, uint8_t>();
  } catch (cl::sycl::exception e) {
    std::cerr << e.what();
  }

  return 0;
}

