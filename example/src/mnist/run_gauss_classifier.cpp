#include "run_classifier.hpp"
#include "ml/classifiers/bayes/bayes_classifier.hpp"
#include "ml/classifiers/bayes/distributions/log_gaussian_distribution.hpp"

int main(int argc, char** argv) {
  std::string mnist_path = "data/mnist";
  if (argc >= 2)
    mnist_path = argv[1];

  // Runs the gaussian classifier on MNIST with a PCA
  using distribution_t = ml::buffered_log_gaussian_distribution<ml::buffer_data_type>;
  ml::pca_args<ml::buffer_data_type> pca_args;
  pca_args.min_nb_vecs = 64;    // Keep at least 64 basis vector
  pca_args.keep_percent = 0.8;  // Keep at least 80% of information
  pca_args.scale_factor = 1E2;  // More accurate but slower PCA
  try {
    run_classifier<ml::bayes_classifier<distribution_t, uint8_t>>(mnist_path, pca_args);
  } catch (cl::sycl::exception e) {
    std::cerr << e.what();
  }

  return 0;
}
