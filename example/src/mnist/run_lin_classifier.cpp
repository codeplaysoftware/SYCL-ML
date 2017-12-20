#include "run_classifier.hpp"
#include "ml/classifiers/bayes/linear_classifier.hpp"

int main(int argc, char** argv) {
  std::string mnist_path = "data/mnist";
  if (argc >= 2)
    mnist_path = argv[1];
  // Runs the linear classifier on MNIST with a PCA
  ml::pca_args<ml::buffer_data_type> pca_args;
  pca_args.min_nb_vecs = 128;   // Keep at least 128 basis vector
  pca_args.keep_percent = 0.8;  // Keep at least 80% of information
  pca_args.scale_factor = 1E2;  // More accurate but slower PCA
  run_classifier<ml::linear_classifier<ml::buffer_data_type, uint8_t>>(mnist_path, pca_args);

  return 0;
}
