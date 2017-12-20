#include "run_classifier.hpp"
#include "ml/classifiers/svm/svm.hpp"

int main(int argc, char** argv) {
  std::string mnist_path = "data/mnist";
  if (argc >= 2)
    mnist_path = argv[1];

  // Runs the SVM with the RBF kernel on MNIST with a PCA.
  // The SVM will store 2 rows of the kernel matrix and has a tolerance of 0.1
  using data_t = ml::buffer_data_type;
  using label_t = uint8_t;
  using svm_kernel_t = ml::svm_rbf_kernel<data_t>;

  const data_t C = 5;           // Parameter of a C-SVM
  const svm_kernel_t ker(0.05); // Parameter of the RBF kernel

  ml::pca_args<data_t> pca_args;
  pca_args.min_nb_vecs = 64;    // Keep at least 64 basis vector
  pca_args.keep_percent = 0.8;  // Keep at least 80% of information
  pca_args.scale_factor = 1E2;  // More accurate but slower PCA

  run_classifier(mnist_path, pca_args, ml::svm<svm_kernel_t, label_t>(C, ker, 2, 0.1));

  return 0;
}
