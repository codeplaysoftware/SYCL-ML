# SYCL-ML

## What is it?
SYCL-ML is a framework providing simple classical machine learning algorithms using SYCL.
It is meant to be accelerated on any OpenCL device supporting SPIR or SPIR-V.
The following links give more details on what SYCL is:
- https://www.khronos.org/sycl
- https://developer.codeplay.com/computecppce/latest/sycl-guide-introduction

## What can it do?
Some linear algebra operations had to be implemented such as:
- **Matrix inversion**
- **SVD decomposition**
- **QR decomposition**

In terms of machine learning related algorithms it includes:
- **Principal Component Analysis**: used to reduce the dimensionality of a problem.
- **Linear Classifier** (see naive Bayes classifier): classify assuming all variables are equally as important.
- **Gaussian Classifier**: classify using the Gaussian distribution.
- **Gaussian Mixture Model**: based on the EM algorithm, uses multiple Gaussian distribution for each labels.
- **Support Vector Machine**: C-SVM with any kernel function.

SYCL-ML is a header only library which makes it easy to integrate.

More details on what the project implements and how it works can be found on our [website](https://www.codeplay.com/portal/12-21-17-alternative-machine-learning-algorithms-using-sycl-and-opencl).

## TODO list
- Optimize **SVD** decomposition for faster PCA. The algorithm probably needs to be changed to compute eigenpairs differently.
- Optimize **SVM** for GPU. More recent papers on SVM for GPU should be experimented.
- Implement an **LDA** (or dimensionality reduction algorithms) which would be used as a preprocessing step similarly to a PCA.
- Implement a **K-means** (or other clustering algorithms) which could be used to improve the initialization of the EM.
- Add a proper way to select a SYCL device.

## Prerequisites
SYCL-ML has been tested with:
- Ubuntu 16.04, amdgpu pro driver 17.40
- CMake 3.0
- g++ 5.4
- ComputeCpp 1.2.0

ComputeCpp can be downloaded from the [CodePlay](https://www.codeplay.com/products/computesuite/computecpp) website.
Once extracted, ComputeCpp path should be set as an environment variable to `COMPUTECPP_DIR` (usually `/usr/local/computecpp`).
Alternatively, it can be given as an argument to cmake with `-DComputeCpp_DIR=path/to/computecpp`.

## Building
Build all the targets with:
```bash
mkdir build
cd build
cmake ..
make
```
CMake will take care of downloading the Eigen dependency and MNIST dataset.
On Unix it will automatically extract the MNIST dataset using `gunzip`.

It is recommended to run the tests before running the examples:
```bash
cd build/tests
ctest --output-on-failure
```

The documentation can be built with `doxygen`. It requires `dot` from the `graphviz` package. Simply run:
```bash
doxygen
```

## Contributing
The project is under the Apache 2.0 license. Any contribution is welcome! Also feel free to raise an issue for any
questions or suggestions.
