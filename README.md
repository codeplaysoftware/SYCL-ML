# SYCL-ML

## What is it?
SYCL-ML is a framework providing simple classical machine learning algorithms using SYCL.
It is meant to be accelerated on any OpenCL device supporting SPIR or SPIRV (experimental).
The following links give more details on what SYCL is:
- https://www.khronos.org/sycl
- https://developer.codeplay.com/computecppce/latest/sycl-guide-introduction

## What can it do?
Some linear algebra operations had to be written from scratch such as:
- **Matrix inversion**
- **SVD decomposition**
- **QR decomposition**

In terms of more machine learning related operations it includes:
- **Principal Component Analysis**: used to reduce the dimensionality of a problem.
- **Linear Classifier** (see naive Bayes classifier): classify assuming all variables are equally as important.
- **Gaussian Classifier**: classify using the Gaussian distribution.
- **Gaussian Mixture Model**: based on the EM algorithm, uses multiple Gaussian distribution for each labels.
- **Support Vector Machine**: C-SVM with any possible kernel function.

SYCL-ML is a header only library which make it easy to integrate.

More details on what the project implements and how it works can be found on [our website](https://www.codeplay.com/portal/12-21-17-alternative-machine-learning-algorithms-using-sycl-and-opencl). Make sure to use the blogpost branch if you want to observe the same results as shown there.

## TODO list
- Optimize **SVD** decomposition for faster PCA. The algorithm probably needs to be changed to compute eigenpairs differently.
- Optimize **SVM** for GPU. More recent papers on SVM for GPU should be experimented.
- Implement an **LDA** (or dimensionality reduction algorithms) which would be used as a preprocessing step similarly to a PCA.
- Implement a **K-means** (or other clustering algorithms) which could be used to improve the initialization of the EM.

## Prerequisites
SYCL-ML has been tested with:
- Ubuntu 16.04.3, kernel 4.10.0-28, amdgpu pro driver 17.30  OR  Ubuntu 14.04.5, kernel 3.19.0-79, fglrx driver 2:15.302
- CMake 3.0
- g++ 5.4
- ComputeCpp 0.5.0

ComputeCpp can be downloaded from the [CodePlay](https://www.codeplay.com/products/computesuite/computecpp) website.
Once extracted, ComputeCpp path should be set as an environment variable to `COMPUTECPP_PACKAGE_ROOT_DIR` (usually */usr/local/computecpp*).
Alternatively, it can be given as an argument to cmake with `COMPUTECPP_PACKAGE_ROOT_DIR`.

SYCL-ML depends on [SYCLParallelSTL](https://github.com/KhronosGroup/SyclParallelSTL).
SYCLParallelSTL's path must be set to `SYCL_PARALLEL_STL_ROOT` either as an environment variable or as an argument to cmake.
```bash
git clone https://github.com/KhronosGroup/SyclParallelSTL.git
```

The last requirement is the Eigen-Optimised-Tensor-Vector-Contraction branch of [Eigen](https://bitbucket.org/mehdi_goli/opencl).
Eigen's path must be set to `EIGEN_INCLUDE_DIRS` either as an environment variable or as an argument to cmake.
The version of Eigen needed is slightly different than the upstream.
The changes are packed in the `eigen.patch` file which the next section shows how to apply.
```bash
hg clone https://bitbucket.org/mehdi_goli/opencl
hg up Eigen-Optimised-Tensor-Vector-Contraction
```

## Building
The eigen patch file must be applied first then cmake and make:
```bash
patch -p1 -d <Eigen_root> < eigen.patch
mkdir build && cd build
cmake -DSYCL_PARALLEL_STL_ROOT=<SYCLParallelSTL_root> -DEIGEN_INCLUDE_DIRS=<Eigen_root> ..
make
```
Note that on Unix CMake will take care of downloading the MNIST dataset using *wget* and *gunzip*.

It is recommended to run the tests before running the examples:
```bash
cd build/tests
ctest --output-on-failure
```

The documentation can be built with *doxygen*. It requires *dot* from the *graphviz* package. Simply run:
```bash
doxygen
```

## Contributing
The project is under the Apache 2.0 license. Any contribution is welcome! Also feel free to raise an issue for any
questions or suggestions.
