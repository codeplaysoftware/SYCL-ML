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
#ifndef EXAMPLE_SRC_MNIST_READ_MNIST_HPP
#define EXAMPLE_SRC_MNIST_READ_MNIST_HPP

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/*
 * Load the MNIST data set: http://yann.lecun.com/exdb/mnist/
 * The functions read *ubyte files meaning the .gz files have to be decompressed
 * (handled by CMake)
 */

// Convert from little to big endian.
static uint32_t reverse_int(uint32_t i) {
  unsigned char c1 = i & 255;
  unsigned char c2 = (i >> 8) & 255;
  unsigned char c3 = (i >> 16) & 255;
  unsigned char c4 = (i >> 24) & 255;

  return ((uint32_t) c1 << 24) + ((uint32_t) c2 << 16) + ((uint32_t) c3 << 8) +
         c4;
}

static void read_int(std::ifstream& file, uint32_t& i) {
  file.read(reinterpret_cast<char*>(&i), sizeof(i));
  i = reverse_int(i);
}

// Return the closest power of 2 higher or equal to x
template <class T>
static inline T to_pow2(T x) {
  return std::pow(2, std::ceil(std::log2(x)));
}

template <class T>
struct static_cast_func {
  template <class U>
  T operator()(const U& x) const {
    return static_cast<T>(x);
  }
};

static std::ifstream open_mnist_file(const std::string& full_path) {
  std::ifstream file(full_path, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    // The gz format does not specify the output filename.
    // If the file couldn't open with the suffix "-ubyte", try with ".ubyte"
    std::string other_full_path = full_path;
    other_full_path[other_full_path.size() - 6] = '.';
    file = std::ifstream(other_full_path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Cannot open file `" << full_path << "` nor `"
                << other_full_path << "`" << std::endl;
    }
  }
  return file;
}

// Read mnist, cast uchar to type T and transpose it (so that an image is a
// column)
template <class T>
std::vector<T> read_mnist_images(const std::string& full_path,
                                 unsigned& image_size,
                                 unsigned& padded_image_size,
                                 unsigned& nb_images, bool transpose,
                                 bool round_pow2, T norm_factor = 1) {
  std::ifstream file = open_mnist_file(full_path);
  if (!file.is_open()) {
    std::cerr << "Could not open file: " << full_path << std::endl;
    return {};
  }

  uint32_t magic_number = 0;
  read_int(file, magic_number);
  if (magic_number != 2051) {
    std::cerr << "Invalid MNIST file: " << full_path << std::endl;
    return {};
  }

  uint32_t read_nb_images = 0, read_nb_rows = 0, read_nb_cols = 0;
  read_int(file, read_nb_images);
  read_int(file, read_nb_rows);
  read_int(file, read_nb_cols);

  uint32_t out_read_nb_rows = read_nb_rows;
  uint32_t out_read_nb_cols = read_nb_cols;
  if (round_pow2) {
    out_read_nb_rows = to_pow2(read_nb_rows);
    out_read_nb_cols = to_pow2(read_nb_cols);
  }

  nb_images = read_nb_images;
  image_size = read_nb_rows * read_nb_cols;
  unsigned buffer_total_size = nb_images * image_size;
  std::vector<unsigned char> buffer(buffer_total_size);

  padded_image_size = out_read_nb_rows * out_read_nb_cols;
  unsigned dataset_total_size = nb_images * padded_image_size;
  std::vector<T> dataset(dataset_total_size);

  file.read(reinterpret_cast<char*>(buffer.data()), buffer_total_size);

  if (transpose) {
    for (unsigned c = 0; c < nb_images; ++c) {
      for (unsigned r = 0; r < image_size; ++r) {
        // Cast, normalize and transpose
        dataset[r * nb_images + c] =
            static_cast<T>(buffer[c * image_size + r]) / norm_factor;
      }
    }
    if (round_pow2) {  // Set all zeros in the end
      std::memset(&dataset[image_size * nb_images], 0,
                  (padded_image_size - image_size) * nb_images * sizeof(T));
    }
  } else {
    for (unsigned r = 0; r < nb_images; ++r) {
      for (unsigned c = 0; c < image_size; ++c) {
        // Cast and normalize
        dataset[r * padded_image_size + c] =
            static_cast<T>(buffer[r * image_size + c]) / norm_factor;
      }
      std::memset(&dataset[r * padded_image_size + image_size], 0,
                  (padded_image_size - image_size) * sizeof(T));
    }
  }

  return dataset;
}

template <class T>
std::vector<T> read_mnist_labels(const std::string& full_path,
                                 unsigned& nb_labels) {
  std::ifstream file = open_mnist_file(full_path);
  if (!file.is_open()) {
    std::cerr << "Could not open file: " << full_path << std::endl;
    return {};
  }

  uint32_t magic_number = 0;
  read_int(file, magic_number);
  if (magic_number != 2049) {
    std::cerr << "Invalid MNIST file: " << full_path << std::endl;
    return {};
  }

  uint32_t read_nb_labels = 0;
  read_int(file, read_nb_labels);
  nb_labels = read_nb_labels;

  std::vector<unsigned char> buffer(nb_labels);
  std::vector<T> labels(nb_labels);

  file.read(reinterpret_cast<char*>(buffer.data()), nb_labels);
  std::transform(buffer.begin(), buffer.end(), labels.begin(),
                 static_cast_func<T>());

  return labels;
}

inline std::string mnist_get_train_images_path(const std::string& prefix) {
  return prefix + "/train-images-idx3-ubyte";
}

inline std::string mnist_get_train_labels_path(const std::string& prefix) {
  return prefix + "/train-labels-idx1-ubyte";
}

inline std::string mnist_get_test_images_path(const std::string& prefix) {
  return prefix + "/t10k-images-idx3-ubyte";
}

inline std::string mnist_get_test_labels_path(const std::string& prefix) {
  return prefix + "/t10k-labels-idx1-ubyte";
}

#endif  // EXAMPLE_SRC_MNIST_READ_MNIST_HPP
