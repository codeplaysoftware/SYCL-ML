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
#ifndef INCLUDE_ML_UTILS_DEBUG_WRITE_BMP_HPP
#define INCLUDE_ML_UTILS_DEBUG_WRITE_BMP_HPP

#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>

#include "ml/utils/sycl_types.hpp"

namespace ml {

namespace detail {

inline void write_pixel(std::ofstream& ofs,
                        const std::array<unsigned char, 3>& pixel) {
  ofs.write(reinterpret_cast<const char*>(&pixel), 3 * sizeof(unsigned char));
}

}  // namespace detail

/**
 * @brief Save a 1d host buffer to a bmp file.
 *
 * The function allows to write a 1d buffer as if it were 2d and can also ignore
 * the padding. It also prints statistics such as min, max, avg, std, number of
 * nan and number of inf.
 *
 * @tparam T underlying type of each element of \p data
 * @param filename file to save to. Left unmodified if it ends by ".bmp"
 * otherwise append whether the normalize and abs options where used.
 * @param data
 * @param data_nb_cols offset factor for ach row
 * @param img_nb_rows output image width
 * @param img_nb_cols output image height
 * @param normalize move the data range in [0, 255]
 * @param abs apply abs before reading any element in data
 * @param min avoid computing the couple min max if provided
 * @param max avoid computing the couple min max if provided
 * @param from_r offset to start reading the row from
 * @param from_c offset to start reading the col from
 * @param to_r last row to read (excluded)
 * @param to_c last col to read (excluded)
 * @param nb_repeat if greater than 1, this will continue reading data and write
 * the image below the existing one (thus extending the image output height)
 * @param repeat_offset if \p nb_repeat is greater than 1, this will ignore a
 * block of data after the one already read.
 */
template <class T>
void write_bmp_grayscale(std::string filename, const T* data,
                         unsigned data_nb_cols, unsigned img_nb_rows,
                         unsigned img_nb_cols, bool normalize = true,
                         bool abs = false, T min = 0, T max = 0,
                         unsigned from_r = 0, unsigned from_c = 0,
                         unsigned to_r = 0, unsigned to_c = 0,
                         unsigned nb_repeat = 1, unsigned repeat_offset = 0) {
  if (to_r == 0) {
    to_r = img_nb_rows;
  }
  if (to_c == 0) {
    to_c = img_nb_cols;
  }

  assert(to_r > from_r);
  assert(to_c > from_c);

  static int count = 0;
  if (filename.length() <= 4 ||
      filename.find(".bmp") != filename.length() - 4) {
    std::stringstream ss;
    if (filename[filename.length() - 1] == '_') {
      ss << count++;
    }
    ss << "_normalize_" << (int) normalize;
    ss << "_abs_" << (int) abs;
    ss << ".bmp";
    filename += ss.str();
  }

  std::ofstream ofs(filename, std::ios::binary | std::ios::out);
  if (!ofs.is_open()) {
    std::cerr << "Could not create file " << filename << std::endl;
    return;
  } else {
    std::cout << "Writing to " << filename;
  }

  using get_data_f = std::function<T(long, long)>;
  auto for_each_pixel = [&](get_data_f get, std::function<void(T)> op) {
    for (long i = nb_repeat - 1; i >= 0; --i) {
      for (long r = to_r - 1; r >= 0; --r) {
        for (long c = from_c; c < to_c; ++c) {
          op(get(r + i * (to_r - from_r), c + i * repeat_offset));
        }
      }
    }
  };

  get_data_f get_data = [&](long r, long c) {
    return data[r * data_nb_cols + c];
  };
  get_data_f get_data_if_abs = get_data;
  if (abs) {
    get_data_if_abs = [&](long r, long c) { return std::fabs(get_data(r, c)); };
  }

  unsigned image_size = nb_repeat * img_nb_rows * img_nb_cols;
  unsigned nb_nan = 0;
  unsigned nb_inf = 0;
  if (max == min) {
    min = std::numeric_limits<T>::infinity();
    max = -std::numeric_limits<T>::infinity();
    float avg = 0;
    for_each_pixel(get_data_if_abs, [&](T d) {
      if (std::isnan(d)) {
        ++nb_nan;
      } else if (std::isinf(d)) {
        ++nb_inf;
      } else {
        if (d > max) {
          max = d;
        }
        if (d < min) {
          min = d;
        }
        avg += (float) d / image_size;
      }
    });

    float dev = 0;
    for_each_pixel(get_data_if_abs, [&](T d) {
      if (std::isfinite(d)) {
        T diff = d - avg;
        dev += diff * diff;
      }
    });
    dev = std::sqrt(dev / image_size);
    // min and max are promoted to a printable number (even if T=char)
    std::cout << " (min=" << +min << " max=" << +max << " avg=" << avg
              << " dev=" << dev;
    std::cout << " nb_nan=" << nb_nan << " nb_inf=" << nb_inf << ")";
  }
  std::cout << "..." << std::endl;

  get_data_f get_data_if_norm = get_data_if_abs;
  T min_max_diff = max - min;
  T norm_factor = min_max_diff ? T(255.0) / min_max_diff : T(1);
  if (normalize) {
    get_data_if_norm = [&](long r, long c) {
      return (get_data_if_abs(r, c) - min) * norm_factor;
    };
  }

  ofs << "BM";
  struct {
    uint32_t file_size;
    uint32_t reserved = 0;
    uint32_t off_bits = 54;
  } file_header;
  uint32_t pad_size = (4 - ((3 * img_nb_cols) % 4)) % 4;
  file_header.file_size =
      54 + 3 * (image_size + nb_repeat * img_nb_rows * pad_size);
  ofs.write(reinterpret_cast<const char*>(&file_header), sizeof(file_header));

  struct {
    uint32_t size = 40;
    uint32_t w;
    uint32_t h;
    uint16_t planes = 0;
    uint16_t bit_count = 24;  // 24 to skip the color table
    uint32_t compression = 0;
    uint32_t size_image = 0;
    uint32_t x_pels_per_meter = 0;
    uint32_t y_pels_per_meter = 0;
    uint32_t clr_used = 0;
    uint32_t clr_important = 0;
  } file_info;
  file_info.w = static_cast<uint32_t>(img_nb_cols);
  file_info.h = static_cast<uint32_t>(img_nb_rows * nb_repeat);
  ofs.write(reinterpret_cast<const char*>(&file_info), sizeof(file_info));

  char pad[]{0, 0, 0};
  unsigned char byte;
  unsigned act_byte = 0;

  for_each_pixel(get_data_if_norm, [&](T d) {
    if (std::isnan(d)) {
      detail::write_pixel(ofs, {0, 0, 255});  // red
    } else if (std::isinf(d)) {
      detail::write_pixel(ofs, {0, 255, 0});  // green
    } else {
      byte = static_cast<unsigned char>(
          std::min(std::max(T(std::round(d)), T(0)), T(255)));
      if (normalize && std::fabs(byte - d) > 0.5) {
        detail::write_pixel(ofs, {255, 0, 0});  // blue
      } else {
        detail::write_pixel(ofs, {byte, byte, byte});
      }
    }

    if (++act_byte % img_nb_cols == 0) {
      ofs.write(pad, pad_size);
    }
  });

  ofs.close();

  assert(nb_nan == 0);
  assert(nb_inf == 0);
}

/**
 * @brief Save a SYCL vector to a bmp file.
 *
 * The number of row of the image is the size of the vector's data range.
 * @see write_bmp_grayscale(std::string, const Data&, unsigned, unsigned,
 * unsigned, bool, bool, T, T, unsigned, unsigned, unsigned, unsigned,
 * unsigned, unsigned)
 */
template <class T>
inline void write_bmp_grayscale(std::string filename, vector_t<T>& data,
                                bool normalize = true, bool abs = false,
                                T min = 0, T max = 0, unsigned from_r = 0,
                                unsigned from_c = 0, unsigned to_r = 0,
                                unsigned to_c = 0) {
  std::vector<T> host_data(data.get_kernel_size());
  auto event = sycl_copy_device_to_host(get_eigen_device().sycl_queue(), data,
                                        host_data.data());
  event.wait_and_throw();
  write_bmp_grayscale<T>(filename, host_data.data(), 1, data.data_range[0], 1,
                         normalize, abs, min, max, from_r, from_c, to_r, to_c);
}

/**
 * @brief Save a SYCL vector to a bmp file.
 *
 * The number of row of the image is the size of the vector's kernel
 * range.
 * @see write_bmp_grayscale(std::string, const Data&, unsigned, unsigned,
 * unsigned, bool, bool, T, T, unsigned, unsigned, unsigned, unsigned,
 * unsigned, unsigned)
 */
template <class T>
inline void write_bmp_grayscale_ker_rng(std::string filename, vector_t<T>& data,
                                        bool normalize = true, bool abs = false,
                                        T min = 0, T max = 0,
                                        unsigned from_r = 0,
                                        unsigned from_c = 0, unsigned to_r = 0,
                                        unsigned to_c = 0) {
  std::vector<T> host_data(data.get_kernel_size());
  auto event = sycl_copy_device_to_host(get_eigen_device().sycl_queue(), data,
                                        host_data.data());
  event.wait_and_throw();
  write_bmp_grayscale<T>(filename, host_data.data(), 1,
                         data.get_kernel_range()[0], 1, normalize, abs, min,
                         max, from_r, from_c, to_r, to_c);
}

/**
 * @brief Save a SYCL matrix to a bmp file.
 *
 * The size of the image is the size of the matrix's data range.
 * @see write_bmp_grayscale(std::string, const Data&, unsigned, unsigned,
 * unsigned, bool, bool, T, T, unsigned, unsigned, unsigned, unsigned,
 * unsigned, unsigned)
 */
template <class T>
inline void write_bmp_grayscale(std::string filename, matrix_t<T>& data,
                                bool normalize = true, bool abs = false,
                                T min = 0, T max = 0, unsigned from_r = 0,
                                unsigned from_c = 0, unsigned to_r = 0,
                                unsigned to_c = 0) {
  std::vector<T> host_data(data.get_kernel_size());
  auto event = sycl_copy_device_to_host(get_eigen_device().sycl_queue(), data,
                                        host_data.data());
  event.wait_and_throw();
  write_bmp_grayscale<T>(filename, host_data.data(), access_ker_dim(data, 1),
                         access_data_dim(data, 0), access_data_dim(data, 1),
                         normalize, abs, min, max, from_r, from_c, to_r, to_c);
}

/**
 * @brief Save a SYCL matrix to a bmp file.
 *
 * The size of the image is the size of the matrix's kernel range.
 * @see write_bmp_grayscale(std::string, const Data&, unsigned, unsigned,
 * unsigned, bool, bool, T, T, unsigned, unsigned, unsigned, unsigned,
 * unsigned, unsigned)
 */
template <class T>
inline void write_bmp_grayscale_ker_rng(std::string filename, matrix_t<T>& data,
                                        bool normalize = true, bool abs = false,
                                        T min = 0, T max = 0,
                                        unsigned from_r = 0,
                                        unsigned from_c = 0, unsigned to_r = 0,
                                        unsigned to_c = 0) {
  std::vector<T> host_data(data.get_kernel_size());
  auto event = sycl_copy_device_to_host(get_eigen_device().sycl_queue(), data,
                                        host_data.data());
  event.wait_and_throw();
  write_bmp_grayscale<T>(filename, host_data.data(), access_ker_dim(data, 1),
                         access_ker_dim(data, 0), access_ker_dim(data, 1),
                         normalize, abs, min, max, from_r, from_c, to_r, to_c);
}

/**
 * @brief Save a Tensor to a bmp file with the size specified by \p r.
 *
 * @see write_bmp_grayscale(std::string, const Data&, unsigned, unsigned,
 * unsigned, bool, bool, T, T, unsigned, unsigned, unsigned, unsigned,
 * unsigned, unsigned)
 */
template <class Tensor, int DIM, class T = typename Tensor::Scalar>
inline void write_bmp_grayscale(std::string filename, Tensor t,
                                const range<DIM>& r, bool normalize = true,
                                bool abs = false, T min = 0, T max = 0,
                                unsigned from_r = 0, unsigned from_c = 0,
                                unsigned to_r = 0, unsigned to_c = 0) {
  buffer_t<T, DIM> tmp_buf(r);
  {
    auto eig_buf = sycl_to_eigen(tmp_buf);
    eig_buf.device() = t;
  }
  write_bmp_grayscale(filename, tmp_buf, normalize, abs, min, max, from_r,
                      from_c, to_r, to_c);
}

}  // namespace ml

#endif  // INCLUDE_ML_UTILS_DEBUG_WRITE_BMP_HPP
