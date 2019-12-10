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
/**
 * @file
 * @brief Allow the loading and saving of generic arrays and SYCL buffers to and
 * from disk
 */

#ifndef INCLUDE_ML_UTILS_SAVE_UTILS_HPP
#define INCLUDE_ML_UTILS_SAVE_UTILS_HPP

#include <fstream>
#include <iostream>
#include <string>

#include "ml/utils/copy.hpp"

namespace ml {

template <class T>
void save_array(const T* data, size_t length, const std::string& file_path) {
  std::cout << "Saving to " << file_path << "..." << std::endl;
  std::ofstream os(file_path.c_str(), std::ios::binary | std::ios::out);
  if (!os.is_open()) {
    std::cerr << "Could not open " << file_path << std::endl;
    return;
  }
  os.write(reinterpret_cast<const char*>(data), length * sizeof(T));
  os.close();
}

template <class T>
void load_array(T* data, size_t length, const std::string& file_path) {
  std::cout << "Loading from " << file_path << "..." << std::endl;
  std::ifstream is(file_path.c_str(), std::ios::binary | std::ios::in);
  if (!is.is_open()) {
    std::cerr << "Could not open " << file_path << std::endl;
    return;
  }
  is.read(reinterpret_cast<char*>(data), length * sizeof(T));
  is.close();
}

template <class T, int DIM>
void save_array(queue& q, buffer_t<T, DIM>& buf, const std::string& file_path) {
  std::vector<T> host_buf(buf.get_kernel_size());
  auto event = sycl_copy_device_to_host(q, buf, host_buf.data());
  event.wait_and_throw();
  save_array(host_buf.data(), host_buf.size(), file_path);
}

template <class T, int DIM>
void load_array(queue& q, buffer_t<T, DIM>& buf, const std::string& file_path) {
  std::vector<T> host_buf(buf.get_kernel_size());
  load_array(host_buf.data(), host_buf.size(), file_path);
  auto event = sycl_copy_host_to_device(q, host_buf.data(), buf);
  event.wait_and_throw();
}

}  // namespace ml

#endif  // INCLUDE_ML_UTILS_SAVE_UTILS_HPP
