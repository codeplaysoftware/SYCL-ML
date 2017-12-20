/**
 * @file
 * @brief Allow the loading and saving of generic arrays and SYCL buffers to and from disk
 */

#ifndef INCLUDE_ML_UTILS_SAVE_UTILS_HPP
#define INCLUDE_ML_UTILS_SAVE_UTILS_HPP

#include <string>
#include <fstream>
#include <iostream>

#include "ml/utils/copy.hpp"
#include "ml/utils/memory_helper.hpp"

namespace ml
{

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

template <class T>
void save_array(queue& q, sycl_vec_t<T>& buf, const std::string& file_path) {
  auto host_ptr = make_shared_array(new T[buf.get_count()]);
  sycl_copy_device_to_host(q, buf, host_ptr);
  q.wait_and_throw(); // Make sure to wait (avoid AMD driver bug?)
  save_array(host_ptr.get(), buf.get_count(), file_path);
}

template <class T>
void load_array(queue& q, sycl_vec_t<T>& buf, const std::string& file_path) {
  auto loaded_host = make_shared_array(new T[buf.get_count()]);
  load_array(loaded_host.get(), buf.get_count(), file_path);
  sycl_copy_host_to_device(q, loaded_host, buf);
  q.wait_and_throw(); // Make sure to wait (avoid AMD driver bug?)
}

} // ml

#endif //INCLUDE_ML_UTILS_SAVE_UTILS_HPP
