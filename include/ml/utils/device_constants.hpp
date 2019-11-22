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
#ifndef INCLUDE_ML_UTILS_DEVICE_CONSTANTS_HPP
#define INCLUDE_ML_UTILS_DEVICE_CONSTANTS_HPP

#include <iostream>
#include <string>

#include "ml/utils/sycl_types.hpp"

namespace ml {

/**
 * @brief Singleton that holds device specific constant.
 *
 * The user must initialize the instance before using it.\n
 * This will create a \p sycl::queue which can be retrieved with the \p
 * Eigen::SyclDevice. Note that the library assumes that only one device is used
 * for now.
 * @tparam Void Do not use, only here to avoid the use of a source file
 */
template <class Void = void>
class device_constants {
 public:
  static device_constants<Void>* instance;

  device_constants()
      : _eigen_queue(default_selector()), _eigen_device(&_eigen_queue) {
    const cl::sycl::device& sycl_device =
        _eigen_queue.sycl_queue().get_device();
    const cl::sycl::platform& platform = sycl_device.get_platform();
    using namespace cl::sycl::info;
    std::cout << "Selected device: "
              << sycl_device.get_info<info::device::name>() << ", ";
    std::cout << "type: "
              << device_type_to_str(
                     sycl_device.get_info<info::device::device_type>())
              << ", ";
    std::cout << "platform: " << platform.get_info<info::platform::name>();
    std::cout << " [" << platform.get_info<info::platform::vendor>() << "]\n";
    std::cout << std::endl;

    MAX_WORK_GROUP_SIZE =
        sycl_device.get_info<info::device::max_work_group_size>();
    MEM_BASE_ADDR_ALIGN =
        sycl_device.get_info<info::device::mem_base_addr_align>();
    MAX_WORK_ITEM_SIZES =
        sycl_device.get_info<info::device::max_work_item_sizes>();
  }

  inline size_t get_max_work_group_size() { return MAX_WORK_GROUP_SIZE; }
  inline size_t get_mem_base_addr_align() { return MEM_BASE_ADDR_ALIGN; }
  inline id<3> get_max_work_item_sizes() { return MAX_WORK_ITEM_SIZES; }

  /**
   * @tparam T
   * @return Return the value by which the size of a sub-buffer of type T must
   * be divisible.
   */
  template <class T>
  inline size_t get_sub_buffer_range_divisor() {
    return get_mem_base_addr_align() / (sizeof(T) * CHAR_BIT);
  }

  /**
   * @brief Round size up to be used by a sub-buffer.
   *
   * @see get_sub_buffer_range_divisor
   * @tparam T
   * @param size
   * @return a size usable by a sub-buffer
   */
  template <class T>
  inline size_t pad_sub_buffer_size(size_t size) {
    auto divisor = get_sub_buffer_range_divisor<T>();
    return static_cast<size_t>((size / divisor + (size % divisor > 0)) *
                               divisor);
  }

  inline Eigen::SyclDevice& get_eigen_device() { return _eigen_device; }

 private:
  size_t MAX_WORK_GROUP_SIZE;
  size_t MEM_BASE_ADDR_ALIGN;
  id<3> MAX_WORK_ITEM_SIZES;

  Eigen::QueueInterface _eigen_queue;
  Eigen::SyclDevice _eigen_device;

  inline std::string device_type_to_str(cl::sycl::info::device_type type) {
    using namespace cl::sycl::info;
    switch (type) {
      case info::device_type::cpu:
        return "CPU";
      case info::device_type::gpu:
        return "GPU";
      case info::device_type::accelerator:
        return "accelerator";
      case info::device_type::custom:
        return "custom";
      case info::device_type::automatic:
        return "automatic";
      case info::device_type::host:
        return "host";
      default:
        return "NONE";
    }
  }
};

template <>
device_constants<>* device_constants<>::instance = nullptr;

/// @brief Return the device_constants instance.
inline device_constants<>* get_device_constants() {
  return device_constants<>::instance;
}

/// @brief Return the \p Eigen::SyclDevice.
inline Eigen::SyclDevice& get_eigen_device() {
  return get_device_constants()->get_eigen_device();
}

}  // namespace ml

#endif  // INCLUDE_ML_UTILS_DEVICE_CONSTANTS_HPP
