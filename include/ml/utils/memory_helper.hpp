#ifndef INCLUDE_ML_UTILS_MEMORY_HELPER_HPP
#define INCLUDE_ML_UTILS_MEMORY_HELPER_HPP

#include <memory>

namespace ml
{

/**
 * @brief Create shared_ptr<T> for array when the type shared_ptr<T[]> is not usable.
 *
 * @tparam T
 * @param ptr
 * @return shared_ptr with a custom Deleter
 */
template <class T>
inline std::shared_ptr<T> make_shared_array(T* ptr) {
  return std::shared_ptr<T>(ptr, std::default_delete<T[]>());
}

} // ml

#endif //INCLUDE_ML_UTILS_MEMORY_HELPER_HPP
