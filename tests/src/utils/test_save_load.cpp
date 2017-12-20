#include <iostream>

#include "ml/utils/buffer_t.hpp"
#include "utils/utils.hpp"

template <class T>
void test_save_load_host() {
  constexpr size_t SIZE = 4;
  std::array<T, SIZE> buf {-1, 0, -1.5, 0.5};
  std::array<T, SIZE> res;

  ml::save_array(buf.data(), SIZE, "test_buf");
  ml::load_array(res.data(), SIZE, "test_buf");

  std::cout << "Saved: ";
  ml::print(buf, 1, SIZE);
  std::cout << "Loaded: ";
  ml::print(res, 1, SIZE);

  assert_vec_almost_eq(res, buf);
}

template <class T>
void test_save_load_device() {
  constexpr size_t SIZE = 6;
  std::array<T, SIZE> buf {-10, 0, -1.5, 3, 3, 1};
  std::array<T, SIZE> res;

  {
    cl::sycl::queue& q = create_queue();
    {
      ml::matrix_t<T> sycl_buf(const_cast<const T*>(buf.data()), cl::sycl::range<2>(2, 3));
      ml::save_array(q, sycl_buf, "test_buf");
    }
    ml::matrix_t<T> sycl_res(cl::sycl::range<2>(2, 3));
    ml::load_array(q, sycl_res, "test_buf");

    sycl_res.set_final_data(res.data());
    clear_eigen_device();
  }

  std::cout << "Saved: ";
  ml::print(buf, 1, SIZE);
  std::cout << "Loaded: ";
  ml::print(res, 1, SIZE);

  assert_vec_almost_eq(res, buf);
}

int main() {
  test_save_load_host<float>();
  test_save_load_host<double>();
  test_save_load_device<float>();
  test_save_load_device<double>();

  return 0;
}

