#ifndef EXAMPLE_SRC_UTILS_SCOPED_TIMER_HPP
#define EXAMPLE_SRC_UTILS_SCOPED_TIMER_HPP

#include <string>
#include <iostream>
#include <chrono>

/**
 * @brief scoped_timer measures and print the time between the creation and destruction of the object.
 * Also print with an indentation when several scoped_timer are used.
 */
class scoped_timer {
using sc = std::chrono::high_resolution_clock;

public:
  scoped_timer(const std::string& name) : _name(name) {
    std::cout << std::string(indent, ' ') << "Starting " << _name << std::endl;
    indent += 2;
    _t0 = sc::now();
  }

  ~scoped_timer() {
    std::chrono::duration<double> diff = sc::now() - _t0;
    indent -= 2;
    std::cout << std::string(indent, ' ') << _name << ": " << diff.count() << "s" << std::endl;
  }

private:
  static unsigned indent;
  std::string _name;
  sc::time_point _t0;
};

unsigned scoped_timer::indent = 0;

/**
 * @brief Create a timer and mark the variable unused.
 */
#define TIME(name) scoped_timer _timer_##name(#name); (void)_timer_##name

#endif //EXAMPLE_SRC_UTILS_SCOPED_TIMER_HPP
