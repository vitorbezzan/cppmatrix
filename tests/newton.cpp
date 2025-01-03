/* tests root for newton methods */
#include "../include/newton.h"
#include "../include/function.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace cppmatrix;

class Function : public RealFunction<float> {
public:
  float operator()(const float &x) { return x * x - 2.0; }

  float d1(const float &x) { return 2 * x; }

  float d2(const float &x) { return 2.0; }
};

TEST(newton, test_newton_real_function) {
  Function f{};
  auto root = Newton(f, 0.0001, 100).run(std::sqrt(3));

  ASSERT_FLOAT_EQ(root, std::sqrt(2));
}
