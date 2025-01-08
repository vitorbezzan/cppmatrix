/* tests root for newton methods */
#include "../include/cppmatrix.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace cppmatrix;

class TestFunction : public RealFunction<float> {
public:
  float operator()(float &x) { return x * x - 2.0; }
  float d1(float &x) { return 2 * x; }
  float d2(float &x) { return 2.0; }
};

TEST(newton, test_newton_real_function) {
  TestFunction f{};
  auto root = Newton(f, 0.0001, 100).run((float)std::sqrt(3));

  ASSERT_FLOAT_EQ(root, std::sqrt(2));
}

// class TestField : public ScalarField<double> {
// public:
//   double operator()(ColumnVector<double> &x) {
//     return std::pow(x(0) - 4.0, 2.0) + std::pow(x(1) + 2.0, 2.0);
//   }

//   ColumnVector<double> d1(ColumnVector<double> &x) {
//     return ColumnVector<double>({2.0 * (x(0) - 4.0), 2.0 * (x(0) + 2.0)});
//   }

//   Matrix<double> d2(ColumnVector<double> &x) { return Matrix<double>(); }
// };

// TEST(newton, test_newton_scalar_field) {
//   TestField f{};
//   auto root = Newton(f, 0.0001, 100).run(ColumnVector<double>({-4.0, 2.0}));

//   ASSERT_FLOAT_EQ(root(0), 4.0);
//   ASSERT_FLOAT_EQ(root(1), -2.0);
// }
