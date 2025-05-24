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
    auto root = Newton(f, 0.0001, 100).run((float) std::sqrt(3));

    ASSERT_FLOAT_EQ(root, std::sqrt(2));
}
