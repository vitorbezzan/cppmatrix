#include "../include/cppmatrix.h"
#include <cmath>
#include <gtest/gtest.h>
#include <variant>

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

class TestScalarField : public ScalarField<float> {
public:
    float operator()(ColumnVector<float> &x) {
        return x[0] * x[0] + x[1] * x[1] - 4.0;
    }

    ColumnVector<float> d1(ColumnVector<float> &x) {
        return ColumnVector<float>({2 * x[0], 2 * x[1]});
    }

    Matrix<float> d2(ColumnVector<float> &x) { return Matrix<float>(); }
};

TEST(newton, test_polyak) {
    TestScalarField f{};

    auto x0 = ColumnVector<float>({1.0, 1.0});
    auto root = Polyak(f, 0.0001, 100).run(x0);

    ASSERT_FLOAT_EQ(root[0], std::sqrt(2));
    ASSERT_FLOAT_EQ(root[1], std::sqrt(2));
}
