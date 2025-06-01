#include "../include/cppmatrix.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace cppmatrix;

class Integrand : public RealFunction<float> {
public:
    float operator()(const float &x) { return std::cos(x); }
    float d1(const float &x) { return -std::sin(x); }
    float d2(const float &x) { return -std::cos(x); }
};

TEST(integral, test_base) {
    Integrand f{};
    auto value = Base1DIntegrator(0.0, M_PI / 2, 1000).run(f.get_function());

    ASSERT_NEAR(value, 1.0, 1e-3);
}

TEST(integral, test_trapezoidal_1d_integrator) {
    Integrand f{};
    auto value = Trapezoidal1DIntegrator(0.0, M_PI / 2, 1000).run(f.get_function());

    ASSERT_NEAR(value, 1.0, 1e-6);
}

TEST(integral, test_simpson_1d_integrator) {
    Integrand f{};
    auto value = Simpson1DIntegrator(0.0, M_PI / 2, 100).run(f.get_function());

    ASSERT_NEAR(value, 1.0, 1e-6);
}
