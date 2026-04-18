/**
 * @file integration.cpp
 * @brief Tests for numerical integration (RealFunction + integrator templates).
 *
 * Integrators follow the same pattern as Newton: template<IsRealFunction F>, store F,
 * call run() with no separate std::function argument.
 */

#include "../include/cppmatrix.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace cppmatrix;

namespace {
    constexpr float kPiHalfF = static_cast<float>(M_PI) / 2.0F;
    constexpr double kPiHalfD = M_PI / 2.0;
}

class Integrand : public RealFunction<float> {
public:
    float operator()(const float& x) const override { return std::cos(x); }
    float d1(const float& x) const override { return -std::sin(x); }
    float d2(const float& x) const override { return -std::cos(x); }
};

class IntegrandDouble : public RealFunction<double> {
public:
    double operator()(const double& x) const override { return std::cos(x); }
    double d1(const double& x) const override { return -std::sin(x); }
    double d2(const double& x) const override { return -std::cos(x); }
};

TEST(integral, riemann_real_function_float) {
    Integrand f{};
    auto value = Riemann1DIntegrator(f, 0.0F, kPiHalfF, 1000).run();

    ASSERT_NEAR(value, 1.0, 5e-3);
}

TEST(integral, trapezoidal_real_function_float) {
    Integrand f{};
    auto value = Trapezoidal1DIntegrator(f, 0.0F, kPiHalfF, 1000).run();

    ASSERT_NEAR(value, 1.0, 5e-5);
}

TEST(integral, simpson_real_function_float) {
    Integrand f{};
    auto value = Simpson1DIntegrator(f, 0.0F, kPiHalfF, 100).run();

    ASSERT_NEAR(value, 1.0, 5e-5);
}

TEST(integral, trapezoidal_real_function_double) {
    IntegrandDouble f{};
    auto value = Trapezoidal1DIntegrator(f, 0.0, kPiHalfD, 1000).run();

    ASSERT_NEAR(value, 1.0, 1e-6);
}

TEST(integral, zero_interval_returns_zero) {
    Integrand f{};

    auto riemann = Riemann1DIntegrator(f, 1.0F, 1.0F, 100).run();
    auto trapezoidal = Trapezoidal1DIntegrator(f, 1.0F, 1.0F, 100).run();
    auto simpson = Simpson1DIntegrator(f, 1.0F, 1.0F, 100).run();

    EXPECT_FLOAT_EQ(riemann, 0.0F);
    EXPECT_FLOAT_EQ(trapezoidal, 0.0F);
    EXPECT_FLOAT_EQ(simpson, 0.0F);
}

