/**
 * @file integration.cpp
 * @brief Tests for numerical integration methods
 * 
 * This file contains unit tests that verify:
 * - Base 1D integration
 * - Trapezoidal rule integration
 * - Simpson's rule integration
 * - Integration of trigonometric functions
 * - Accuracy comparison between different methods
 */

#include "../include/cppmatrix.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace cppmatrix;

/**
 * @brief Test function for integration
 * 
 * Implements f(x) = cos(x) with its derivatives
 * Used to test integration methods on a known function
 */
class Integrand : public RealFunction<float> {
public:
    float operator()(const float &x) { return std::cos(x); }
    float d1(const float &x) { return -std::sin(x); }
    float d2(const float &x) { return -std::cos(x); }
};

/**
 * @brief Tests base 1D integration method
 * 
 * Verifies integration of cos(x) from 0 to π/2
 * using the base integrator with 1000 points
 */
TEST(integral, test_base) {
    Integrand f{};
    auto value = Base1DIntegrator(0.0, M_PI / 2, 1000).run(f.get_function());

    ASSERT_NEAR(value, 1.0, 1e-3);
}

/**
 * @brief Tests trapezoidal rule integration
 * 
 * Verifies integration of cos(x) from 0 to π/2
 * using trapezoidal rule with 1000 points
 * Tests higher accuracy compared to base method
 */
TEST(integral, test_trapezoidal_1d_integrator) {
    Integrand f{};
    auto value = Trapezoidal1DIntegrator(0.0, M_PI / 2, 1000).run(f.get_function());

    ASSERT_NEAR(value, 1.0, 1e-6);
}

/**
 * @brief Tests Simpson's rule integration
 * 
 * Verifies integration of cos(x) from 0 to π/2
 * using Simpson's rule with 100 points
 * Tests higher accuracy with fewer points
 */
TEST(integral, test_simpson_1d_integrator) {
    Integrand f{};
    auto value = Simpson1DIntegrator(0.0, M_PI / 2, 100).run(f.get_function());

    ASSERT_NEAR(value, 1.0, 1e-6);
}
