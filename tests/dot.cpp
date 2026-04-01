/**
 * @file dot.cpp
 * @brief Tests for dot product and vector norm operations
 * 
 * This file contains unit tests that verify:
 * - Dot product operations between row and column vectors
 * - L2 norm calculations for vectors
 * - Tests are performed for both float and double precision
 * - Results are compared against naive implementations using std::inner_product
 */

#include "../include/cppmatrix.h"
#include <cmath>
#include <numeric>
#include <gtest/gtest.h>

using namespace cppmatrix;

namespace {
    constexpr float kDotAbsTolFloat = 1e-4F;
    constexpr double kDotAbsTolDouble = 1e-10;

    template<typename T>
    void expect_dot_and_norm_match_naive(const T abs_tol) {
        auto g = VNormalFill<T>(42, T(0), T(1));
        auto left = RowVector<T>(10, g.filler());
        auto right = ColumnVector<T>(10, g.filler());

        const auto result = dot(left, right);
        const auto result_naive = std::inner_product(left.data(), left.data() + left.N(),
                                                     right.data(), T(0));

        const auto l2_norm = norm(left);
        const auto l2_norm_naive = std::sqrt(std::inner_product(
            left.data(), left.data() + left.N(), left.data(), T(0)));

        ASSERT_NEAR(result, result_naive, abs_tol);
        ASSERT_NEAR(l2_norm, l2_norm_naive, abs_tol);
    }
}

TEST(dot, test_dot_float) {
    expect_dot_and_norm_match_naive<float>(kDotAbsTolFloat);
}

TEST(dot, test_dot_double) {
    expect_dot_and_norm_match_naive<double>(kDotAbsTolDouble);
}

TEST(dot, mismatch_throws) {
    auto left = RowVector<double>(4, 1.0);
    auto right = ColumnVector<double>(3, 2.0);
    EXPECT_THROW((void) dot(left, right), std::runtime_error);
}
