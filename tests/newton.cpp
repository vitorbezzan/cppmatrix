/**
 * @file newton.cpp
 * @brief Tests for Newton's method and Polyak optimization
 *
 * This file contains unit tests that verify:
 * - Newton's method for root finding in real functions
 * - Polyak optimization for scalar fields
 * - Custom function implementations (TestFunction and TestScalarField)
 * - Convergence to known solutions (e.g. sqrt(2))
 * - Gradient and Hessian computations
 */

#include "../include/cppmatrix.h"
#include <cmath>
#include <gtest/gtest.h>
#include <variant>

using namespace cppmatrix;

namespace {
    constexpr float kRootTol = 1e-4F;
    constexpr float kResidualTol = 1e-3F;
}

class TestFunction : public RealFunction<float> {
public:
    float operator()(const float& x) const override { return x * x - 2.0; }
    float d1(const float& x) const override { return 2 * x; }
    float d2(const float& x) const override { return 2.0; }
};

TEST(newton, test_newton_real_function) {
    TestFunction f{};
    auto root = Newton(f, 0.0001, 100).run((float) std::sqrt(3));

    ASSERT_NEAR(root, std::sqrt(2.0F), kRootTol);
}

class TestScalarField : public ScalarField<float> {
public:
    float operator()(const ColumnVector<float>& x) const override {
        return x[0] * x[0] + x[1] * x[1] - 4.0;
    }

    ColumnVector<float> d1(const ColumnVector<float>& x) const override {
        return ColumnVector<float>({2 * x[0], 2 * x[1]});
    }

    Matrix<float> d2(const ColumnVector<float>& x) const override { return Matrix<float>(); }
};

class CubicFunction : public RealFunction<float> {
public:
    float operator()(const float& x) const override { return x * x * x; }
    float d1(const float& x) const override { return 3.0F * x * x; }
    float d2(const float& x) const override { return 6.0F * x; }
};

class NoRealRootFunction : public RealFunction<float> {
public:
    float operator()(const float& x) const override { return x * x + 1.0F; }
    float d1(const float& x) const override { return 2.0F * x; }
    float d2(const float& x) const override { return 2.0F; }
};

class FlatScalarField : public ScalarField<float> {
public:
    float operator()(const ColumnVector<float>& x) const override {
        (void) x;
        return 1.0F;
    }

    ColumnVector<float> d1(const ColumnVector<float>& x) const override {
        (void) x;
        return ColumnVector<float>({0.0F, 0.0F});
    }

    Matrix<float> d2(const ColumnVector<float>& x) const override {
        (void) x;
        return Matrix<float>();
    }
};

TEST(newton, test_polyak) {
    TestScalarField f{};

    auto x0 = ColumnVector<float>({1.0, 1.0});
    auto root = Polyak(f, 1e-8, 100).run(x0);

    ASSERT_NEAR(root[0], std::sqrt(2.0F), kRootTol);
    ASSERT_NEAR(root[1], std::sqrt(2.0F), kRootTol);
}

TEST(newton, parallel_newton_run_multi_and_all) {
    TestFunction f{};
    ParallelNewton<TestFunction> solver(f, 1e-4F, 100);

    std::vector<float> starts = {-3.0F, 0.5F, 3.0F};
    auto best = solver.run_multi(starts);
    auto all = solver.run_multi_all(starts);

    EXPECT_EQ(all.size(), starts.size());
    EXPECT_NEAR(std::abs(best), std::sqrt(2.0F), kRootTol);
    for (float root : all) {
        EXPECT_NEAR(f(root), 0.0F, kResidualTol);
    }
}

TEST(newton, parallel_newton_empty_starting_points_throw) {
    TestFunction f{};
    ParallelNewton<TestFunction> solver(f, 1e-4F, 100);
    EXPECT_THROW((void) solver.run_multi({}), std::runtime_error);
    EXPECT_THROW((void) solver.run_multi_all({}), std::runtime_error);
}

TEST(newton, parallel_polyak_run_multi_and_empty_throw) {
    TestScalarField f{};
    ParallelPolyak<TestScalarField> solver(f, 1e-6F, 200);

    std::vector<ColumnVector<float> > starts = {
        ColumnVector<float>({1.0F, 1.0F}),
        ColumnVector<float>({2.0F, 2.0F})
    };

    auto best = solver.run_multi(starts);
    EXPECT_NEAR(f(best), 0.0F, kResidualTol);

    EXPECT_THROW((void) solver.run_multi({}), std::runtime_error);
}

TEST(newton, zero_derivative_start_can_return_non_finite) {
    CubicFunction f{};
    auto root = Newton(f, 1e-6F, 5).run(0.0F);
    EXPECT_FALSE(std::isfinite(root));
}

TEST(newton, non_convergent_function_keeps_large_residual) {
    NoRealRootFunction f{};
    auto root = Newton(f, 1e-6F, 5).run(0.5F);
    EXPECT_GT(std::abs(f(root)), 1e-2F);
}

TEST(newton, flat_gradient_polyak_can_return_non_finite) {
    FlatScalarField f{};
    auto x0 = ColumnVector<float>({1.0F, -1.0F});
    auto root = Polyak(f, 1e-6F, 5).run(x0);
    EXPECT_FALSE(std::isfinite(root[0]));
    EXPECT_FALSE(std::isfinite(root[1]));
}

