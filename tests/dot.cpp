/* tests root for dot product */
#include "../include/fillers.h"
#include "../include/matrix.h"
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>

using namespace cppmatrix;

TEST(dot, test_dot_float) {
  auto g = NormalFill<float>(42, 0.0, 1.0);

  auto left = RowVector<float>(10, g.filler());
  auto right = ColumnVector<float>(10, g.filler());

  auto result = dot(left, right);
  auto result_naive = std::inner_product(left.data(), left.data() + left.N(),
                                         right.data(), 0.0);

  auto l2_norm = fabs(left);
  auto l2_norm_naive = std::sqrt(std::inner_product(
      left.data(), left.data() + left.N(), left.data(), 0.0));

  ASSERT_LE(std::fabs(result - result_naive), 0.01);
  ASSERT_FLOAT_EQ(l2_norm, l2_norm_naive);
}

TEST(dot, test_dot_double) {
  auto g = NormalFill<double>(42, 0.0, 1.0);

  auto left = RowVector<double>(10, g.filler());
  auto right = ColumnVector<double>(10, g.filler());

  auto result = dot(left, right);
  auto result_naive = std::inner_product(left.data(), left.data() + left.N(),
                                         right.data(), 0.0);

  auto l2_norm = fabs(right);
  auto l2_norm_naive = std::sqrt(std::inner_product(
      right.data(), right.data() + right.N(), right.data(), 0.0));

  ASSERT_LE(std::fabs(result - result_naive), 0.01);
  ASSERT_FLOAT_EQ(l2_norm, l2_norm_naive);
}
