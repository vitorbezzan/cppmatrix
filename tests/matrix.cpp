/* tests root for Matrix */
#include "../include/cppmatrix.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace cppmatrix;

TEST(Matrix, base_constructor) {
  uint64_t rows = 2;
  uint64_t cols = 2;

  auto M = Matrix<float>(rows, cols, 10.0);

  for (uint64_t i = 0; i < M.rows(); i++)
    for (uint64_t j = 0; j < M.cols(); j++)
      EXPECT_FLOAT_EQ(M(i, j), 10.0);
}

float _fill(const uint64_t &i, const uint64_t &j) {
  return std::sin(i) + std::cos(j);
}

TEST(Matrix, function_constructor) {
  uint64_t rows = 2;
  uint64_t cols = 2;

  auto M = Matrix<float>(rows, cols, _fill);

  for (uint64_t i = 0; i < M.rows(); i++)
    for (uint64_t j = 0; j < M.cols(); j++)
      EXPECT_FLOAT_EQ(M(i, j), std::sin(i) + std::cos(j));
}

TEST(Matrix, copy_constructor) {
  uint64_t rows = 2;
  uint64_t cols = 2;

  auto M = Matrix<float>(rows, cols, _fill);
  auto G = Matrix(M);

  for (uint64_t i = 0; i < G.rows(); i++)
    for (uint64_t j = 0; j < G.cols(); j++)
      EXPECT_FLOAT_EQ(G(i, j), std::sin(i) + std::cos(j));
}

TEST(Matrix, sum_overload) {
  uint64_t rows = 2;
  uint64_t cols = 2;

  auto M = Matrix<float>(rows, cols, _fill);
  auto G = Matrix(M);

  auto sum = M + G;

  for (uint64_t i = 0; i < sum.rows(); i++)
    for (uint64_t j = 0; j < sum.cols(); j++)
      EXPECT_FLOAT_EQ(sum(i, j), 2 * (std::sin(i) + std::cos(j)));
}

TEST(Matrix, minus_overload) {
  uint64_t rows = 2;
  uint64_t cols = 2;

  auto M = Matrix<float>(rows, cols, _fill);
  auto G = Matrix(M);

  auto sum = M - G;

  for (uint64_t i = 0; i < sum.rows(); i++)
    for (uint64_t j = 0; j < sum.cols(); j++)
      EXPECT_FLOAT_EQ(sum(i, j), 0.0);
}

TEST(Matrix, multiply_matrix_scalar) {
  uint64_t rows = 2;
  uint64_t cols = 2;

  auto M = Matrix<float>(rows, cols, _fill);

  auto multiply = M * 2;

  for (uint64_t i = 0; i < multiply.rows(); i++)
    for (uint64_t j = 0; j < multiply.cols(); j++)
      EXPECT_FLOAT_EQ(multiply(i, j), 2 * (std::sin(i) + std::cos(j)));
}

TEST(Matrix, multiply_scalar_matrix) {
  uint64_t rows = 2;
  uint64_t cols = 2;

  auto M = Matrix<float>(rows, cols, _fill);

  auto multiply = 2 * M;

  for (uint64_t i = 0; i < multiply.rows(); i++)
    for (uint64_t j = 0; j < multiply.cols(); j++)
      EXPECT_FLOAT_EQ(multiply(i, j), 2 * (std::sin(i) + std::cos(j)));
}

TEST(Matrix, matrix_multiply) {
  auto g = NormalFill<double>(42, 0.0, 1.0);

  auto M1 = Matrix<double>(10, 10, g.filler());
  auto M2 = Matrix<double>(10, 10, g.filler());

  auto result_naive = naive_multiply(M1, M2);
  auto result_cblas = M1 * M2;

  for (uint64_t i = 0; i < result_cblas.rows(); i++)
    for (uint64_t j = 0; j < result_cblas.cols(); j++)
      EXPECT_FLOAT_EQ(result_naive(i, j), result_cblas(i, j));
}

TEST(Matrix, matrix_plus_number) {
  auto M1 = Matrix<double>(10, 10, NormalFill<double>(42, 0.0, 1.0).filler());
  auto M2(M1);

  M1 += 1.0;
  auto M3 = 1.0 + M2;

  for (uint64_t i = 0; i < M2.rows(); i++)
    for (uint64_t j = 0; j < M2.cols(); j++) {
      EXPECT_FLOAT_EQ(M1(i, j), M2(i, j) + 1.0);
      EXPECT_FLOAT_EQ(M3(i, j), M2(i, j) + 1.0);
    }
}

TEST(Matrix, matrix_minus_number) {
  auto M1 = Matrix<double>(10, 10, NormalFill<double>(42, 0.0, 1.0).filler());
  auto M2(M1);

  M1 -= 1.0;
  auto M3 = 3.0 - M2;
  auto M4 = M2 - 1.5;

  for (uint64_t i = 0; i < M2.rows(); i++)
    for (uint64_t j = 0; j < M2.cols(); j++) {
      EXPECT_FLOAT_EQ(M1(i, j), M2(i, j) - 1.0);
      EXPECT_FLOAT_EQ(M3(i, j), 3.0 - M2(i, j));
      EXPECT_FLOAT_EQ(M4(i, j), M2(i, j) - 1.5);
    }
}

TEST(Matrix, vector_fill) {
  auto M1 = ColumnVector<float>(3, VNormalFill<float>(42, 0.0, 1.0).filler());
  auto M2 = ColumnVector<float>({M1(0), M1(1), M1(2)});

  for (uint64_t i = 0; i < M2.N(); i++)
    EXPECT_FLOAT_EQ(M1(i), M2(i));
}
