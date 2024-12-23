/* tests root for Matrix */
#include "../include/matrix.h"
#include "../include/matrix_helpers.h"
#include <cmath>
#include <gtest/gtest.h>

TEST(Matrix, base_constructor) {
  uint64_t rows = 2;
  uint64_t cols = 2;

  auto M = cppmatrix::Matrix<float>(rows, cols, 10.0);

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

  auto M = cppmatrix::Matrix<float>(rows, cols, _fill);

  for (uint64_t i = 0; i < M.rows(); i++)
    for (uint64_t j = 0; j < M.cols(); j++)
      EXPECT_FLOAT_EQ(M(i, j), std::sin(i) + std::cos(j));
}

TEST(Matrix, copy_constructor) {
  uint64_t rows = 2;
  uint64_t cols = 2;

  auto M = cppmatrix::Matrix<float>(rows, cols, _fill);
  auto G = cppmatrix::Matrix(M);

  for (uint64_t i = 0; i < G.rows(); i++)
    for (uint64_t j = 0; j < G.cols(); j++)
      EXPECT_FLOAT_EQ(G(i, j), std::sin(i) + std::cos(j));
}

TEST(Matrix, sum_overload) {
  uint64_t rows = 2;
  uint64_t cols = 2;

  auto M = cppmatrix::Matrix<float>(rows, cols, _fill);
  auto G = cppmatrix::Matrix(M);

  auto sum = M + G;

  for (uint64_t i = 0; i < sum.rows(); i++)
    for (uint64_t j = 0; j < sum.cols(); j++)
      EXPECT_FLOAT_EQ(sum(i, j), 2 * (std::sin(i) + std::cos(j)));
}

TEST(Matrix, minus_overload) {
  uint64_t rows = 2;
  uint64_t cols = 2;

  auto M = cppmatrix::Matrix<float>(rows, cols, _fill);
  auto G = cppmatrix::Matrix(M);

  auto sum = M - G;

  for (uint64_t i = 0; i < sum.rows(); i++)
    for (uint64_t j = 0; j < sum.cols(); j++)
      EXPECT_FLOAT_EQ(sum(i, j), 0.0);
}

TEST(Matrix, multiply_matrix_scalar) {
  uint64_t rows = 2;
  uint64_t cols = 2;

  auto M = cppmatrix::Matrix<float>(rows, cols, _fill);

  auto multiply = M * 2;

  for (uint64_t i = 0; i < multiply.rows(); i++)
    for (uint64_t j = 0; j < multiply.cols(); j++)
      EXPECT_FLOAT_EQ(multiply(i, j), 2 * (std::sin(i) + std::cos(j)));
}

TEST(Matrix, multiply_scalar_matrix) {
  uint64_t rows = 2;
  uint64_t cols = 2;

  auto M = cppmatrix::Matrix<float>(rows, cols, _fill);

  auto multiply = 2 * M;

  for (uint64_t i = 0; i < multiply.rows(); i++)
    for (uint64_t j = 0; j < multiply.cols(); j++)
      EXPECT_FLOAT_EQ(multiply(i, j), 2 * (std::sin(i) + std::cos(j)));
}

TEST(Matrix, matrix_multiply) {
  auto M1 = cppmatrix::Matrix<float>(10, 10, cppmatrix::normal);
  auto M2 = cppmatrix::Matrix<float>(10, 10, cppmatrix::normal);

  auto result_naive = cppmatrix::naive_multiply(M1, M2);
  auto result_cblas = M1 * M2;

  for (uint64_t i = 0; i < result_cblas.rows(); i++)
    for (uint64_t j = 0; j < result_cblas.cols(); j++)
      EXPECT_FLOAT_EQ(result_naive(i, j), result_cblas(i, j));
}
