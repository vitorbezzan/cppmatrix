/**
 * @file matrix.cpp
 * @brief Tests for Matrix class operations and functionality
 * 
 * This file contains unit tests that verify:
 * - Matrix constructors (base, function-based, copy)
 * - Matrix arithmetic operations (+, -, *)
 * - Scalar operations (matrix-scalar multiplication)
 * - Vector operations (column vectors, matrix-vector multiplication)
 * - Performance comparison between naive and CBLAS implementations
 */

#include "../include/cppmatrix.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace cppmatrix;

namespace {
    constexpr double kMatmulAbsTol = 1e-8;
}

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

double _fill_linear_double(const uint64_t &i, const uint64_t &j) {
    return static_cast<double>(i * 10 + j);
}

float _fill_linear_float(const uint64_t &i, const uint64_t &j) {
    return static_cast<float>(i + 2 * j);
}

float _fill_row(const uint64_t &i, const uint64_t &j) {
    (void) i;
    return static_cast<float>(j);
}

float _fill_col(const uint64_t &i, const uint64_t &j) {
    (void) j;
    return static_cast<float>(i);
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

    auto multiply = M * 2.0;

    for (uint64_t i = 0; i < multiply.rows(); i++)
        for (uint64_t j = 0; j < multiply.cols(); j++)
            EXPECT_FLOAT_EQ(multiply(i, j), 2 * (std::sin(i) + std::cos(j)));
}

TEST(Matrix, multiply_scalar_matrix) {
    uint64_t rows = 2;
    uint64_t cols = 2;

    auto M = Matrix<float>(rows, cols, _fill);
    auto multiply = 2.0 * M;

    for (uint64_t i = 0; i < multiply.rows(); i++)
        for (uint64_t j = 0; j < multiply.cols(); j++)
            EXPECT_FLOAT_EQ(multiply(i, j), 2 * (std::sin(i) + std::cos(j)));
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

TEST(Matrix, matrix_multiply) {
    auto g = NormalFill<double>(42, 0.0, 1.0);

    auto M1 = Matrix<double>(8, 10, g.filler());
    auto M2 = Matrix<double>(10, 6, g.filler());

    auto result_naive = multiply_naive(M1, M2);
    auto result_cblas = M1 * M2;

    for (uint64_t i = 0; i < result_cblas.rows(); i++)
        for (uint64_t j = 0; j < result_cblas.cols(); j++)
            EXPECT_NEAR(result_naive(i, j), result_cblas(i, j), kMatmulAbsTol);
}

TEST(Vector, matrix_column_vector) {
    auto M = Matrix<double>(8, 10, NormalFill<double>(42, 0.0, 1.0).filler());
    auto v = ColumnVector<double>(10, VNormalFill<double>(42, 0.0, 1.0).filler());

    auto result = M * v;
    Matrix<double> convert = v;

    auto naive = multiply_naive(M, convert);

    for (int n = 0; n < static_cast<int>(result.N()); n++)
        EXPECT_NEAR(result(n), naive(n, 0), kMatmulAbsTol);
}

TEST(Vector, column_to_row_transpose) {
    auto v = ColumnVector<float>(5, VNormalFill<float>(42, 0.0, 1.0).filler());

    auto r = v.transpose();

    EXPECT_EQ(r.N(), v.N());
    EXPECT_EQ(r.rows(), 1);
    EXPECT_EQ(r.cols(), v.N());

    for (uint64_t i = 0; i < v.N(); ++i)
        EXPECT_FLOAT_EQ(v(i), r(i));
}

TEST(Vector, row_to_column_transpose) {
    auto r = RowVector<double>(6, VNormalFill<double>(42, 0.0, 1.0).filler());

    auto v = r.transpose();

    EXPECT_EQ(v.N(), r.N());
    EXPECT_EQ(v.rows(), r.N());
    EXPECT_EQ(v.cols(), 1);

    for (uint64_t i = 0; i < r.N(); ++i)
        EXPECT_DOUBLE_EQ(r(i), v(i));
}

TEST(Vector, vector_transpose_free_functions) {
    auto v = ColumnVector<float>(4, VNormalFill<float>(42, 0.0, 1.0).filler());
    auto r = RowVector<float>(4, VNormalFill<float>(43, 0.0, 1.0).filler());

    auto r_from_free = transpose(v);
    auto v_from_free = transpose(r);

    EXPECT_EQ(r_from_free.N(), v.N());
    EXPECT_EQ(v_from_free.N(), r.N());

    for (uint64_t i = 0; i < v.N(); ++i)
        EXPECT_FLOAT_EQ(v(i), r_from_free(i));

    for (uint64_t i = 0; i < r.N(); ++i)
        EXPECT_FLOAT_EQ(r(i), v_from_free(i));
}

TEST(Matrix, transpose_basic) {
    uint64_t rows = 2;
    uint64_t cols = 3;

    auto M = Matrix<double>(rows, cols, _fill_linear_double);

    auto Mt = M.transpose();

    EXPECT_EQ(Mt.rows(), cols);
    EXPECT_EQ(Mt.cols(), rows);

    for (uint64_t i = 0; i < rows; ++i)
        for (uint64_t j = 0; j < cols; ++j)
            EXPECT_DOUBLE_EQ(M(i, j), Mt(j, i));
}

TEST(Matrix, transpose_free_function) {
    uint64_t rows = 3;
    uint64_t cols = 4;

    auto M = Matrix<float>(rows, cols, _fill_linear_float);

    auto Mt = transpose(M);

    EXPECT_EQ(Mt.rows(), cols);
    EXPECT_EQ(Mt.cols(), rows);

    for (uint64_t i = 0; i < rows; ++i)
        for (uint64_t j = 0; j < cols; ++j)
            EXPECT_FLOAT_EQ(M(i, j), Mt(j, i));
}

TEST(Matrix, transpose_double_transpose_identity) {
    auto g = NormalFill<double>(42, 0.0, 1.0);
    auto M = Matrix<double>(5, 4, g.filler());

    auto Mt = M.transpose();
    auto Mtt = Mt.transpose();

    EXPECT_TRUE(M == Mtt);
}

TEST(Matrix, transpose_edge_cases) {
    auto row = Matrix<float>(1, 5, _fill_row);
    auto row_t = row.transpose();
    EXPECT_EQ(row_t.rows(), 5);
    EXPECT_EQ(row_t.cols(), 1);
    for (uint64_t j = 0; j < 5; ++j)
        EXPECT_EQ(row(0, j), row_t(j, 0));

    auto col = Matrix<float>(4, 1, _fill_col);
    auto col_t = col.transpose();
    EXPECT_EQ(col_t.rows(), 1);
    EXPECT_EQ(col_t.cols(), 4);
    for (uint64_t i = 0; i < 4; ++i)
        EXPECT_EQ(col(i, 0), col_t(0, i));

    auto single = Matrix<float>(1, 1, 3.14f);
    auto single_t = single.transpose();
    EXPECT_EQ(single_t.rows(), 1);
    EXPECT_EQ(single_t.cols(), 1);
    EXPECT_FLOAT_EQ(single(0, 0), single_t(0, 0));
}

TEST(Matrix, add_size_mismatch_throws) {
    auto left = Matrix<double>(2, 3, 1.0);
    auto right = Matrix<double>(3, 2, 2.0);
    EXPECT_THROW(left += right, std::runtime_error);
}

TEST(Matrix, subtract_size_mismatch_throws) {
    auto left = Matrix<double>(2, 3, 1.0);
    auto right = Matrix<double>(3, 2, 2.0);
    EXPECT_THROW(left -= right, std::runtime_error);
}

TEST(Matrix, multiply_shape_mismatch_throws) {
    auto left = Matrix<double>(2, 3, 1.0);
    auto right = Matrix<double>(4, 2, 2.0);
    EXPECT_THROW((void) (left * right), std::runtime_error);
}

TEST(Matrix, divide_by_zero_throws) {
    auto left = Matrix<double>(2, 2, 1.0);
    EXPECT_THROW(left /= 0.0, std::runtime_error);
}

TEST(Vector, constructor_rejects_wrong_shape_matrix) {
    auto matrix_2x2 = Matrix<double>(2, 2, 1.0);
    EXPECT_THROW((void) RowVector<double>(matrix_2x2), std::runtime_error);
    EXPECT_THROW((void) ColumnVector<double>(matrix_2x2), std::runtime_error);
}

