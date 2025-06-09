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

/**
 * @brief Tests the base constructor of Matrix class
 * 
 * Verifies that a matrix initialized with a constant value
 * has all elements set to that value
 */
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

/**
 * @brief Tests the function-based constructor of Matrix class
 * 
 * Verifies that a matrix initialized with a function
 * correctly applies that function to each element
 */
TEST(Matrix, function_constructor) {
    uint64_t rows = 2;
    uint64_t cols = 2;

    auto M = Matrix<float>(rows, cols, _fill);

    for (uint64_t i = 0; i < M.rows(); i++)
        for (uint64_t j = 0; j < M.cols(); j++)
            EXPECT_FLOAT_EQ(M(i, j), std::sin(i) + std::cos(j));
}

/**
 * @brief Tests the copy constructor of Matrix class
 * 
 * Verifies that copying a matrix creates an exact duplicate
 * with all elements preserved
 */
TEST(Matrix, copy_constructor) {
    uint64_t rows = 2;
    uint64_t cols = 2;

    auto M = Matrix<float>(rows, cols, _fill);
    auto G = Matrix(M);

    for (uint64_t i = 0; i < G.rows(); i++)
        for (uint64_t j = 0; j < G.cols(); j++)
            EXPECT_FLOAT_EQ(G(i, j), std::sin(i) + std::cos(j));
}

/**
 * @brief Tests matrix addition operation
 * 
 * Verifies that adding two matrices element-wise
 * produces the expected result
 */
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

/**
 * @brief Tests matrix subtraction operation
 * 
 * Verifies that subtracting two identical matrices
 * results in a zero matrix
 */
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

/**
 * @brief Tests matrix-scalar multiplication
 * 
 * Verifies that multiplying a matrix by a scalar
 * correctly scales all elements
 */
TEST(Matrix, multiply_matrix_scalar) {
    uint64_t rows = 2;
    uint64_t cols = 2;

    auto M = Matrix<float>(rows, cols, _fill);

    auto multiply = M * 2.0;

    for (uint64_t i = 0; i < multiply.rows(); i++)
        for (uint64_t j = 0; j < multiply.cols(); j++)
            EXPECT_FLOAT_EQ(multiply(i, j), 2 * (std::sin(i) + std::cos(j)));
}

/**
 * @brief Tests scalar-matrix multiplication
 * 
 * Verifies that multiplying a scalar by a matrix
 * correctly scales all elements
 */
TEST(Matrix, multiply_scalar_matrix) {
    uint64_t rows = 2;
    uint64_t cols = 2;

    auto M = Matrix<float>(rows, cols, _fill);
    auto multiply = 2.0 * M;

    for (uint64_t i = 0; i < multiply.rows(); i++)
        for (uint64_t j = 0; j < multiply.cols(); j++)
            EXPECT_FLOAT_EQ(multiply(i, j), 2 * (std::sin(i) + std::cos(j)));
}

/**
 * @brief Tests matrix addition with scalar
 * 
 * Verifies both in-place (+=) and non-in-place addition
 * of a scalar to all matrix elements
 */
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

/**
 * @brief Tests matrix subtraction with scalar
 * 
 * Verifies both in-place (-=) and non-in-place subtraction
 * of a scalar from all matrix elements
 */
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

/**
 * @brief Tests vector initialization and construction
 * 
 * Verifies that column vectors can be created using
 * both random fill and explicit element values
 */
TEST(Matrix, vector_fill) {
    auto M1 = ColumnVector<float>(3, VNormalFill<float>(42, 0.0, 1.0).filler());
    auto M2 = ColumnVector<float>({M1(0), M1(1), M1(2)});

    for (uint64_t i = 0; i < M2.N(); i++)
        EXPECT_FLOAT_EQ(M1(i), M2(i));
}

/**
 * @brief Tests matrix multiplication
 * 
 * Verifies that matrix multiplication results match
 * between naive and CBLAS implementations
 */
TEST(Matrix, matrix_multiply) {
    auto g = NormalFill<double>(42, 0.0, 1.0);

    auto M1 = Matrix<double>(8, 10, g.filler());
    auto M2 = Matrix<double>(10, 6, g.filler());

    auto result_naive = naive_multiply(M1, M2);
    auto result_cblas = M1 * M2;

    for (uint64_t i = 0; i < result_cblas.rows(); i++)
        for (uint64_t j = 0; j < result_cblas.cols(); j++)
            EXPECT_FLOAT_EQ(result_naive(i, j), result_cblas(i, j));
}

/**
 * @brief Tests matrix-vector multiplication
 * 
 * Verifies that multiplying a matrix by a column vector
 * produces correct results compared to naive implementation
 */
TEST(Vector, matrix_column_vector) {
    auto M = Matrix<double>(8, 10, NormalFill<double>(42, 0.0, 1.0).filler());
    auto v = ColumnVector<double>(10, VNormalFill<double>(42, 0.0, 1.0).filler());

    auto result = M * v;
    Matrix<double> convert = v;

    print(M);

    auto naive = naive_multiply(M, convert);

    for (int n = 0; n < v.N(); n++)
        EXPECT_FLOAT_EQ(result(n), naive(n, 0));
}
