/**
 * @file ndarray.cpp
 * @brief Tests for N-dimensional array operations
 *
 * This file contains unit tests that verify:
 * - NDArray constructors and basic operations
 * - Type handling (float and double precision)
 * - Arithmetic operations (+, -, *)
 * - Mixed type operations (float-double interactions)
 * - Reference and non-reference operations
 * - Multi-dimensional indexing and access
 */

#include "../include/cppmatrix.h"
#include <gtest/gtest.h>
#include <limits>

using namespace cppmatrix;

TEST(NDArray, base_constructor_float) {
    float value = 10.0;

    auto A = NDArray<float>({2, 2, 2}, value);
    auto data = A.data();

    EXPECT_EQ(A.ndim(), 3);
    EXPECT_EQ(A.N(), 8);

    for (size_t i = 0; i < A.N(); i++) {
        EXPECT_FLOAT_EQ(data[i], value);
    }
}

TEST(NDArray, base_operator_reference_float) {
    double value = 10.0;

    auto A = NDArray<float>({2, 2, 2}, value);
    auto data = A.data();

    EXPECT_EQ(A.ndim(), 3);
    EXPECT_EQ(A.N(), 8);

    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            for (size_t k = 0; k < 2; k++) {
                uint64_t index[] = {i, j, k};
                EXPECT_FLOAT_EQ(A(index), value);
            }
}

TEST(NDArray, base_constructor_double) {
    double value = 10.0;

    auto A = NDArray<double>({2, 2, 2}, value);
    auto data = A.data();

    EXPECT_EQ(A.ndim(), 3);
    EXPECT_EQ(A.N(), 8);

    for (size_t i = 0; i < A.N(); i++) {
        EXPECT_FLOAT_EQ(data[i], value);
    }
}

TEST(NDArray, base_operator_reference_double) {
    float value = 10.0;

    auto A = NDArray<double>({2, 2, 2}, value);
    auto data = A.data();

    EXPECT_EQ(A.ndim(), 3);
    EXPECT_EQ(A.N(), 8);

    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            for (size_t k = 0; k < 2; k++) {
                uint64_t index[] = {i, j, k};
                EXPECT_FLOAT_EQ(A(index), value);
            }
}

TEST(NDArray, equals_float_float) {
    float value = 10.0;

    auto A = NDArray<float>({2, 2, 2}, value);
    auto B = NDArray<float>();

    B = A;

    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            for (size_t k = 0; k < 2; k++) {
                uint64_t index[] = {i, j, k};
                EXPECT_FLOAT_EQ(B(index), value);
            }
}

TEST(NDArray, equals_double_double) {
    double value = 10.0;

    auto A = NDArray<float>({2, 2, 2}, value);
    auto B = NDArray<float>();

    B = A;

    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            for (size_t k = 0; k < 2; k++) {
                uint64_t index[] = {i, j, k};
                EXPECT_FLOAT_EQ(B(index), value);
            }
}

TEST(NDArray, sum_ndarray_float_float) {
    float value1 = 1.0;
    float value2 = 2.0;

    auto A = NDArray<float>({2, 2, 2}, value1);
    auto B = NDArray<float>({2, 2, 2}, value2);

    A += B;
    auto C = A + B;

    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            for (size_t k = 0; k < 2; k++) {
                uint64_t index[] = {i, j, k};
                EXPECT_FLOAT_EQ(A(index), value1 + value2);
                EXPECT_FLOAT_EQ(C(index), value1 + 2 * value2);
            }
}

TEST(NDArray, sum_ndarray_double_double) {
    double value1 = 1.0;
    double value2 = 2.0;

    auto A = NDArray<double>({2, 2, 2}, value1);
    auto B = NDArray<double>({2, 2, 2}, value2);

    A += B;

    auto C = A + B;

    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            for (size_t k = 0; k < 2; k++) {
                uint64_t index[] = {i, j, k};
                EXPECT_FLOAT_EQ(A(index), value1 + value2);
                EXPECT_FLOAT_EQ(C(index), value1 + 2 * value2);
            }
}

TEST(NDArray, sum_ndarray_double_float) {
    double value1 = 1.0;
    double value2 = 2.0;

    auto A = NDArray<double>({2, 2, 2}, value1);
    auto B = NDArray<float>({2, 2, 2}, value2);

    A += B;

    auto C = A + B;

    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            for (size_t k = 0; k < 2; k++) {
                uint64_t index[] = {i, j, k};
                EXPECT_FLOAT_EQ(A(index), value1 + value2);
                EXPECT_FLOAT_EQ(C(index), value1 + 2 * value2);
            }
}

TEST(NDArray, diff_ndarray_double_float) {
    double value1 = 1.0;
    double value2 = 2.0;

    auto A = NDArray<double>({2, 2, 2}, value2);
    auto B = NDArray<float>({2, 2, 2}, value1);

    A -= B;

    auto C = A - B;

    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            for (size_t k = 0; k < 2; k++) {
                uint64_t index[] = {i, j, k};
                EXPECT_FLOAT_EQ(A(index), value2 - value1);
                EXPECT_FLOAT_EQ(C(index), 0.0);
            }
}

TEST(NDArray, mult_ndarray_double_double) {
    double value2 = 2.0;

    auto A = NDArray<double>({2, 2, 2}, value2);

    A *= value2;

    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            for (size_t k = 0; k < 2; k++) {
                uint64_t index[] = {i, j, k};
                EXPECT_FLOAT_EQ(A(index), value2 * value2);
            }
}

TEST(NDArray, mult_ndarray_double_float) {
    float value2 = 2.0;

    auto A = NDArray<double>({2, 2, 2}, value2);

    A *= value2;

    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            for (size_t k = 0; k < 2; k++) {
                uint64_t index[] = {i, j, k};
                EXPECT_FLOAT_EQ(A(index), value2 * value2);
            }
}

TEST(NDArray, mult_ndarray_double_float_noref) {
    float value2 = 2.0;

    auto A = NDArray<double>({2, 2, 2}, value2);
    auto C = A * value2;

    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            for (size_t k = 0; k < 2; k++) {
                uint64_t index[] = {i, j, k};
                EXPECT_FLOAT_EQ(C(index), value2 * value2);
            }
}

TEST(NDArray, mult_ndarray_float_double_noref) {
    float value2 = 2.0;
    double value1 = 1.0;

    auto A = NDArray<double>({2, 2, 2}, value2);
    auto B = NDArray<double>({2, 2, 2}, value1);
    auto C = (value2 * A) + B;

    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            for (size_t k = 0; k < 2; k++) {
                uint64_t index[] = {i, j, k};
                EXPECT_FLOAT_EQ(C(index), (value2 * value2) + value1);
            }
}

TEST(NDArray, sum_shape_mismatch_throws) {
    double value = 1.0;
    auto A = NDArray<double>({2, 2, 2}, value);
    auto B = NDArray<double>({2, 2, 3}, value);
    EXPECT_THROW(A += B, std::runtime_error);
}

TEST(NDArray, divide_by_zero_throws) {
    double value = 1.0;
    auto A = NDArray<double>({2, 2, 2}, value);
    EXPECT_THROW(A /= 0.0, std::runtime_error);
}

TEST(NDArraySecurity, indexing_rank_mismatch_throws_even_unchecked) {
    double value = 1.0;
    auto A = NDArray<double>({2, 2, 2}, value);

    uint64_t short_index[] = {0, 0};
    EXPECT_THROW((void)A(short_index), std::out_of_range);
}

TEST(NDArraySecurity, at_bounds_check_throws) {
    double value = 1.0;
    auto A = NDArray<double>({2, 2, 2}, value);

    uint64_t bad_index[] = {0, 0, 2};
    EXPECT_THROW((void)A.at(bad_index), std::out_of_range);
}

TEST(NDArraySecurity, shape_overflow_throws) {
    // Force element-count overflow in size_t multiplication or exceed CPPMATRIX_MAX_BYTES cap.
    // Using max/2 ensures multiplication overflows or produces a huge byte count.
    uint64_t big = static_cast<uint64_t>((std::numeric_limits<std::size_t>::max)() / 2);
    EXPECT_THROW((void)NDArray<double>({big, big}), std::exception);
}

TEST(NDArray, unary_minus_member_and_free) {
    double value = 3.0;
    auto A = NDArray<double>({2, 2}, value);
    auto neg_member = -A;
    auto neg_free = operator-(A);

    EXPECT_TRUE(A == -neg_member);
    EXPECT_TRUE(neg_member == neg_free);

    for (uint64_t i = 0; i < 2; ++i)
        for (uint64_t j = 0; j < 2; ++j) {
            uint64_t index[] = {i, j};
            EXPECT_DOUBLE_EQ(neg_member(index), -3.0);
        }
}

TEST(NDArray, scalar_minus_ndarray_const) {
    double value = 2.0;
    const auto A = NDArray<double>({2, 2}, value);
    auto result = 5.0 - A;

    for (uint64_t i = 0; i < 2; ++i)
        for (uint64_t j = 0; j < 2; ++j) {
            uint64_t index[] = {i, j};
            EXPECT_DOUBLE_EQ(result(index), 3.0);
        }
}

TEST(NDArray, divide_ndarray_scalar) {
    double value = 8.0;
    auto A = NDArray<double>({2, 2}, value);
    A /= 2.0;
    auto B = A / 4.0;

    for (uint64_t i = 0; i < 2; ++i)
        for (uint64_t j = 0; j < 2; ++j) {
            uint64_t index[] = {i, j};
            EXPECT_DOUBLE_EQ(A(index), 4.0);
            EXPECT_DOUBLE_EQ(B(index), 1.0);
        }
}

