/* tests root for NDArray */
#include "../include/cppmatrix.h"
#include <gtest/gtest.h>

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
