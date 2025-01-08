#ifndef VECTOR_H
#define VECTOR_H

#include "column_vector.h"
#include "row_vector.h"

namespace cppmatrix {

template <typename T1, typename T2>
T1 dot(RowVector<T1> &left, ColumnVector<T2> &right) {
  if (left.N() == right.N())
    return std::inner_product(left.data(), left.data() + left.N(), right.data(),
                              0.0);
  else
    throw std::runtime_error("Shape mismatch for dot.");
}

template <> float dot(RowVector<float> &left, ColumnVector<float> &right) {
  if (left.N() == right.N())
    return cblas_sdot(left.N(), left.data(), 1, right.data(), 1);
  else
    throw std::runtime_error("Shape mismatch for dot.");
}

template <> double dot(RowVector<double> &left, ColumnVector<double> &right) {
  if (left.N() == right.N())
    return cblas_ddot(left.N(), left.data(), 1, right.data(), 1);
  else
    throw std::runtime_error("Shape mismatch for dot.");
}

template <typename T1, typename T2> T1 dot(const T1 &left, const T2 &right) {
  return left * right;
}

template <typename T> T fabs(const T &v) { return std::fabs(v); }

} // namespace cppmatrix

#endif // VECTOR_H
