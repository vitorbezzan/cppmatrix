#ifndef COLUMN_VECTOR_H
#define COLUMN_VECTOR_H

#include "matrix.h"
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace cppmatrix {

template <typename T> class ColumnVector : public Matrix<T> {
public:
  ColumnVector(const uint64_t &N) : Matrix<T>(N, 1) { this->_N = N; }
  ColumnVector(const uint64_t &N, const T &value) : Matrix<T>(N, 1, value) {
    this->_N = N;
  }
  ColumnVector(const uint64_t &N, T (*f)(const uint64_t &, const uint64_t &))
      : Matrix<T>(N, 1, f) {
    this->_N = N;
  }
  ColumnVector(const uint64_t &N,
               const std::function<T(const uint64_t &, const uint64_t &)> &f)
      : Matrix<T>(N, 1, f) {
    this->_N = N;
  }

  template <uint64_t ndim>
  ColumnVector(const T (&values)[ndim]) : Matrix<T>(ndim, 1) {
    this->_N = ndim;
    std::copy(values, values + ndim, this->data());
  }

  ColumnVector(const uint64_t &N, T (*f)(const uint64_t &)) : Matrix<T>(N, 1) {
    this->_N = N;
    for (uint64_t n = 0; n < this->_N; n++)
      this->operator()(n) = f(n);
  }

  ColumnVector(const uint64_t &N, const std::function<T(const uint64_t &)> &f)
      : Matrix<T>(N, 1) {
    this->_N = N;
    for (uint64_t n = 0; n < this->_N; n++)
      this->operator()(n) = f(n);
  }

  ColumnVector() : Matrix<T>() {}

  T &operator()(uint64_t n) { return Matrix<T>::operator()(n, 0); }

  uint64_t N() { return this->_N; }

private:
  uint64_t _N;
};

template <typename T1, typename T2>
T1 dot(ColumnVector<T1> &left, ColumnVector<T2> &right) {
  if (left.N() == right.N())
    return std::inner_product(left.data(), left.data() + left.N(), right.data(),
                              0.0);
  else
    throw std::runtime_error("Shape mismatch for dot.");
}

template <> float dot(ColumnVector<float> &left, ColumnVector<float> &right) {
  if (left.N() == right.N())
    return cblas_sdot(left.N(), left.data(), 1, right.data(), 1);
  else
    throw std::runtime_error("Shape mismatch for dot.");
}

template <>
double dot(ColumnVector<double> &left, ColumnVector<double> &right) {
  if (left.N() == right.N())
    return cblas_ddot(left.N(), left.data(), 1, right.data(), 1);
  else
    throw std::runtime_error("Shape mismatch for dot.");
}

template <typename T> T fabs(ColumnVector<T> &v) {
  return std::sqrt(dot(v, v));
}

} // namespace cppmatrix

#endif // COLUMN_VECTOR_H
