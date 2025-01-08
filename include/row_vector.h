#ifndef ROW_VECTOR_H
#define ROW_VECTOR_H

#include "matrix.h"
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace cppmatrix {

template <typename T> class RowVector : public Matrix<T> {
public:
  RowVector(const uint64_t &N) : Matrix<T>(1, N) { this->_N = N; }
  RowVector(const uint64_t &N, const T &value) : Matrix<T>(1, N, value) {
    this->_N = N;
  }
  RowVector(const uint64_t &N, T (*f)(const uint64_t &, const uint64_t &))
      : Matrix<T>(1, N, f) {
    this->_N = N;
  }
  RowVector(const uint64_t &N,
            const std::function<T(const uint64_t &, const uint64_t &)> &f)
      : Matrix<T>(1, N, f) {
    this->_N = N;
  }

  template <uint64_t ndim>
  RowVector(const T (&values)[ndim]) : Matrix<T>(1, N) {
    this->_N = ndim;
    std::copy(values, values + ndim, this->data());
  }

  RowVector(const uint64_t &N, T (*f)(const uint64_t &)) : Matrix<T>(1, N) {
    this->_N = N;
    for (uint64_t n = 0; n < this->_N; n++)
      this->operator()(n) = f(n);
  }

  RowVector(const uint64_t &N, const std::function<T(const uint64_t &)> &f)
      : Matrix<T>(1, N) {
    this->_N = N;
    for (uint64_t n = 0; n < this->_N; n++)
      this->operator()(n) = f(n);
  }

  RowVector() : Matrix<T>() {}

  T &operator()(uint64_t n) { return Matrix<T>::operator()(0, n); }

  uint64_t N() { return this->_N; }

private:
  uint64_t _N;
};

template <typename T1, typename T2>
T1 dot(RowVector<T1> &left, RowVector<T2> &right) {
  if (left.N() == right.N())
    return std::inner_product(left.data(), left.data() + left.N(), right.data(),
                              0.0);
  else
    throw std::runtime_error("Shape mismatch for dot.");
}

template <> float dot(RowVector<float> &left, RowVector<float> &right) {
  if (left.N() == right.N())
    return cblas_sdot(left.N(), left.data(), 1, right.data(), 1);
  else
    throw std::runtime_error("Shape mismatch for dot.");
}

template <> double dot(RowVector<double> &left, RowVector<double> &right) {
  if (left.N() == right.N())
    return cblas_ddot(left.N(), left.data(), 1, right.data(), 1);
  else
    throw std::runtime_error("Shape mismatch for dot.");
}

template <typename T> T fabs(RowVector<T> &v) { return std::sqrt(dot(v, v)); }

} // namespace cppmatrix

#endif // ROW_VECTOR_H
