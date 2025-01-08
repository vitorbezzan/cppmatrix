#ifndef MATRIX_H
#define MATRIX_H

#include "ndarray.h"
#include <algorithm>
#include <cblas.h>
#include <functional>
#include <stdexcept>
#include <type_traits>

namespace cppmatrix {

template <typename T>
  requires std::is_floating_point_v<T>
class Matrix : public NDArray<T> {
public:
  template <typename U>
    requires std::is_floating_point_v<U>
  friend class Matrix;

  Matrix() : NDArray<T>() {};

  Matrix(const uint64_t &rows, const uint64_t &cols)
      : NDArray<T>({rows, cols}) {
    this->_rows = rows;
    this->_cols = cols;
  }

  Matrix(const uint64_t &rows, const uint64_t &cols, const T &value)
      : NDArray<T>({rows, cols}, value) {
    this->_rows = rows;
    this->_cols = cols;
  }

  Matrix(const uint64_t &rows, const uint64_t &cols,
         T (*f)(const uint64_t &, const uint64_t &))
      : NDArray<T>({rows, cols}) {
    this->_rows = rows;
    this->_cols = cols;

    for (uint64_t row = 0; row < this->_rows; row++)
      for (uint64_t col = 0; col < this->_cols; col++)
        this->operator()(row, col) = f(row, col);
  }

  Matrix(const uint64_t &rows, const uint64_t &cols,
         const std::function<T(const uint64_t &, const uint64_t &)> &f)
      : NDArray<T>({rows, cols}) {
    this->_rows = rows;
    this->_cols = cols;

    for (uint64_t row = 0; row < this->_rows; row++)
      for (uint64_t col = 0; col < this->_cols; col++)
        this->operator()(row, col) = f(row, col);
  }

  Matrix(const Matrix<T> &M) : NDArray<T>(M) {
    this->_rows = M._rows;
    this->_cols = M._cols;
  }

  Matrix(const NDArray<T> &base) : NDArray<T>(base) {
    if (base.ndim() != 2)
      throw std::runtime_error("Dimension size mismatch.");

    this->_rows = base.shape()[0];
    this->_cols = base.shape()[1];
  }

  T &operator()(uint64_t row, uint64_t col) {
    uint64_t index[2] = {row, col};
    return NDArray<T>::operator()(index);
  }

  uint64_t rows() const { return _rows; }

  uint64_t cols() const { return _cols; }

  Matrix<T> &operator*=(const T &right) {
    std::transform(
        this->data(), this->data() + this->N(), this->data(),
        std::bind(std::multiplies<T>(), std::placeholders::_1, right));
    return *this;
  }

  Matrix<T> operator*(const T &right) { return Matrix<T>(*this) *= right; }

private:
  uint64_t _rows;
  uint64_t _cols;
};

template <typename T1, typename T2>
Matrix<T1> &operator+=(Matrix<T1> &left, const Matrix<T2> &right) {
  if ((left.cols() != right.cols()) | (left.rows() != right.rows())) {
    throw std::runtime_error("Size mismatch for operator+=");
  }

  std::transform(left.data(), left.data() + left.N(), right.data(), left.data(),
                 std::plus<T1>());

  return left;
}

template <typename T1>
Matrix<T1> &operator+=(Matrix<T1> &left, const float &right) {
  std::transform(left.data(), left.data() + left.N(), left.data(),
                 [right](float element) { return element + right; });
  return left;
}

template <typename T1>
Matrix<T1> &operator+=(Matrix<T1> &left, const double &right) {
  std::transform(left.data(), left.data() + left.N(), left.data(),
                 [right](double element) { return element + right; });
  return left;
}

template <typename T1, typename T2>
Matrix<T1> operator+(const Matrix<T1> &left, const Matrix<T2> &right) {
  auto result = Matrix(left);
  operator+=(result, right);

  return result;
}

template <typename T1>
Matrix<T1> operator+(Matrix<T1> &left, const float &right) {
  auto result = Matrix(left);
  operator+=(result, right);

  return result;
}

template <typename T1>
Matrix<T1> operator+(Matrix<T1> &left, const double &right) {
  auto result = Matrix(left);
  operator+=(result, right);

  return result;
}

template <typename T1>
Matrix<T1> operator+(const float &left, Matrix<T1> &right) {
  auto result = Matrix(right);
  operator+=(result, left);

  return result;
}

template <typename T1>
Matrix<T1> operator+(const double &left, Matrix<T1> &right) {
  auto result = Matrix(right);
  operator+=(result, left);

  return result;
}

template <typename T1, typename T2>
Matrix<T1> &operator-=(Matrix<T1> &left, const Matrix<T2> &right) {
  if ((left.cols() != right.cols()) | (left.rows() != right.rows())) {
    throw std::runtime_error("Size mismatch for operator-=");
  }

  std::transform(left.data(), left.data() + left.N(), right.data(), left.data(),
                 std::minus<T1>());

  return left;
}

template <typename T1>
Matrix<T1> &operator-=(Matrix<T1> &left, const float &right) {
  std::transform(left.data(), left.data() + left.N(), left.data(),
                 [right](float element) { return element - right; });
  return left;
}

template <typename T1>
Matrix<T1> &operator-=(Matrix<T1> &left, const double &right) {
  std::transform(left.data(), left.data() + left.N(), left.data(),
                 [right](double element) { return element - right; });
  return left;
}

template <typename T1, typename T2>
Matrix<T1> operator-(const Matrix<T1> &left, const Matrix<T2> &right) {
  auto result = Matrix(left);
  operator-=(result, right);

  return result;
}

template <typename T1>
Matrix<T1> operator-(Matrix<T1> &left, const float &right) {
  auto result = Matrix(left);
  operator-=(result, right);

  return result;
}

template <typename T1>
Matrix<T1> operator-(Matrix<T1> &left, const double &right) {
  auto result = Matrix(left);
  operator-=(result, right);

  return result;
}

template <typename T1>
Matrix<T1> operator-(const float &left, Matrix<T1> &right) {
  auto result = Matrix(right) * -1.0;
  operator+=(result, left);

  return result;
}

template <typename T1>
Matrix<T1> operator-(const double &left, Matrix<T1> &right) {
  auto result = Matrix(right) * -1.0;
  operator+=(result, left);

  return result;
}

template <typename T1, typename T2>
Matrix<T1> &operator*=(Matrix<T1> &left, const T2 &right) {
  auto new_mult = T1(right);
  return left *= new_mult;
}

template <typename T1, typename T2>
Matrix<T1> operator*(Matrix<T1> &left, const T2 &right) {
  auto new_mult = T1(right);
  return left * new_mult;
}

template <typename T1, typename T2>
Matrix<T2> operator*(const T1 &left, Matrix<T2> &right) {
  auto new_mult = T2(left);
  return right * new_mult;
}

// Matrix multiplication
template <typename T1, typename T2>
inline Matrix<T1> _check_compat(const Matrix<T1> &left,
                                const Matrix<T2> &right) {
  if (left.cols() != right.rows())
    throw std::runtime_error("Shape mismatch for _check_compat");

  return Matrix<T1>(left.rows(), right.cols(), T1(0));
}

template <typename T1, typename T2>
Matrix<T1> naive_multiply(Matrix<T1> &left, Matrix<T2> &right) {
  auto C = _check_compat(left, right);

  for (uint64_t i = 0; i < left.rows(); i++)
    for (uint64_t j = 0; j < right.cols(); j++)
      for (uint64_t k = 0; k < left.cols(); k++)
        C(i, j) += left(i, k) * right(k, j);

  return C;
}

template <typename T1, typename T2>
Matrix<T1> operator*(Matrix<T1> &left, Matrix<T2> &right) {
  return naive_multiply(left, right);
}

Matrix<float> operator*(Matrix<float> &left, Matrix<float> &right) {
  Matrix<float> C = _check_compat(left, right);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, left.rows(),
              right.cols(), left.cols(), 1.0, left.data(), left.rows(),
              right.data(), right.rows(), 1.0, C.data(), left.rows());

  return C;
}

Matrix<double> operator*(Matrix<double> &left, Matrix<double> &right) {
  Matrix<double> C = _check_compat(left, right);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, left.rows(),
              right.cols(), left.cols(), 1.0, left.data(), left.rows(),
              right.data(), right.rows(), 1.0, C.data(), left.rows());

  return C;
}

} // namespace cppmatrix

#endif // MATRIX_H
