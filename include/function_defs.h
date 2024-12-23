#ifndef FUNCTION_DEFS_H
#define FUNCTION_DEFS_H

#include "matrix.h"

namespace cppmatrix {

template <typename T>
  requires std::is_floating_point_v<T>
class RtoR_Function {
  // R -> R
  // This type of function needs a real number

public:
  // Function value
  virtual T operator()(const T &x) {}

  // First derivative
  virtual T f1d(const T &x) {}

  // Second derivative
  virtual T f2d(const T &x) {}
};

typedef RtoR_Function<double> Function;

template <typename T>
  requires std::is_floating_point_v<T>
class RntoR_Function {
  // Rn -> R
  // This type of function needs a (n,1) Matrix

public:
  // Function value
  virtual T operator()(Matrix<T> &x) {}

  // First derivative - gradient. Returns (n,1) Matrix
  virtual Matrix<T> f1d(Matrix<T> &x) {}

  // Second derivative - hessian. Returns (n,n) Matrix
  virtual Matrix<T> f2d(Matrix<T> &x) {}
};

typedef RntoR_Function<double> ScalarField;

template <typename T>
  requires std::is_floating_point_v<T>
class RntoRn_Function {
  // Rn -> Rn
  // This type of function needs a (n,1) Matrix

public:
  // Function value
  virtual T operator()(Matrix<T> &x) {}

  // First derivative - jacobian. Returns (n,1) Matrix
  virtual Matrix<T> f1d(Matrix<T> &x) {}
};

typedef RntoRn_Function<double> VectorField;

} // namespace cppmatrix

#endif // FUNCTION_DEFS_H
