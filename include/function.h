#ifndef FUNCTION_H
#define FUNCTION_H

#include "matrix.h"
#include "ndarray.h"

namespace cppmatrix {

template <typename _Input, typename _FOutput, typename _F1Output,
          typename _F2Output, typename _RootT>
class BaseFunction {

public:
  BaseFunction() = default;

  virtual _FOutput operator()(const _Input &x) { return _FOutput(); }
  virtual _F1Output d1(const _Input &x) { return _F1Output(); }
  virtual _F2Output d2(const _Input &x) { return _F2Output(); }

  virtual ~BaseFunction() = default;

  // Tricking the compiler to get the types- placeholders to extract the types
  // during compilation
  static _RootT __RootType__;
  static _Input __InputType__;
  static _FOutput __FOutputType__;
  static _F1Output __F1OutputType__;
  static _F2Output __F2OutputType__;
};

// Real function of real values
template <typename T> using RealFunction = BaseFunction<T, T, T, T, T>;

// Scalar fields (Real functions of multiple coordinates)
template <typename T>
using ScalarField =
    BaseFunction<ColumnVector<T>, T, ColumnVector<T>, Matrix<T>, T>;

// Vector fields (Vector functions of multiple coordinates)
template <typename T>
using VectorField =
    BaseFunction<ColumnVector<T>, ColumnVector<T>, Matrix<T>, NDArray<T>, T>;

// Extractors for types
template <class F> using InputT = decltype(F::__InputType__);

template <class F> using FOutputT = decltype(F::__FOutputType__);

template <class F> using F1OutputT = decltype(F::__F1OutputType__);

template <class F> using F2OutputT = decltype(F::__F2OutputType__);

template <class F> using RootT = decltype(F::__RootType__);

} // namespace cppmatrix

#endif // FUNCTION_H
