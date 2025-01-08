#ifndef FUNCTION_H
#define FUNCTION_H

#include "matrix.h"
#include "ndarray.h"
#include "vector.h"
#include <type_traits>

namespace cppmatrix {

template <typename _Precision, typename _Input, typename _FOutput,
          typename _F1Output, typename _F2Output>
class BaseFunction {

public:
  BaseFunction() = default;

  virtual _FOutput operator()(_Input &x) { return _FOutput(); }
  virtual _F1Output d1(_Input &x) { return _F1Output(); }
  virtual _F2Output d2(_Input &x) { return _F2Output(); }

  virtual ~BaseFunction() = default;

  // Tricking compiler to get types for function internals

  static _Precision
      __PrecisionT__; // Precision to use for function. float or double

  static _Input __InputT__;       // Type of input for the function
  static _FOutput __FOutputT__;   // Type of function output
  static _F1Output __F1OutputT__; // Type of derivative output
  static _F2Output __F2OutputT__; // Type of 2nd derivative output
};

// Real function of real values
template <typename T>
  requires std::is_floating_point_v<T>
using RealFunction = BaseFunction<T, T, T, T, T>;

template <class F>
concept _RealFunction = std::is_base_of_v<RealFunction<float>, F> ||
                        std::is_base_of_v<RealFunction<double>, F>;

// Scalar fields (Real functions of multiple coordinates)
template <typename T>
  requires std::is_floating_point_v<T>
using ScalarField =
    BaseFunction<T, ColumnVector<T>, T, ColumnVector<T>, Matrix<T>>;

template <class F>
concept _ScalarField = std::is_base_of_v<ScalarField<float>, F> ||
                       std::is_base_of_v<ScalarField<double>, F>;

// Vector fields (Vector functions of multiple coordinates)
template <typename T>
  requires std::is_floating_point_v<T>
using VectorField =
    BaseFunction<T, ColumnVector<T>, ColumnVector<T>, Matrix<T>, NDArray<T>>;

template <class F>
concept _VectorField = std::is_base_of_v<VectorField<float>, F> ||
                       std::is_base_of_v<VectorField<double>, F>;

// Extractors for types
template <class F>
concept _FloatFunction = std::is_base_of_v<RealFunction<float>, F> ||
                         std::is_base_of_v<ScalarField<float>, F> ||
                         std::is_base_of_v<VectorField<float>, F>;

template <class F>
concept _DoubleFunction = std::is_base_of_v<RealFunction<double>, F> ||
                          std::is_base_of_v<ScalarField<double>, F> ||
                          std::is_base_of_v<VectorField<double>, F>;

template <class F>
concept _Function = _RealFunction<F> || _ScalarField<F> || _VectorField<F>;

template <_Function F> using PrecisionT = decltype(F::__PrecisionT__);
template <_Function F> using InputT = decltype(F::__InputT__);
template <_Function F> using FOutputT = decltype(F::__FOutputT__);
template <_Function F> using F1OutputT = decltype(F::__F1OutputT__);
template <_Function F> using F2OutputT = decltype(F::__F2OutputT__);

} // namespace cppmatrix

#endif // FUNCTION_H
