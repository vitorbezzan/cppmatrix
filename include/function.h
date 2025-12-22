/**
 * @file function.h
 * @brief Provides base classes for mathematical function implementations.
 * 
 * This module provides:
 * - Base class for implementing mathematical functions
 * - Support for real functions (R->R)
 * - Support for scalar fields (R^n->R)
 * - Support for vector fields (R^n->R^n)
 * - Type traits and concepts for function type checking
 * - Automatic derivative type deduction
 */

#ifndef FUNCTION_H
#define FUNCTION_H

#include "matrix.h"
#include "ndarray.h"
#include "vector.h"
#include <type_traits>
#include <functional>

namespace cppmatrix {
    template<typename P, typename I, typename O, typename D1, typename D2>
    class BaseFunction {
    public:
        BaseFunction() = default;

        virtual ~BaseFunction() = default;

        virtual O operator()(const I &x) = 0;

        virtual D1 d1(const I &x) = 0;

        virtual D2 d2(const I &x) = 0;

        static P PrecisionT;
        static I InputT;
        static O ValueT;
        static D1 D1ValueT;
        static D2 D2ValueT;

        std::function<O(const I &)> get_function() {
            return std::function < O(const I &) > ([this](const I &x) { return this->operator()(x); });
        }
    };

    template<typename T>
        requires std::is_floating_point_v<T>
    using RealFunction = BaseFunction<T, T, T, T, T>;

    template<class F>
    concept IsRealFunction = std::is_base_of_v<RealFunction<float>, F> ||
                             std::is_base_of_v<RealFunction<double>, F>;

    template<typename T>
        requires std::is_floating_point_v<T>
    using ScalarField =
    BaseFunction<T, ColumnVector<T>, T, ColumnVector<T>, Matrix<T> >;

    template<class F>
    concept IsScalarField = std::is_base_of_v<ScalarField<float>, F> ||
                            std::is_base_of_v<ScalarField<double>, F>;

    template<typename T>
        requires std::is_floating_point_v<T>
    using VectorField =
    BaseFunction<T, ColumnVector<T>, ColumnVector<T>, Matrix<T>, NDArray<T> >;

    template<class F>
    concept IsVectorField = std::is_base_of_v<VectorField<float>, F> ||
                            std::is_base_of_v<VectorField<double>, F>;

    template<class F>
    concept IsFloatFunction = std::is_base_of_v<RealFunction<float>, F> ||
                              std::is_base_of_v<ScalarField<float>, F> ||
                              std::is_base_of_v<VectorField<float>, F>;

    template<class F>
    concept IsDoubleFunction = std::is_base_of_v<RealFunction<double>, F> ||
                               std::is_base_of_v<ScalarField<double>, F> ||
                               std::is_base_of_v<VectorField<double>, F>;

    template<class F>
    concept IsFunction = IsRealFunction<F> || IsScalarField<F> || IsVectorField<F>;

    template<IsFunction F>
    using PrecisionT = decltype(F::PrecisionT);

    template<IsFunction F>
    using InputT = decltype(F::InputT);

    template<IsFunction F>
    using ValueT = decltype(F::ValueT);

    template<IsFunction F>
    using D1ValueT = decltype(F::D1ValueT);

    template<IsFunction F>
    using D2ValueT = decltype(F::D2ValueT);
}

#endif