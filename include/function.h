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

namespace cppmatrix {
    // Abstract class defining a function.
    template<typename P, typename I, typename O, typename D1, typename D2>
    class BaseFunction {
    public:
        // Constructor and destructor
        BaseFunction() = default;

        virtual ~BaseFunction() = default;

        // Function value, 1st and 2nd derivative
        virtual O operator()(I &x) = 0;

        virtual D1 d1(I &x) = 0;

        virtual D2 d2(I &x) = 0;

        // Tricking the compiler into giving the types for the function.
        static P PrecisionT;
        static I InputT;
        static O ValueT;
        static D1 D1ValueT;
        static D2 D2ValueT;
    };

    // Function R->R
    template<typename T>
        requires std::is_floating_point_v<T>
    using RealFunction = BaseFunction<T, T, T, T, T>;

    template<class F>
    concept IsRealFunction = std::is_base_of_v<RealFunction<float>, F> ||
                             std::is_base_of_v<RealFunction<double>, F>;

    // Scalar fields (R^n->R)
    template<typename T>
        requires std::is_floating_point_v<T>
    using ScalarField =
    BaseFunction<T, Vector<T>, T, Vector<T>, Matrix<T> >;

    template<class F>
    concept IsScalarField = std::is_base_of_v<ScalarField<float>, F> ||
                            std::is_base_of_v<ScalarField<double>, F>;

    // Vector fields (R^n->R^n)
    template<typename T>
        requires std::is_floating_point_v<T>
    using VectorField =
    BaseFunction<T, Vector<T>, Vector<T>, Matrix<T>, NDArray<T> >;

    template<class F>
    concept IsVectorField = std::is_base_of_v<VectorField<float>, F> ||
                            std::is_base_of_v<VectorField<double>, F>;

    // General definitions of functions
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

    // Extracts precision types for functions
    template<IsFunction F>
    using PrecisionT = decltype(F::PrecisionT);

    // Extracts input type for functions
    template<IsFunction F>
    using InputT = decltype(F::InputT);

    // Extracts values type for functions
    template<IsFunction F>
    using ValueT = decltype(F::ValueT);

    // Extracts 1st derivative type for functions
    template<IsFunction F>
    using D1ValueT = decltype(F::D1ValueT);

    // Extracts 2nd derivative type for functions
    template<IsFunction F>
    using D2ValueT = decltype(F::D2ValueT);
} // namespace cppmatrix

#endif // FUNCTION_H
