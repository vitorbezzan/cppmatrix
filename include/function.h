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
 * - Automatic derivative type deduction via FunctionTraits
 */

#ifndef FUNCTION_H
#define FUNCTION_H

#include "matrix.h"
#include "ndarray.h"
#include "vector.h"
#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>

namespace cppmatrix {
    template<typename P, typename I, typename O, typename D1, typename D2>
    class BaseFunction {
    public:
        BaseFunction() = default;

        virtual ~BaseFunction() = default;

        virtual O operator()(const I &x) const = 0;

        virtual D1 d1(const I &x) const = 0;

        virtual D2 d2(const I &x) const = 0;

        /**
         * @brief Wraps this object in a std::function.
         * @warning The result captures `this`. Do not call it after this object is destroyed.
         *          Use copy_callable() when the std::function must outlive a short-lived functor.
         */
        [[nodiscard]] std::function<O(const I &)> get_function() const {
            return [this](const I &x) { return (*this)(x); };
        }
    };

    template<typename T>
        requires std::is_floating_point_v<T>
    using RealFunction = BaseFunction<T, T, T, T, T>;

    template<class F>
    concept IsRealFunction = std::derived_from<F, RealFunction<float>> ||
                             std::derived_from<F, RealFunction<double>>;

    template<typename T>
        requires std::is_floating_point_v<T>
    using ScalarField =
        BaseFunction<T, ColumnVector<T>, T, ColumnVector<T>, Matrix<T> >;

    template<class F>
    concept IsScalarField = std::derived_from<F, ScalarField<float>> ||
                            std::derived_from<F, ScalarField<double>>;

    template<typename T>
        requires std::is_floating_point_v<T>
    using VectorField =
        BaseFunction<T, ColumnVector<T>, ColumnVector<T>, Matrix<T>, NDArray<T> >;

    template<class F>
    concept IsVectorField = std::derived_from<F, VectorField<float>> ||
                            std::derived_from<F, VectorField<double>>;

    template<class F>
    concept IsFloatFunction = std::derived_from<F, RealFunction<float>> ||
                              std::derived_from<F, ScalarField<float>> ||
                              std::derived_from<F, VectorField<float>>;

    template<class F>
    concept IsDoubleFunction = std::derived_from<F, RealFunction<double>> ||
                               std::derived_from<F, ScalarField<double>> ||
                               std::derived_from<F, VectorField<double>>;

    template<class F>
    concept IsFunction = IsRealFunction<F> || IsScalarField<F> || IsVectorField<F>;

    template<typename F>
    struct FunctionTraits;

    template<typename F>
        requires std::derived_from<F, RealFunction<float>>
    struct FunctionTraits<F> {
        using precision_type = float;
        using input_type = float;
        using value_type = float;
        using d1_type = float;
        using d2_type = float;
    };

    template<typename F>
        requires std::derived_from<F, RealFunction<double>> &&
                 (!std::derived_from<F, RealFunction<float>>)
    struct FunctionTraits<F> {
        using precision_type = double;
        using input_type = double;
        using value_type = double;
        using d1_type = double;
        using d2_type = double;
    };

    template<typename F>
        requires std::derived_from<F, ScalarField<float>>
    struct FunctionTraits<F> {
        using precision_type = float;
        using input_type = ColumnVector<float>;
        using value_type = float;
        using d1_type = ColumnVector<float>;
        using d2_type = Matrix<float>;
    };

    template<typename F>
        requires std::derived_from<F, ScalarField<double>> &&
                 (!std::derived_from<F, ScalarField<float>>)
    struct FunctionTraits<F> {
        using precision_type = double;
        using input_type = ColumnVector<double>;
        using value_type = double;
        using d1_type = ColumnVector<double>;
        using d2_type = Matrix<double>;
    };

    template<typename F>
        requires std::derived_from<F, VectorField<float>>
    struct FunctionTraits<F> {
        using precision_type = float;
        using input_type = ColumnVector<float>;
        using value_type = ColumnVector<float>;
        using d1_type = Matrix<float>;
        using d2_type = NDArray<float>;
    };

    template<typename F>
        requires std::derived_from<F, VectorField<double>> &&
                 (!std::derived_from<F, VectorField<float>>)
    struct FunctionTraits<F> {
        using precision_type = double;
        using input_type = ColumnVector<double>;
        using value_type = ColumnVector<double>;
        using d1_type = Matrix<double>;
        using d2_type = NDArray<double>;
    };

    /**
     * @brief Owns a copy of f inside the std::function; safe to store after f goes out of scope.
     */
    template<typename F>
        requires IsRealFunction<std::remove_cvref_t<F>>
    std::function<typename FunctionTraits<std::remove_cvref_t<F>>::value_type(
        const typename FunctionTraits<std::remove_cvref_t<F>>::input_type &)>
    copy_callable(F &&f) {
        using G = std::remove_cvref_t<F>;
        using Traits = FunctionTraits<G>;
        return [owned = G(std::forward<F>(f))](const typename Traits::input_type &x)
                   -> typename Traits::value_type { return owned(x); };
    }

    template<typename F>
        requires IsFunction<std::remove_cvref_t<F>>
    using PrecisionT = typename FunctionTraits<std::remove_cvref_t<F>>::precision_type;

    template<typename F>
        requires IsFunction<std::remove_cvref_t<F>>
    using InputT = typename FunctionTraits<std::remove_cvref_t<F>>::input_type;

    template<typename F>
        requires IsFunction<std::remove_cvref_t<F>>
    using ValueT = typename FunctionTraits<std::remove_cvref_t<F>>::value_type;

    template<typename F>
        requires IsFunction<std::remove_cvref_t<F>>
    using D1ValueT = typename FunctionTraits<std::remove_cvref_t<F>>::d1_type;

    template<typename F>
        requires IsFunction<std::remove_cvref_t<F>>
    using D2ValueT = typename FunctionTraits<std::remove_cvref_t<F>>::d2_type;
}

#endif
