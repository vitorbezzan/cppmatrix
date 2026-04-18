/**
 * @file integration.h
 * @brief Numerical quadrature for scalar functions (RealFunction).
 *
 * Mirrors the style of newton.h: algorithms are class templates constrained with
 * IsRealFunction, hold a copy of the functor like Newton/Polyak hold `F _f`, and expose
 * `run()` with no std::function parameter. Use RealFunction<float/double> types from
 * function.h (or subclasses) as F.
 *
 * Provides:
 * - Base1DIntegrator<P> — interval [lb, ub] and subinterval count n()
 * - Riemann1DIntegrator<F> — left-endpoint (rectangle) rule
 * - Trapezoidal1DIntegrator<F> — composite trapezoidal rule
 * - Simpson1DIntegrator<F> — composite Simpson rule (one panel per subinterval)
 */

#ifndef INTEGRATION_H
#define INTEGRATION_H

#include "function.h"
#include <cstdint>
#ifdef CPPMATRIX_USE_OPENMP
#include <omp.h>
#endif

namespace cppmatrix {
    /**
     * @brief Common state for 1D quadrature on [lb, ub].
     * @tparam P Scalar type for bounds and partial sums (matches PrecisionT<F> for RealFunction).
     */
    template<typename P>
    class Base1DIntegrator {
    public:
        Base1DIntegrator(const P &lb, const P &ub, const uint64_t& n) : _lb(lb), _ub(ub), _n(n) {
        }

        virtual ~Base1DIntegrator() = default;

        [[nodiscard]] uint64_t n() const { return _n; }
        [[nodiscard]] P lb() const { return _lb; }
        [[nodiscard]] P ub() const { return _ub; }

    protected:
        P _lb;
        P _ub;
        uint64_t _n;
    };

    /**
     * @brief Left-endpoint Riemann (rectangle) sum over n equal subintervals.
     * @tparam F Type satisfying IsRealFunction (e.g. RealFunction<float> subclass).
     */
    template<IsRealFunction F>
    class Riemann1DIntegrator final : public Base1DIntegrator<PrecisionT<F>> {
    public:
        Riemann1DIntegrator(const F &f, const PrecisionT<F>& lb, const PrecisionT<F>& ub, const uint64_t& n)
            : Base1DIntegrator<PrecisionT<F>>(lb, ub, n), _f(f) {
        }

        /**
         * @return Approximate integral of _f over [lb, ub].
         */
        PrecisionT<F> run() {
            using P = PrecisionT<F>;
            P result = P(0);
            const P step = (this->ub() - this->lb()) / static_cast<P>(this->n());

#ifdef CPPMATRIX_USE_OPENMP
            #pragma omp parallel for reduction(+:result)
#endif
            for (uint64_t i = 0; i < this->n(); i++)
                result += _f(this->lb() + static_cast<P>(i) * step);

            return result * step;
        }

    private:
        F _f;
    };

    /**
     * @brief Composite trapezoidal rule with n subintervals.
     * @tparam F Type satisfying IsRealFunction.
     */
    template<IsRealFunction F>
    class Trapezoidal1DIntegrator final : public Base1DIntegrator<PrecisionT<F>> {
    public:
        Trapezoidal1DIntegrator(const F &f, const PrecisionT<F>& lb, const PrecisionT<F>& ub, const uint64_t& n)
            : Base1DIntegrator<PrecisionT<F>>(lb, ub, n), _f(f) {
        }

        PrecisionT<F> run() {
            using P = PrecisionT<F>;
            P result = P(0);
            const P step = (this->ub() - this->lb()) / static_cast<P>(this->n());

#ifdef CPPMATRIX_USE_OPENMP
            #pragma omp parallel for reduction(+:result)
#endif
            for (uint64_t i = 0; i < this->n(); i++) {
                const P a_n = this->lb() + step * static_cast<P>(i);
                result += _f(a_n) + _f(a_n + step);
            }

            return result * (step / P(2));
        }

    private:
        F _f;
    };

    /**
     * @brief Composite Simpson rule (one Simpson panel per subinterval).
     * @tparam F Type satisfying IsRealFunction.
     */
    template<IsRealFunction F>
    class Simpson1DIntegrator final : public Base1DIntegrator<PrecisionT<F>> {
    public:
        Simpson1DIntegrator(const F &f, const PrecisionT<F>& lb, const PrecisionT<F>& ub, const uint64_t& n)
            : Base1DIntegrator<PrecisionT<F>>(lb, ub, n), _f(f) {
        }

        PrecisionT<F> run() {
            using P = PrecisionT<F>;
            P result = P(0);
            const P step = (this->ub() - this->lb()) / static_cast<P>(this->n());
            const P half = step / P(2);

#ifdef CPPMATRIX_USE_OPENMP
            #pragma omp parallel for reduction(+:result)
#endif
            for (uint64_t i = 0; i < this->n(); i++) {
                const P a_n = this->lb() + step * static_cast<P>(i);
                result += _f(a_n) + P(4) * _f(a_n + half) + _f(a_n + step);
            }

            return result * (step / P(6));
        }

    private:
        F _f;
    };
}

#endif
