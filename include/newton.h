/**
 * @file newton.h
 * @brief Implements Newton's method for root finding.
 *
 * This module provides:
 * - Base class for root finding algorithms
 * - Implementation of Newton's method for finding zeros of functions
 * - Support for real functions (R->R)
 * - Configurable precision and maximum iterations
 * - Automatic derivative usage from function classes
 */

#ifndef NEWTON_H
#define NEWTON_H

#include "function.h"
#include <vector>
#include <limits>
#include <cmath>
#include <stdexcept>
#ifdef CPPMATRIX_USE_OPENMP
#include <omp.h>
#endif

namespace cppmatrix {
    template<typename P, typename I>
    class BaseRootFind {
    public:
        BaseRootFind(const P &precision, uint64_t n)
            : _precision(precision)
            , _n(n) {}

        virtual ~BaseRootFind() = default;

        [[nodiscard]] P precision() const { return _precision; }
        [[nodiscard]] uint64_t n() const { return this->_n; }

        virtual I run(const I &x0) = 0;

    private:
        P _precision;
        uint64_t _n;
    };

    template<IsRealFunction F>
    class Newton final : public BaseRootFind<PrecisionT<F>, InputT<F>> {
    public:
        using Precision = PrecisionT<F>;
        using Input = InputT<F>;
        using Base = BaseRootFind<Precision, Input>;

        Newton(const F &f, const Precision &precision, uint64_t n)
            : Base(precision, n)
            , _f(f) {}

        Input run(const Input &x0) override {
            Input x = x0;
            Input x_new;

            for (uint64_t N = 0; N < this->n(); N++) {
                x_new = x - this->_f(x) / this->_f.d1(x);

                if (std::fabs(x_new - x) <= this->precision())
                    break;

                x = x_new;
            }

            return x;
        }

    private:
        F _f;
    };

    template<IsScalarField F>
    class Polyak final : public BaseRootFind<PrecisionT<F>, InputT<F>> {
    public:
        using Precision = PrecisionT<F>;
        using Input = InputT<F>;
        using Base = BaseRootFind<Precision, Input>;

        Polyak(const F &f, const Precision &precision, uint64_t n)
            : Base(precision, n)
            , _f(f) {}

        Input run(const Input &x0) override {
            Input x(x0);
            Input x_new(x0);

            for (uint64_t N = 0; N < this->n(); N++) {
                auto d1 = this->_f.d1(x);
                auto norm_d1 = inverse_squared_norm(this->_f.d1(x)) * d1;

                x_new = x - this->_f(x) * norm_d1;

                if (norm(x_new - x) <= this->precision())
                    break;

                x = x_new;
            }

            return x;
        }

    private:
        F _f;
    };

    /**
     * @brief Parallel multi-start Newton's method
     *
     * Runs Newton's method from multiple starting points in parallel.
     * Useful for finding multiple roots or improving convergence reliability.
     * Returns the best converged result based on function value at convergence.
     */
    template<IsRealFunction F>
    class ParallelNewton final : public BaseRootFind<PrecisionT<F>, InputT<F>> {
    public:
        using Precision = PrecisionT<F>;
        using Input = InputT<F>;
        using Base = BaseRootFind<Precision, Input>;

        ParallelNewton(const F &f, const Precision &precision, uint64_t n)
            : Base(precision, n)
            , _f(f) {}

        /**
         * @brief Run Newton's method from multiple starting points in parallel
         *
         * @param starting_points Vector of initial guesses
         * @return Best converged root (closest to zero)
         */
        Input run_multi(const std::vector<Input>& starting_points) {
            if (starting_points.empty()) {
                throw std::runtime_error("ParallelNewton: no starting points provided");
            }

            const std::size_t n_starts = starting_points.size();
            std::vector<Input> results(n_starts);
            std::vector<Precision> residuals(n_starts);

#ifdef CPPMATRIX_USE_OPENMP
            #pragma omp parallel for
#endif
            for (std::size_t i = 0; i < n_starts; ++i) {
                results[i] = this->run(starting_points[i]);
                residuals[i] = std::abs(this->_f(results[i]));
            }

            std::size_t best_idx = 0;
            Precision best_residual = residuals[0];

            for (std::size_t i = 1; i < n_starts; ++i) {
                if (residuals[i] < best_residual) {
                    best_residual = residuals[i];
                    best_idx = i;
                }
            }

            return results[best_idx];
        }

        /**
         * @brief Run Newton's method from multiple starting points and return all results
         *
         * @param starting_points Vector of initial guesses
         * @return Vector of all converged roots
         */
        std::vector<Input> run_multi_all(const std::vector<Input>& starting_points) {
            if (starting_points.empty()) {
                throw std::runtime_error("ParallelNewton: no starting points provided");
            }

            const std::size_t n_starts = starting_points.size();
            std::vector<Input> results(n_starts);

#ifdef CPPMATRIX_USE_OPENMP
            #pragma omp parallel for
#endif
            for (std::size_t i = 0; i < n_starts; ++i) {
                results[i] = this->run(starting_points[i]);
            }

            return results;
        }

        /**
         * @brief Single starting point interface (for compatibility)
         */
        Input run(const Input &x0) override {
            Input x = x0;
            Input x_new;

            for (uint64_t N = 0; N < this->n(); N++) {
                x_new = x - this->_f(x) / this->_f.d1(x);

                if (std::fabs(x_new - x) <= this->precision())
                    break;

                x = x_new;
            }

            return x;
        }

    private:
        F _f;
    };

    /**
     * @brief Parallel multi-start Polyak's method for scalar fields
     */
    template<IsScalarField F>
    class ParallelPolyak final : public BaseRootFind<PrecisionT<F>, InputT<F>> {
    public:
        using Precision = PrecisionT<F>;
        using Input = InputT<F>;
        using Base = BaseRootFind<Precision, Input>;

        ParallelPolyak(const F &f, const Precision &precision, uint64_t n)
            : Base(precision, n)
            , _f(f) {}

        /**
         * @brief Run Polyak's method from multiple starting points in parallel
         *
         * @param starting_points Vector of initial guesses
         * @return Best converged root (closest to zero)
         */
        Input run_multi(const std::vector<Input>& starting_points) {
            if (starting_points.empty()) {
                throw std::runtime_error("ParallelPolyak: no starting points provided");
            }

            const std::size_t n_starts = starting_points.size();
            std::vector<Input> results(n_starts);
            std::vector<Precision> residuals(n_starts);

#ifdef CPPMATRIX_USE_OPENMP
            #pragma omp parallel for
#endif
            for (std::size_t i = 0; i < n_starts; ++i) {
                results[i] = this->run(starting_points[i]);
                residuals[i] = std::abs(this->_f(results[i]));
            }

            std::size_t best_idx = 0;
            Precision best_residual = residuals[0];

            for (std::size_t i = 1; i < n_starts; ++i) {
                if (residuals[i] < best_residual) {
                    best_residual = residuals[i];
                    best_idx = i;
                }
            }

            return results[best_idx];
        }

        /**
         * @brief Single starting point interface (for compatibility)
         */
        Input run(const Input &x0) override {
            Input x(x0);
            Input x_new(x0);

            for (uint64_t N = 0; N < this->n(); N++) {
                auto d1 = this->_f.d1(x);
                auto norm_d1 = inverse_squared_norm(this->_f.d1(x)) * d1;

                x_new = x - this->_f(x) * norm_d1;

                if (norm(x_new - x) <= this->precision())
                    break;

                x = x_new;
            }

            return x;
        }

    private:
        F _f;
    };
}

#endif