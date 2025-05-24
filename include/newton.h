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

namespace cppmatrix {
    // Defines a base rootfinding with defined precision and multiple outputs
    template<typename P, typename I, typename O>
    class BaseRootFind {
    public:
        BaseRootFind(const P &precision, const uint64_t &n) {
            this->_precision = precision;
            this->_n = n;
        }

        virtual ~BaseRootFind() = default;

        P precision() { return this->_precision; }
        [[nodiscard]] uint64_t n() const { return this->_n; }

        virtual O run(const I &x0) = 0;

    private:
        P _precision;
        uint64_t _n;
    };

    // Newton rootfinding algorithm for real functions
    template<IsRealFunction F>
    class Newton final : public BaseRootFind<PrecisionT<F>, InputT<F>, ValueT<F> > {
    public:
        Newton(const F &f, const PrecisionT<F> &precision, const uint64_t &n)
            : BaseRootFind<PrecisionT<F>, InputT<F>, ValueT<F> >(precision, n) {
            this->_f = f;
        }

        ValueT<F> run(const PrecisionT<F> &x0) override {
            InputT<F> x = x0;
            InputT<F> x_new;

            for (uint64_t N = 0; N < this->n(); N++) {
                x_new = x - this->_f(x) / this->_f.d1(x);

                if (fabs(x_new - x) <= this->precision())
                    break;

                x = x_new;
            }

            return x;
        }

    private:
        F _f;
    };
} // namespace cppmatrix

#endif // NEWTON_H
