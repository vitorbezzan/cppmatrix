/**
 * @file integration.h
 * @brief Provides numerical integration algorithms for mathematical functions.
 * 
 * This module implements various numerical integration methods and provides:
 * - Base class for 1D integration algorithms
 * - Trapezoidal rule integration
 * - Simpson's rule integration
 * - Support for arbitrary precision types
 * - Configurable integration intervals and steps
 */

#ifndef INTEGRATION_H
#define INTEGRATION_H

#include <functional>

namespace cppmatrix {
    // Defines a base 1D integrator using Rectangle rule with defined precision and multiple outputs.
    template<typename P = float, typename I = float>
    class Base1DIntegrator {
    public:
        Base1DIntegrator(const P &lb, const P &ub, const uint64_t &n) {
            this->_lb = lb;
            this->_ub = ub;
            this->_n = n;
        }

        virtual ~Base1DIntegrator() = default;

        [[nodiscard]] uint64_t n() const { return this->_n; }
        [[nodiscard]] P lb() const { return this->_lb; }
        [[nodiscard]] P ub() const { return this->_ub; }

        virtual P run(const std::function<P(const I &)> &f) {
            P result = 0;
            P step = (this->_ub - this->_lb) / this->_n;

            for (uint64_t i = 0; i < this->_n; i++) {
                result += f(this->_lb + i * step);
            }

            return result * step;
        }

    private:
        P _lb;
        P _ub;
        uint64_t _n;
    };

    // Defines a 1D integrator using the trapezoidal rule.
    template<typename P = float, typename I = float>
    class Trapezoidal1DIntegrator final : public Base1DIntegrator<P, I> {
    public:
        Trapezoidal1DIntegrator(const P &lb, const P &ub, const uint64_t &n) : Base1DIntegrator<P, I>(lb, ub, n) {
        }

        P run(const std::function<P(const I &)> &f) override {
            P result = 0;
            P step = (this->ub() - this->lb()) / this->n();

            for (uint64_t i = 0; i < this->n(); i++) {
                auto a_n = (this->lb() + step * i);
                result += f(a_n) + f(a_n + step);
            }

            return result * (step / 2);
        }
    };

    // Defines a 1D integrator using Simpson's rule.
    template<typename P = float, typename I = float>
    class Simpson1DIntegrator final : public Base1DIntegrator<P, I> {
    public:
        Simpson1DIntegrator(const P &lb, const P &ub, const uint64_t &n) : Base1DIntegrator<P, I>(lb, ub, n) {
        }

        P run(const std::function<P(const I &)> &f) override {
            P result = 0;
            P step = (this->ub() - this->lb()) / this->n();

            for (uint64_t i = 0; i < this->n(); i++) {
                auto a_n = (this->lb() + step * i);
                result += f(a_n) + 4 * f(a_n + step / 2) + f(a_n + step);
            }

            return result * (step / 6);
        }
    };
}

#endif
