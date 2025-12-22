/**
 * @file fillers.h
 * @brief Provides initialization functions and classes for matrices and vectors.
 * 
 * This module provides:
 * - Basic filler functions (zeros, ones, identity)
 * - Random number generation fillers with configurable distributions
 * - Normal distribution fillers with configurable mean and standard deviation
 * - Base classes for creating custom fillers for both matrices and vectors
 */

#ifndef FILLERS_H
#define FILLERS_H

#include <functional>
#include <random>

namespace cppmatrix {
    template<typename T>
    T identity(const uint64_t &i, const uint64_t &j) {
        if (i == j)
            return T(1);

        return T(0);
    }

    template<typename T>
    T zeros(const uint64_t &i, const uint64_t &j) {
        return T(0);
    }

    template<typename T>
    T ones(const uint64_t &i, const uint64_t &j) {
        return T(1);
    }

    template<typename T>
    class BaseFill {
    public:
        virtual ~BaseFill() = default;

        virtual std::function<T(const uint64_t &, const uint64_t &)> filler() {
            return [](const uint64_t &i, const uint64_t &j) { return T(0); };
        }
    };

    template<typename T>
    class VBaseFill {
    public:
        virtual ~VBaseFill() = default;

        virtual std::function<T(const uint64_t &)> filler() {
            return [](const uint64_t &i) { return T(0); };
        }
    };

    template<typename T>
    class BaseRandomFill : BaseFill<T> {
    public:
        explicit BaseRandomFill(const uint64_t &seed) { this->_rng = std::mt19937_64(seed); }

        virtual std::function<T(const uint64_t &, const uint64_t &)> filler() {
            return [this](const uint64_t &i, const uint64_t &j) {
                return std::uniform_real_distribution<T>(0.0, 1.0)(this->_rng);
            };
        }

        std::mt19937_64 _rng;
    };

    template<typename T>
    class NormalFill : BaseRandomFill<T> {
    public:
        NormalFill(const uint64_t &seed, const T &mean, const T &std)
            : BaseRandomFill<T>(seed) {
            this->_mean = mean;
            this->_std = std;
        }

        std::function<T(const uint64_t &, const uint64_t &)> filler() final {
            return [this](const uint64_t &i, const uint64_t &j) {
                return std::normal_distribution<T>(this->_mean, this->_std)(this->_rng);
            };
        }

        T _mean;
        T _std;
    };

    template<typename T>
    class VBaseRandomFill : VBaseFill<T> {
    public:
        explicit VBaseRandomFill(const uint64_t &seed) { this->_rng = std::mt19937_64(seed); }

        virtual std::function<T(const uint64_t &)> filler() {
            return [this](const uint64_t &i) {
                return std::uniform_real_distribution<T>(0.0, 1.0)(this->_rng);
            };
        }

        std::mt19937_64 _rng;
    };

    template<typename T>
    class VNormalFill : VBaseRandomFill<T> {
    public:
        VNormalFill(const uint64_t &seed, const T &mean, const T &std)
            : VBaseRandomFill<T>(seed) {
            this->_mean = mean;
            this->_std = std;
        }

        std::function<T(const uint64_t &)> filler() final {
            return [this](const uint64_t &i) {
                return std::normal_distribution<T>(this->_mean, this->_std)(this->_rng);
            };
        }

        T _mean;
        T _std;
    };
}

#endif
