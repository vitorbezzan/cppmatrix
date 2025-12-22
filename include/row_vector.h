/**
 * @file row_vector.h
 * @brief Provides the RowVector class for horizontal vector operations.
 * 
 * This module implements a templated RowVector class derived from Matrix that provides:
 * - Specialized operations for row vectors (1xN matrices)
 * - BLAS-optimized vector operations for float and double types
 * - Dot product and vector norm calculations
 * - Efficient vector-matrix multiplication
 * - Support for scalar operations and vector arithmetic
 * - Template-based mixed-type operations
 * - Various constructors including function-based initialization
 * - Optimized memory layout for performance
 * - Exception handling for size mismatches
 */

#ifndef ROW_VECTOR_H
#define ROW_VECTOR_H

#include "matrix.h"
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <print>

namespace cppmatrix {
    template<typename T>
    class RowVector final : public Matrix<T> {
    public:
        template<typename U>
        friend class RowVector;

        RowVector() : Matrix<T>() {
        };

        explicit RowVector(const uint64_t &N) : Matrix<T>(1, N) { this->_N = N; }

        RowVector(const uint64_t &N, const T &value) : Matrix<T>(1, N, value) {
            this->_N = N;
        }

        template<uint64_t ndim>
        explicit RowVector(const T (&values)[ndim]) : Matrix<T>(1, ndim) {
            this->_N = ndim;
            std::copy(values, values + ndim, this->data());
        }

        RowVector(const uint64_t &N, T (*f)(const uint64_t &)) : Matrix<T>(1, N) {
            this->_N = N;
            for (uint64_t n = 0; n < this->_N; n++)
                this->operator()(n) = f(n);
        }

        RowVector(const uint64_t &N, const std::function<T(const uint64_t &)> &f)
            : Matrix<T>(1, N) {
            this->_N = N;
            for (uint64_t n = 0; n < this->_N; n++)
                this->operator()(n) = f(n);
        }

        explicit RowVector(const Matrix<T> &M) : Matrix<T>(M) {
            if (M.rows() > 1)
                throw std::runtime_error("Size mismatch for RowVector constructor: matrix must have only one row.");
            this->_N = M.cols();
        }

        RowVector(const RowVector<T> &v) : Matrix<T>(1, v.N()) {
            this->_N = v.N();
            std::copy(v.data(), v.data() + v.N(), this->data());
        }

        T &operator()(uint64_t n) { return Matrix<T>::operator()(0, n); }
        const T &operator()(uint64_t n) const { return Matrix<T>::operator()(0, n); }

        T &operator[](uint64_t n) { return this->operator()(n); }
        const T &operator[](uint64_t n) const { return this->operator()(n); }

        template<typename T2>
        RowVector<T> &operator*=(const T2 &right) {
            std::transform(
                this->data(), this->data() + this->N(), this->data(),
                std::bind(std::multiplies<T>(), std::placeholders::_1, T(right)));
            return *this;
        }

        template<typename T2>
        RowVector<T> operator*(const T2 &right) const {
            RowVector<T> result(*this);
            result *= right;
            return result;
        }

        template<typename T2>
            requires std::is_arithmetic_v<T2>
        RowVector<T> &operator/=(const T2 &right) {
            if (right == T2(0))
                throw std::runtime_error("Division by zero.");
            std::transform(
                this->data(), this->data() + this->N(), this->data(),
                std::bind(std::divides<T>(), std::placeholders::_1, T(right)));
            return *this;
        }

        template<typename T2>
            requires std::is_arithmetic_v<T2>
        RowVector<T> operator/(const T2 &right) const {
            RowVector<T> result(*this);
            result /= right;
            return result;
        }

        template<typename T2>
        bool operator==(const RowVector<T2> &right) const {
            if (this->N() != right.N())
                return false;
            for (uint64_t i = 0; i < this->N(); i++)
                if (std::abs(this->operator()(i) - T(right(i))) > std::numeric_limits<T>::epsilon())
                    return false;
            return true;
        }

        template<typename T2>
        bool operator!=(const RowVector<T2> &right) const {
            return !(*this == right);
        }

        [[nodiscard]] uint64_t N() const override { return this->_N; }

    private:
        uint64_t _N = 0;
    };

    template<typename T1, typename T2>
    RowVector<T1> &operator+=(RowVector<T1> &left,
                              const RowVector<T2> &right) {
        if (left.N() != right.N()) {
            throw std::runtime_error("Size mismatch for operator+=(): vector sizes must match.");
        }

        std::transform(left.data(), left.data() + left.N(), right.data(), left.data(),
                       std::plus<T1>());

        return left;
    }

    template<typename T1, typename T2>
    RowVector<T1> operator+(const RowVector<T1> &left, const RowVector<T2> &right) {
        auto result = RowVector(left);
        operator+=(result, right);

        return result;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T2>
    RowVector<T1> &operator+=(RowVector<T1> &left, const T2 &right) {
        std::transform(left.data(), left.data() + left.N(), left.data(),
                       [right](T2 element) { return element + right; });
        return left;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T2>
    RowVector<T1> operator+(const RowVector<T1> &left, const T2 &right) {
        auto result = RowVector(left);
        operator+=(result, right);

        return result;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1>
    RowVector<T2> operator+(const T1 &left, const RowVector<T2> &right) {
        auto result = RowVector(right);
        operator+=(result, left);

        return result;
    }

    template<typename T1, typename T2>
    RowVector<T1> &operator-=(RowVector<T1> &left, const RowVector<T2> &right) {
        if (left.N() != right.N()) {
            throw std::runtime_error("Size mismatch for operator-=(): vector sizes must match.");
        }

        std::transform(left.data(), left.data() + left.N(), right.data(), left.data(),
                       std::minus<T1>());

        return left;
    }

    template<typename T1, typename T2>
    RowVector<T1> operator-(const RowVector<T1> &left, const RowVector<T2> &right) {
        auto result = RowVector(left);
        operator-=(result, right);

        return result;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T2>
    RowVector<T1> &operator-=(RowVector<T1> &left, const T2 &right) {
        std::transform(left.data(), left.data() + left.N(), left.data(),
                       [right](T1 element) { return element - T1(right); });
        return left;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T2>
    RowVector<T1> operator-(const RowVector<T1> &left, const T2 &right) {
        auto result = RowVector(left);
        operator-=(result, right);

        return result;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1>
    RowVector<T2> operator-(const T1 &left, const RowVector<T2> &right) {
        auto result = RowVector(right) * T2(-1.0);
        operator+=(result, T2(left));

        return result;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1>
    RowVector<T2> operator*(const T1 &left, const RowVector<T2> &right) {
        return right * T2(left);
    }

    inline RowVector<float> operator*(const RowVector<float> &left, const Matrix<float> &right) {
        if (right.rows() != left.N())
            throw std::runtime_error("Size mismatch for operator *().");

        auto result = RowVector<float>(right.cols());
        cblas_sgemv(CblasRowMajor, CblasTrans, right.rows(), right.cols(), 1.0,
                    right.data(), right.cols(), left.data(), 1, 0.0, result.data(),
                    1);

        return result;
    }

    inline RowVector<double> operator*(const RowVector<double> &left, const Matrix<double> &right) {
        if (right.rows() != left.N())
            throw std::runtime_error("Size mismatch for operator *().");

        auto result = RowVector<double>(right.cols());
        cblas_dgemv(CblasRowMajor, CblasTrans, right.rows(), right.cols(), 1.0,
                    right.data(), right.cols(), left.data(), 1, 0.0, result.data(),
                    1);

        return result;
    }

    inline float dot(const RowVector<float> &left, const RowVector<float> &right) {
        if (left.N() == right.N())
            return cblas_sdot(left.N(), left.data(), 1, right.data(), 1);
        else
            throw std::runtime_error("Shape mismatch for dot().");
    }

    inline double dot(const RowVector<double> &left, const RowVector<double> &right) {
        if (left.N() == right.N())
            return cblas_ddot(left.N(), left.data(), 1, right.data(), 1);
        else
            throw std::runtime_error("Shape mismatch for dot().");
    }

    template<typename T>
    T norm(const RowVector<T> &v) { return std::sqrt(dot(v, v)); }

    template<typename T>
    T squared_norm(const RowVector<T> &v) { return dot(v, v); }

    template<typename T>
    T inverse_squared_norm(const RowVector<T> &v) { return 1.0 / dot(v, v); }

    template<typename T>
    void print(const RowVector<T> &v, const int &precision = 5) {
        for (uint64_t j = 0; j < v.N(); j++) {
            std::print("{:.{}} \t", v(j), precision);
        }
        std::print("\n");
    }
}

#endif