/**
 * @file column_vector.h
 * @brief Provides the ColumnVector class for vertical vector operations.
 * 
 * This module implements a templated ColumnVector class derived from Matrix that provides:
 * - Specialized operations for column vectors (Nx1 matrices)
 * - BLAS-optimized vector operations for float and double types
 * - Dot product and vector norm calculations
 * - Efficient matrix-vector multiplication
 * - Support for scalar operations and vector arithmetic
 */

#ifndef COLUMN_VECTOR_H
#define COLUMN_VECTOR_H

#include "matrix.h"
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace cppmatrix {
    template<typename T>
    class ColumnVector final : public Matrix<T> {
    public:
        // Constructors
        ColumnVector() : Matrix<T>() {
        };

        explicit ColumnVector(const uint64_t &N) : Matrix<T>(N, 1) { this->_N = N; }

        ColumnVector(const uint64_t &N, const T &value) : Matrix<T>(N, 1, value) {
            this->_N = N;
        }

        template<uint64_t ndim>
        explicit ColumnVector(const T (&values)[ndim]) : Matrix<T>(ndim, 1) {
            this->_N = ndim;
            std::copy(values, values + ndim, this->data());
        }

        // Constructor for fill function
        ColumnVector(const uint64_t &N, T (*f)(const uint64_t &)) : Matrix<T>(N, 1) {
            this->_N = N;
            for (uint64_t n = 0; n < this->_N; n++)
                this->operator()(n) = f(n);
        }

        // Constructor for fill function
        ColumnVector(const uint64_t &N, const std::function<T(const uint64_t &)> &f)
            : Matrix<T>(N, 1) {
            this->_N = N;
            for (uint64_t n = 0; n < this->_N; n++)
                this->operator()(n) = f(n);
        }

        explicit ColumnVector(const Matrix<T> &M) : Matrix<T>(M) {
            if (M.cols() > 1)
                throw std::runtime_error("Size mismatch for operator-=().");

            this->_N = M.rows();
        }

        ColumnVector(const ColumnVector<T> &v) : Matrix<T>(v.N(), 1) {
            this->_N = v.N();
        }

        // Access operator
        T &operator()(uint64_t n) { return Matrix<T>::operator()(n, 0); }

        // Operators: multiplication from the right
        template<typename T2>
        ColumnVector<T> &operator*=(const T2 &right) {
            std::transform(
                this->data(), this->data() + this->N(), this->data(),
                std::bind(std::multiplies<T>(), std::placeholders::_1, T(right)));
            return *this;
        }

        template<typename T2>
        ColumnVector<T> operator*(const T2 &right) {
            return ColumnVector<T>(*this) *= right;
        }

        // Public API
        [[nodiscard]] uint64_t N() const override { return this->_N; }

    private:
        uint64_t _N = 0;
    };

    // Operators: plus (for different types)
    template<typename T1, typename T2>
    ColumnVector<T1> &operator+=(const ColumnVector<T1> &left,
                                 const ColumnVector<T2> &right) {
        if (left.N() != right.N()) {
            throw std::runtime_error("Size mismatch for operator+=().");
        }

        std::transform(left.data(), left.data() + left.N(), right.data(), left.data(),
                       std::plus<T1>());

        return left;
    }

    template<typename T1, typename T2>
    ColumnVector<T1> operator+(const ColumnVector<T1> &left,
                               const ColumnVector<T2> &right) {
        auto result = ColumnVector(left);
        operator+=(result, right);

        return result;
    }

    // Operators: plus (for scalars)
    template<typename T1, typename T2>
    ColumnVector<T1> &operator+=(ColumnVector<T1> &left, const T2 &right) {
        std::transform(left.data(), left.data() + left.N(), left.data(),
                       [right](T2 element) { return element + right; });
        return left;
    }

    template<typename T1, typename T2>
    ColumnVector<T1> operator+(ColumnVector<T1> &left, const T2 &right) {
        auto result = ColumnVector(left);
        operator+=(result, right);

        return result;
    }

    template<typename T1, typename T2>
    ColumnVector<T2> operator+(const T1 &left, ColumnVector<T2> &right) {
        auto result = Matrix(right);
        operator+=(result, left);

        return result;
    }

    // Operators: minus (for different types)
    template<typename T1, typename T2>
    ColumnVector<T1> &operator-=(ColumnVector<T1> &left,
                                 const ColumnVector<T2> &right) {
        if (left.N() != right.N()) {
            throw std::runtime_error("Size mismatch for operator-=().");
        }

        std::transform(left.data(), left.data() + left.N(), right.data(), left.data(),
                       std::minus<T1>());

        return left;
    }

    template<typename T1, typename T2>
    ColumnVector<T1> operator-(const ColumnVector<T1> &left,
                               const ColumnVector<T2> &right) {
        auto result = ColumnVector(left);
        operator-=(result, right);

        return result;
    }

    // Operators: minus (for scalars)
    template<typename T1, typename T2>
    ColumnVector<T1> &operator-=(ColumnVector<T1> &left, const T2 &right) {
        std::transform(left.data(), left.data() + left.N(), left.data(),
                       [right](T2 element) { return element - right; });
        return left;
    }

    template<typename T1, typename T2>
    ColumnVector<T1> operator-(ColumnVector<T1> &left, const T2 &right) {
        auto result = ColumnVector(left);
        operator-=(result, right);

        return result;
    }

    template<typename T1, typename T2>
    ColumnVector<T2> operator-(const T1 &left, ColumnVector<T2> &right) {
        auto result = ColumnVector(right) * -1.0;
        operator+=(result, left);

        return result;
    }

    // Operators: multiplication from the left
    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1>
    ColumnVector<T2> operator*(const T1 &left, ColumnVector<T2> &right) {
        auto new_mult = T2(left);
        return right * new_mult;
    }

    // Operators: Matrix-vector multiplication
    inline ColumnVector<float> operator*(const Matrix<float> &left, const ColumnVector<float> &right) {
        if (left.cols() != right.N())
            throw std::runtime_error("Size mismatch for operator *().");

        auto result = ColumnVector<float>(left.rows());
        cblas_sgemv(CblasRowMajor, CblasNoTrans, left.rows(), left.cols(), 1.0,
                    left.data(), left.cols(), right.data(), 1, 0.0, result.data(), 1);

        return result;
    }

    inline ColumnVector<double> operator*(const Matrix<double> &left,
                                          const ColumnVector<double> &right) {
        if (left.cols() != right.N())
            throw std::runtime_error("Size mismatch for operator *().");

        auto result = ColumnVector<double>(left.rows());
        cblas_dgemv(CblasRowMajor, CblasNoTrans, left.rows(), left.cols(), 1.0,
                    left.data(), left.cols(), right.data(), 1, 0.0, result.data(), 1);

        return result;
    }

    // Operators: Vector-vector (dot) multiplication
    inline float dot(const ColumnVector<float> &left, const ColumnVector<float> &right) {
        if (left.N() == right.N())
            return cblas_sdot(left.N(), left.data(), 1, right.data(), 1);
        else
            throw std::runtime_error("Shape mismatch for dot().");
    }

    inline double dot(const ColumnVector<double> &left, const ColumnVector<double> &right) {
        if (left.N() == right.N())
            return cblas_ddot(left.N(), left.data(), 1, right.data(), 1);
        else
            throw std::runtime_error("Shape mismatch for dot().");
    }

    template<typename T>
    T fabs(ColumnVector<T> &v) {
        return std::sqrt(dot(v, v));
    }
} // namespace cppmatrix

#endif
