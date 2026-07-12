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
 * - Template-based mixed-type operations
 * - Various constructors including function-based initialization
 * - Optimized memory layout for performance
 */

#ifndef COLUMN_VECTOR_H
#define COLUMN_VECTOR_H

#include "matrix.h"
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#ifdef CPPMATRIX_USE_OPENMP
#include <omp.h>
#endif

namespace cppmatrix {
    template<typename T>
    class RowVector;

    template<typename T>
    class ColumnVector final : public Matrix<T> {
    public:
        template<typename U>
        friend class ColumnVector;

        ColumnVector() : Matrix<T>() {
        };

        explicit ColumnVector(const uint64_t& N) : Matrix<T>(N, 1) { this->_N = N; }

        ColumnVector(const uint64_t& N, const T &value) : Matrix<T>(N, 1, value) {
            this->_N = N;
        }

        template<uint64_t ndim>
        explicit ColumnVector(const T (&values)[ndim]) : Matrix<T>(ndim, 1) {
            this->_N = ndim;
            std::copy(values, values + ndim, this->data());
        }

        ColumnVector(const uint64_t& N, T (*f)(const uint64_t&)) : Matrix<T>(N, 1) {
            this->_N = N;
            for (uint64_t n = 0; n < this->_N; n++)
                this->operator()(n) = f(n);
        }

        ColumnVector(const uint64_t& N, const std::function<T(const uint64_t&)>& f)
            : Matrix<T>(N, 1) {
            this->_N = N;
            for (uint64_t n = 0; n < this->_N; n++)
                this->operator()(n) = f(n);
        }

        explicit ColumnVector(const Matrix<T>& M) : Matrix<T>(M) {
            if (M.cols() > 1)
                throw std::runtime_error(
                    "Size mismatch for ColumnVector constructor: matrix must have only one column.");

            this->_N = M.rows();
        }

        ColumnVector(const ColumnVector<T>& v) : Matrix<T>(v.N(), 1) {
            this->_N = v.N();
            std::copy(v.data(), v.data() + v.N(), this->data());
        }

        [[nodiscard]] RowVector<T> transpose() const;

        T& operator()(uint64_t n) { return Matrix<T>::operator()(n, 0); }
        const T& operator()(uint64_t n) const { return Matrix<T>::operator()(n, 0); }

        T& operator[](uint64_t n) { return this->operator()(n); }
        const T& operator[](uint64_t n) const { return this->operator()(n); }

        template<typename T2>
        ColumnVector<T>& operator*=(const T2 &right) {
            std::transform(
                this->data(), this->data() + this->N(), this->data(),
                std::bind(std::multiplies<T>(), std::placeholders::_1, T(right)));
            return *this;
        }

        template<typename T2>
        ColumnVector<T> operator*(const T2 &right) const {
            ColumnVector<T> result(*this);
            result *= right;
            return result;
        }

        template<typename T2>
        requires std::is_arithmetic_v<T2>
        ColumnVector<T>& operator/=(const T2 &right) {
            if (right == T2(0))
                throw std::runtime_error("Division by zero.");
            std::transform(
                this->data(), this->data() + this->N(), this->data(),
                std::bind(std::divides<T>(), std::placeholders::_1, T(right)));
            return *this;
        }

        template<typename T2>
        requires std::is_arithmetic_v<T2>
        ColumnVector<T> operator/(const T2 &right) const {
            ColumnVector<T> result(*this);
            result /= right;
            return result;
        }

        ColumnVector<T> operator-() const {
            ColumnVector<T> result(*this);
            result *= T(-1);
            return result;
        }

        template<typename T2>
        bool operator==(const ColumnVector<T2>& right) const {
            if (this->N() != right.N())
                return false;
            for (uint64_t i = 0; i < this->N(); i++)
                if (std::abs(this->operator()(i) - T(right(i))) > std::numeric_limits<T>::epsilon())
                    return false;
            return true;
        }

        template<typename T2>
        bool operator!=(const ColumnVector<T2>& right) const {
            return !(*this == right);
        }

        [[nodiscard]] uint64_t N() const override { return this->_N; }

    private:
        uint64_t _N = 0;
    };

    template<typename T1, typename T2>
    ColumnVector<T1>& operator+=(ColumnVector<T1>& left,
                                 const ColumnVector<T2>& right) {
        if (left.N() != right.N()) {
            throw std::runtime_error("Size mismatch for operator+=(): vector sizes must match.");
        }

        std::transform(left.data(), left.data() + left.N(), right.data(), left.data(),
                       std::plus<T1>());

        return left;
    }

    template<typename T1, typename T2>
    ColumnVector<T1> operator+(const ColumnVector<T1>& left,
                               const ColumnVector<T2>& right) {
        auto result = ColumnVector(left);
        operator+=(result, right);

        return result;
    }

    template<typename T1, typename T2>
    requires std::is_arithmetic_v<T2>
    ColumnVector<T1>& operator+=(ColumnVector<T1>& left, const T2 &right) {
        std::transform(left.data(), left.data() + left.N(), left.data(),
        [right](T1 element) { return element + T1(right); });
        return left;
    }

    template<typename T1, typename T2>
    requires std::is_arithmetic_v<T2>
    ColumnVector<T1> operator+(const ColumnVector<T1>& left, const T2 &right) {
        auto result = ColumnVector(left);
        operator+=(result, right);

        return result;
    }

    template<typename T1, typename T2>
    requires std::is_arithmetic_v<T1>
    ColumnVector<T2> operator+(const T1 &left, const ColumnVector<T2>& right) {
        auto result = ColumnVector(right);
        operator+=(result, T2(left));

        return result;
    }

    template<typename T1, typename T2>
    ColumnVector<T1>& operator-=(ColumnVector<T1>& left,
                                 const ColumnVector<T2>& right) {
        if (left.N() != right.N()) {
            throw std::runtime_error("Size mismatch for operator-=(): vector sizes must match.");
        }

        std::transform(left.data(), left.data() + left.N(), right.data(), left.data(),
                       std::minus<T1>());

        return left;
    }

    template<typename T1, typename T2>
    ColumnVector<T1> operator-(const ColumnVector<T1>& left,
                               const ColumnVector<T2>& right) {
        auto result = ColumnVector(left);
        operator-=(result, right);

        return result;
    }

    template<typename T1, typename T2>
    requires std::is_arithmetic_v<T2>
    ColumnVector<T1>& operator-=(ColumnVector<T1>& left, const T2 &right) {
        std::transform(left.data(), left.data() + left.N(), left.data(),
        [right](T1 element) { return element - T1(right); });
        return left;
    }

    template<typename T1, typename T2>
    requires std::is_arithmetic_v<T2>
    ColumnVector<T1> operator-(const ColumnVector<T1>& left, const T2 &right) {
        auto result = ColumnVector(left);
        operator-=(result, right);

        return result;
    }

    template<typename T1, typename T2>
    requires std::is_arithmetic_v<T1>
    ColumnVector<T2> operator-(const T1 &left, const ColumnVector<T2>& right) {
        auto result = ColumnVector(right) * T2(-1.0);
        operator+=(result, T2(left));

        return result;
    }

    template<typename T1, typename T2>
    requires std::is_arithmetic_v<T1>
    ColumnVector<T2> operator*(const T1 &left, const ColumnVector<T2>& right) {
        return right * T2(left);
    }

    template<typename T>
    ColumnVector<T> operator-(const ColumnVector<T>& value) {
        return -value;
    }

    inline ColumnVector<float> operator*(const Matrix<float>& left, const ColumnVector<float>& right) {
        if (left.cols() != right.N())
            throw std::runtime_error("Size mismatch for operator *().");

        auto result = ColumnVector<float>(left.rows());
        cblas_sgemv(CblasRowMajor, CblasNoTrans, left.rows(), left.cols(), 1.0,
                    left.data(), left.cols(), right.data(), 1, 0.0, result.data(), 1);

        return result;
    }

    inline ColumnVector<double> operator*(const Matrix<double>& left,
                                          const ColumnVector<double>& right) {
        if (left.cols() != right.N())
            throw std::runtime_error("Size mismatch for operator *().");

        auto result = ColumnVector<double>(left.rows());
        cblas_dgemv(CblasRowMajor, CblasNoTrans, left.rows(), left.cols(), 1.0,
                    left.data(), left.cols(), right.data(), 1, 0.0, result.data(), 1);

        return result;
    }

    inline float dot(const ColumnVector<float>& left, const ColumnVector<float>& right) {
        if (left.N() == right.N())
            return cblas_sdot(left.N(), left.data(), 1, right.data(), 1);
        else
            throw std::runtime_error("Shape mismatch for dot().");
    }

    inline double dot(const ColumnVector<double>& left, const ColumnVector<double>& right) {
        if (left.N() == right.N())
            return cblas_ddot(left.N(), left.data(), 1, right.data(), 1);
        else
            throw std::runtime_error("Shape mismatch for dot().");
    }

    template<typename T>
    T dot(const ColumnVector<T>& left, const ColumnVector<T>& right) {
        if (left.N() != right.N())
            throw std::runtime_error("Shape mismatch for dot().");
#ifdef CPPMATRIX_USE_OPENMP
        T sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (uint64_t i = 0; i < left.N(); ++i)
            sum += left(i) * right(i);
        return sum;
#else
        return std::inner_product(left.data(), left.data() + left.N(), right.data(), T(0));
#endif
    }

    template<typename T>
    T norm(const ColumnVector<T>& v) {
        return std::sqrt(dot(v, v));
    }

    template<typename T>
    T squared_norm(const ColumnVector<T>& v) { return dot(v, v); }

    template<typename T>
    T inverse_squared_norm(const ColumnVector<T>& v) { return 1.0 / dot(v, v); }

    template<typename T>
    void print(const ColumnVector<T>& v, const int& precision = 5) {
        for (uint64_t i = 0; i < v.N(); i++) {
            std::print("{:.{}} \n", v(i), precision);
        }
        std::print("\n");
    }
}

#endif