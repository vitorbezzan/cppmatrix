/**
 * @file matrix.h
 * @brief Provides the Matrix class for 2D array operations.
 * 
 * This module implements a templated Matrix class derived from NDArray that provides:
 * - Efficient 2D matrix operations
 * - BLAS-optimized matrix multiplication for float and double types
 * - Support for scalar operations and matrix arithmetic
 * - Row-major memory layout for optimal performance
 * - Various constructors including function-based initialization
 * - Template-based mixed-type operations
 * - Efficient element access and iteration
 * - Exception handling for size mismatches
 * - Comprehensive operator overloading
 */

#ifndef MATRIX_H
#define MATRIX_H

#include "ndarray.h"
#include <algorithm>
#include <cblas.h>
#include <functional>
#include <stdexcept>
#include <print>

namespace cppmatrix {
    template<typename T>
    class Matrix : public NDArray<T> {
    public:
        // Friend definitions
        template<typename U>
        friend class Matrix;

        // Constructors
        Matrix() : NDArray<T>() {
        };

        Matrix(const uint64_t &rows, const uint64_t &cols)
            : NDArray<T>({rows, cols}) {
            this->_rows = rows;
            this->_cols = cols;
        }

        Matrix(const uint64_t &rows, const uint64_t &cols, const T &value)
            : NDArray<T>({rows, cols}, value) {
            this->_rows = rows;
            this->_cols = cols;
        }

        // Constructor for fill function
        Matrix(const uint64_t &rows, const uint64_t &cols,
               T (*f)(const uint64_t &, const uint64_t &))
            : NDArray<T>({rows, cols}) {
            this->_rows = rows;
            this->_cols = cols;

            for (uint64_t row = 0; row < this->_rows; row++)
                for (uint64_t col = 0; col < this->_cols; col++)
                    this->operator()(row, col) = f(row, col);
        }

        // Constructor for fill function
        Matrix(const uint64_t &rows, const uint64_t &cols,
               const std::function<T(const uint64_t &, const uint64_t &)> &f)
            : NDArray<T>({rows, cols}) {
            this->_rows = rows;
            this->_cols = cols;

            for (uint64_t row = 0; row < this->_rows; row++)
                for (uint64_t col = 0; col < this->_cols; col++)
                    this->operator()(row, col) = f(row, col);
        }

        Matrix(const Matrix<T> &M) : NDArray<T>(M) {
            this->_rows = M._rows;
            this->_cols = M._cols;
        }

        explicit Matrix(const NDArray<T> &base) : NDArray<T>(base) {
            if (base.ndim() != 2)
                throw std::runtime_error("Dimension size mismatch for constructor.");

            this->_rows = base.shape()[0];
            this->_cols = base.shape()[1];
        }

        // Access operator
        T &operator()(uint64_t row, uint64_t col) {
            uint64_t index[2] = {row, col};
            return NDArray<T>::operator()(index);
        }

        const T &operator()(uint64_t row, uint64_t col) const {
            uint64_t index[2] = {row, col};
            return NDArray<T>::operator()(index);
        }

        // Operators: multiplication from the right
        template<typename T2>
            requires std::is_floating_point_v<T2>
        Matrix<T> &operator*=(const T2 &right) {
            std::transform(
                this->data(), this->data() + this->N(), this->data(),
                std::bind(std::multiplies<T>(), std::placeholders::_1, T(right)));
            return *this;
        }

        template<typename T2>
            requires std::is_floating_point_v<T2>
        Matrix<T> operator*(const T2 &right) {
            return Matrix<T>(*this) *= right;
        }

        // Helpers
        [[nodiscard]] uint64_t rows() const { return _rows; }
        [[nodiscard]] uint64_t cols() const { return _cols; }

    private:
        uint64_t _rows = 0;
        uint64_t _cols = 0;
    };

    // Operators: plus (for different types)
    template<typename T1, typename T2>
    Matrix<T1> &operator+=(Matrix<T1> &left, const Matrix<T2> &right) {
        if ((left.cols() != right.cols()) | (left.rows() != right.rows())) {
            throw std::runtime_error("Size mismatch for operator+=().");
        }

        std::transform(left.data(), left.data() + left.N(), right.data(), left.data(),
                       std::plus<T1>());

        return left;
    }

    template<typename T1, typename T2>
    Matrix<T1> operator+(const Matrix<T1> &left, const Matrix<T2> &right) {
        auto result = Matrix(left);
        operator+=(result, right);

        return result;
    }

    // Operators: plus (for scalars)
    template<typename T1, typename T2>
    Matrix<T1> &operator+=(Matrix<T1> &left, const T2 &right) {
        std::transform(left.data(), left.data() + left.N(), left.data(),
                       [right](T2 element) { return element + right; });
        return left;
    }

    template<typename T1, typename T2>
        requires std::is_floating_point_v<T2>
    Matrix<T1> operator+(Matrix<T1> &left, const T2 &right) {
        auto result = Matrix(left);
        operator+=(result, right);

        return result;
    }

    template<typename T1, typename T2>
        requires std::is_floating_point_v<T1>
    Matrix<T2> operator+(const T1 &left, Matrix<T2> &right) {
        auto result = Matrix(right);
        operator+=(result, left);

        return result;
    }

    // Operators: minus (for different types)
    template<typename T1, typename T2>
    Matrix<T1> &operator-=(Matrix<T1> &left, const Matrix<T2> &right) {
        if ((left.cols() != right.cols()) | (left.rows() != right.rows())) {
            throw std::runtime_error("Size mismatch for operator-=().");
        }

        std::transform(left.data(), left.data() + left.N(), right.data(), left.data(),
                       std::minus<T1>());

        return left;
    }

    template<typename T1, typename T2>
    Matrix<T1> operator-(const Matrix<T1> &left, const Matrix<T2> &right) {
        auto result = Matrix(left);
        operator-=(result, right);

        return result;
    }

    // Operators: minus (for scalars)
    template<typename T1, typename T2>
    Matrix<T1> &operator-=(Matrix<T1> &left, const T2 &right) {
        std::transform(left.data(), left.data() + left.N(), left.data(),
                       [right](T2 element) { return element - right; });
        return left;
    }

    template<typename T1, typename T2>
        requires std::is_floating_point_v<T2>
    Matrix<T1> operator-(Matrix<T1> &left, const T2 &right) {
        auto result = Matrix(left);
        operator-=(result, right);

        return result;
    }

    template<typename T1, typename T2>
        requires std::is_floating_point_v<T1>
    Matrix<T2> operator-(const T1 &left, Matrix<T2> &right) {
        auto result = Matrix(right) * -1.0;
        operator+=(result, left);

        return result;
    }

    // Operators: multiplication from the left
    template<typename T1, typename T2>
        requires std::is_floating_point_v<T1>
    Matrix<T2> operator*(const T1 &left, Matrix<T2> &right) {
        auto new_mult = T2(left);
        return right * new_mult;
    }

    // Operators: Matrix multiplication
    template<typename T1, typename T2>
    Matrix<T1> check_shape(const Matrix<T1> &left, const Matrix<T2> &right) {
        if (left.cols() != right.rows())
            throw std::runtime_error("Shape mismatch for check_shape().");

        return Matrix<T1>(left.rows(), right.cols());
    }

    template<typename T1, typename T2>
    Matrix<T1> naive_multiply(Matrix<T1> &left, Matrix<T2> &right) {
        auto C = check_shape(left, right);

        for (uint64_t i = 0; i < left.rows(); i++)
            for (uint64_t j = 0; j < right.cols(); j++)
                for (uint64_t k = 0; k < left.cols(); k++)
                    C(i, j) += left(i, k) * right(k, j);

        return C;
    }

    inline Matrix<float> operator*(const Matrix<float> &left, const Matrix<float> &right) {
        Matrix<float> C = check_shape(left, right);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C.rows(), C.cols(),
                    left.cols(), 1.0, left.data(), left.cols(), right.data(),
                    right.cols(), 0.0, C.data(), C.cols());

        return C;
    }

    inline Matrix<double> operator*(const Matrix<double> &left, const Matrix<double> &right) {
        Matrix<double> C = check_shape(left, right);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C.rows(), C.cols(),
                    left.cols(), 1.0, left.data(), left.cols(), right.data(),
                    right.cols(), 0.0, C.data(), C.cols());

        return C;
    }

    template<typename T>
    void print(const Matrix<T> &M, const int &precision = 5) {

        for(uint64_t i = 0; i < M.rows(); i++) {
            for (uint64_t j = 0; j < M.cols(); j++) {
                std::print("{:.{}} \t", M(i, j), precision);
            }
            std::print("\n");
        }
    }

} // namespace cppmatrix

#endif
