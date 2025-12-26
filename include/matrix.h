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
#include <limits>
#include <stdexcept>
#include <print>
#ifdef CPPMATRIX_USE_OPENMP
#include <omp.h>
#endif

#ifndef CPPMATRIX_RESTRICT
#if defined(__clang__) || defined(__GNUC__)
#define CPPMATRIX_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define CPPMATRIX_RESTRICT __restrict
#else
#define CPPMATRIX_RESTRICT
#endif
#endif

namespace cppmatrix {
    template<typename T>
    class Matrix : public NDArray<T> {
    public:
        template<typename U>
        friend class Matrix;

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

        Matrix(const uint64_t &rows, const uint64_t &cols,
               T (*f)(const uint64_t &, const uint64_t &))
            : NDArray<T>({rows, cols}) {
            this->_rows = rows;
            this->_cols = cols;

            for (uint64_t row = 0; row < this->_rows; row++)
                for (uint64_t col = 0; col < this->_cols; col++)
                    this->operator()(row, col) = f(row, col);
        }

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

        Matrix(Matrix<T> &&M) noexcept : NDArray<T>(std::move(M)) {
            this->_rows = M._rows;
            this->_cols = M._cols;
            M._rows = 0;
            M._cols = 0;
        }

        explicit Matrix(const NDArray<T> &base) : NDArray<T>(base) {
            if (base.ndim() != 2)
                throw std::runtime_error("Dimension size mismatch for constructor.");

            this->_rows = base.shape()[0];
            this->_cols = base.shape()[1];
        }

        T &operator()(uint64_t row, uint64_t col) {
            uint64_t index[2] = {row, col};
            return NDArray<T>::operator()(index);
        }

        const T &operator()(uint64_t row, uint64_t col) const {
            uint64_t index[2] = {row, col};
            return NDArray<T>::operator()(index);
        }

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
        Matrix<T> operator*(const T2 &right) const {
            return Matrix<T>(*this) *= right;
        }

        // Operators: division from the right
        template<typename T2>
            requires std::is_floating_point_v<T2>
        Matrix<T> &operator/=(const T2 &right) {
            if (right == T2(0))
                throw std::runtime_error("Division by zero.");
            std::transform(
                this->data(), this->data() + this->N(), this->data(),
                std::bind(std::divides<T>(), std::placeholders::_1, T(right)));
            return *this;
        }

        template<typename T2>
            requires std::is_floating_point_v<T2>
        Matrix<T> operator/(const T2 &right) const {
            Matrix<T> result(*this);
            result /= right;
            return result;
        }

        template<typename T2>
        bool operator==(const Matrix<T2> &right) const {
            if (this->rows() != right.rows() || this->cols() != right.cols())
                return false;
            for (uint64_t i = 0; i < this->rows(); i++)
                for (uint64_t j = 0; j < this->cols(); j++)
                    if (std::abs(this->operator()(i, j) - T(right(i, j))) > std::numeric_limits<T>::epsilon())
                        return false;
            return true;
        }

        template<typename T2>
        bool operator!=(const Matrix<T2> &right) const {
            return !(*this == right);
        }

        Matrix<T> &operator=(const Matrix<T> &right) {
            if (this != &right) {
                NDArray<T>::operator=(right);
                this->_rows = right._rows;
                this->_cols = right._cols;
            }
            return *this;
        }

        Matrix<T> &operator=(Matrix<T> &&right) noexcept {
            if (this != &right) {
                NDArray<T>::operator=(std::move(right));
                this->_rows = right._rows;
                this->_cols = right._cols;
                right._rows = 0;
                right._cols = 0;
            }
            return *this;
        }

        [[nodiscard]] uint64_t rows() const { return _rows; }
        [[nodiscard]] uint64_t cols() const { return _cols; }

        [[nodiscard]] Matrix<T> transpose() const {
            Matrix<T> result(this->_cols, this->_rows);

            constexpr uint64_t BLOCK_SIZE = 32;
            const uint64_t rows = this->_rows;
            const uint64_t cols = this->_cols;

#ifdef CPPMATRIX_USE_OPENMP
#pragma omp parallel for collapse(2)
#endif
            for (uint64_t i0 = 0; i0 < rows; i0 += BLOCK_SIZE) {
                for (uint64_t j0 = 0; j0 < cols; j0 += BLOCK_SIZE) {
                    const uint64_t i_max = std::min(rows, i0 + BLOCK_SIZE);
                    const uint64_t j_max = std::min(cols, j0 + BLOCK_SIZE);

                    for (uint64_t i = i0; i < i_max; ++i) {
                        for (uint64_t j = j0; j < j_max; ++j) {
                            result(j, i) = this->operator()(i, j);
                        }
                    }
                }
            }

            return result;
        }

    private:
        uint64_t _rows = 0;
        uint64_t _cols = 0;
    };

    template<typename T>
    Matrix<T> transpose(const Matrix<T> &M) {
        return M.transpose();
    }

    template<typename T1, typename T2>
    Matrix<T1> &operator+=(Matrix<T1> &left, const Matrix<T2> &right) {
        if ((left.cols() != right.cols()) | (left.rows() != right.rows())) {
            throw std::runtime_error("Size mismatch for operator+=(): matrix dimensions must match.");
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

    template<typename T1, typename T2>
    Matrix<T1> &operator+=(Matrix<T1> &left, const T2 &right) {
        std::transform(left.data(), left.data() + left.N(), left.data(),
                       [right](T1 element) { return element + T1(right); });
        return left;
    }

    template<typename T1, typename T2>
        requires std::is_floating_point_v<T2>
    Matrix<T1> operator+(const Matrix<T1> &left, const T2 &right) {
        Matrix<T1> result(left);
        operator+=(result, right);

        return result;
    }

    template<typename T1, typename T2>
        requires std::is_floating_point_v<T1>
    Matrix<T2> operator+(const T1 &left, const Matrix<T2> &right) {
        Matrix<T2> result(right);
        operator+=(result, T2(left));

        return result;
    }

    template<typename T1, typename T2>
    Matrix<T1> &operator-=(Matrix<T1> &left, const Matrix<T2> &right) {
        if ((left.cols() != right.cols()) | (left.rows() != right.rows())) {
            throw std::runtime_error("Size mismatch for operator-=(): matrix dimensions must match.");
        }

        std::transform(left.data(), left.data() + left.N(), right.data(), left.data(),
                       std::minus<T1>());

        return left;
    }

    template<typename T1, typename T2>
    Matrix<T1> operator-(const Matrix<T1> &left, const Matrix<T2> &right) {
        Matrix<T1> result(left);
        operator-=(result, right);

        return result;
    }

    template<typename T1, typename T2>
    Matrix<T1> &operator-=(Matrix<T1> &left, const T2 &right) {
        std::transform(left.data(), left.data() + left.N(), left.data(),
                       [right](T1 element) { return element - T1(right); });
        return left;
    }

    template<typename T1, typename T2>
        requires std::is_floating_point_v<T2>
    Matrix<T1> operator-(const Matrix<T1> &left, const T2 &right) {
        Matrix<T1> result(left);
        operator-=(result, right);

        return result;
    }

    template<typename T1, typename T2>
        requires std::is_floating_point_v<T1>
    Matrix<T2> operator-(const T1 &left, const Matrix<T2> &right) {
        Matrix<T2> result(right);
        result *= T2(-1.0);
        operator+=(result, T2(left));

        return result;
    }

    template<typename T1, typename T2>
        requires std::is_floating_point_v<T1>
    Matrix<T2> operator*(const T1 &left, const Matrix<T2> &right) {
        return right * T2(left);
    }

    namespace detail {
        template<typename T1, typename T2>
        Matrix<T1> create_result_matrix(const Matrix<T1> &left, const Matrix<T2> &right) {
            if (left.cols() != right.rows())
                throw std::runtime_error(
                    "Shape mismatch for matrix multiplication: left columns must equal right rows.");

            return Matrix<T1>(left.rows(), right.cols(), T1(0));
        }
    }

    template<typename T1, typename T2>
    Matrix<T1> multiply_naive(const Matrix<T1> &left, const Matrix<T2> &right) {
        auto C = detail::create_result_matrix(left, right);

        const uint64_t m = left.rows();
        const uint64_t n = right.cols();
        const uint64_t kdim = left.cols();

        T1 * CPPMATRIX_RESTRICT cptr = C.data();
        const T1 * CPPMATRIX_RESTRICT lptr = left.data();
        const T2 * CPPMATRIX_RESTRICT rptr = right.data();

#ifdef CPPMATRIX_USE_OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (uint64_t i = 0; i < m; i++)
            for (uint64_t j = 0; j < n; j++) {
                const T1 * CPPMATRIX_RESTRICT row = lptr + i * kdim;
                const T2 * CPPMATRIX_RESTRICT col = rptr + j;
                T1 acc = 0;
#ifdef CPPMATRIX_USE_OPENMP
#pragma omp simd reduction(+:acc)
#endif
                for (uint64_t k = 0; k < kdim; k++)
                    acc += row[k] * T1(col[k * n]);
                cptr[i * n + j] = acc;
            }

        return C;
    }

    inline Matrix<float> operator*(const Matrix<float> &left, const Matrix<float> &right) {
        Matrix<float> C = detail::create_result_matrix(left, right);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C.rows(), C.cols(),
                    left.cols(), 1.0, left.data(), left.cols(), right.data(),
                    right.cols(), 0.0, C.data(), C.cols());

        return C;
    }

    inline Matrix<double> operator*(const Matrix<double> &left, const Matrix<double> &right) {
        Matrix<double> C = detail::create_result_matrix(left, right);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C.rows(), C.cols(),
                    left.cols(), 1.0, left.data(), left.cols(), right.data(),
                    right.cols(), 0.0, C.data(), C.cols());

        return C;
    }

    template<typename T>
    void print(const Matrix<T> &M, const int &precision = 5) {
        for (uint64_t i = 0; i < M.rows(); i++) {
            for (uint64_t j = 0; j < M.cols(); j++) {
                std::print("{:.{}} \t", M(i, j), precision);
            }
            std::print("\n");
        }
    }
}

#endif
