/**
 * @file expression_templates.h
 * @brief Expression templates for lazy evaluation and elimination of temporaries.
 * 
 * This module provides:
 * - Lazy evaluation of matrix/vector expressions
 * - Elimination of intermediate temporaries in compound expressions
 * - Automatic fusion of operations for better cache locality
 * - Type-safe compile-time expression building
 * - Significant performance improvements for complex expressions like A + B * C
 */

#ifndef EXPRESSION_TEMPLATES_H
#define EXPRESSION_TEMPLATES_H

#include <type_traits>
#include <cstdint>

namespace cppmatrix {
    template<typename T>
    class Matrix;
    template<typename T>
        requires std::is_floating_point_v<T>
    class NDArray;

    template<typename E>
    class MatrixExpression {
    public:
        using value_type = typename E::value_type;

        const E &self() const { return static_cast<const E &>(*this); }
        E &self() { return static_cast<E &>(*this); }

        uint64_t rows() const { return self().rows(); }
        uint64_t cols() const { return self().cols(); }

        value_type operator()(uint64_t i, uint64_t j) const {
            return self()(i, j);
        }
    };

    template<typename Op, typename L, typename R>
    class BinaryMatrixExpr : public MatrixExpression<BinaryMatrixExpr<Op, L, R> > {
    public:
        using value_type = typename L::value_type;

        BinaryMatrixExpr(const L &left, const R &right)
            : _left(left), _right(right) {
        }

        uint64_t rows() const { return _left.rows(); }
        uint64_t cols() const { return _left.cols(); }

        value_type operator()(uint64_t i, uint64_t j) const {
            return Op::apply(_left(i, j), _right(i, j));
        }

    private:
        const L &_left;
        const R &_right;
    };

    template<typename E, typename S>
    class ScalarMultExpr : public MatrixExpression<ScalarMultExpr<E, S> > {
    public:
        using value_type = typename E::value_type;

        ScalarMultExpr(const E &expr, S scalar)
            : _expr(expr), _scalar(scalar) {
        }

        uint64_t rows() const { return _expr.rows(); }
        uint64_t cols() const { return _expr.cols(); }

        value_type operator()(uint64_t i, uint64_t j) const {
            return _expr(i, j) * value_type(_scalar);
        }

    private:
        const E &_expr;
        S _scalar;
    };

    struct AddOp {
        template<typename T1, typename T2>
        static auto apply(const T1 &a, const T2 &b) -> decltype(a + b) {
            return a + b;
        }
    };

    struct SubOp {
        template<typename T1, typename T2>
        static auto apply(const T1 &a, const T2 &b) -> decltype(a - b) {
            return a - b;
        }
    };

    template<typename L, typename R>
    auto operator+(const MatrixExpression<L> &left, const MatrixExpression<R> &right) {
        return BinaryMatrixExpr<AddOp, L, R>(left.self(), right.self());
    }

    template<typename L, typename R>
    auto operator-(const MatrixExpression<L> &left, const MatrixExpression<R> &right) {
        return BinaryMatrixExpr<SubOp, L, R>(left.self(), right.self());
    }

    template<typename E, typename S>
        requires std::is_arithmetic_v<S>
    auto operator*(const MatrixExpression<E> &expr, S scalar) {
        return ScalarMultExpr<E, S>(expr.self(), scalar);
    }

    template<typename S, typename E>
        requires std::is_arithmetic_v<S>
    auto operator*(S scalar, const MatrixExpression<E> &expr) {
        return ScalarMultExpr<E, S>(expr.self(), scalar);
    }

    template<typename T>
    class MatrixWrapper : public MatrixExpression<MatrixWrapper<T> > {
    public:
        using value_type = T;

        explicit MatrixWrapper(const Matrix<T> &mat) : _mat(mat) {
        }

        uint64_t rows() const { return _mat.rows(); }
        uint64_t cols() const { return _mat.cols(); }

        T operator()(uint64_t i, uint64_t j) const {
            return _mat(i, j);
        }

    private:
        const Matrix<T> &_mat;
    };

    template<typename T>
    MatrixWrapper<T> expr(const Matrix<T> &mat) {
        return MatrixWrapper<T>(mat);
    }

    template<typename T, typename E>
    void evaluate_into(Matrix<T> &result, const MatrixExpression<E> &expr) {
        const uint64_t rows = expr.rows();
        const uint64_t cols = expr.cols();

#ifdef CPPMATRIX_USE_OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (uint64_t i = 0; i < rows; ++i) {
            for (uint64_t j = 0; j < cols; ++j) {
                result(i, j) = expr(i, j);
            }
        }
    }

    template<typename T, typename E>
    Matrix<T> &assign_from_expr(Matrix<T> &mat, const MatrixExpression<E> &expr) {
        evaluate_into(mat, expr);
        return mat;
    }
}

#endif