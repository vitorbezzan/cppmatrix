/**
 * @file vector.h
 * @brief Provides unified vector operations and types.
 * 
 * This module provides:
 * - A unified Vector type that can represent both row and column vectors
 * - Mixed-type operations between row and column vectors
 * - BLAS-optimized dot product operations for float and double types
 * - Generic vector operations that work with both vector types
 * - Template specializations for optimized performance
 * - Consistent interface for vector operations
 * - Exception handling for dimension mismatches
 * - Support for standard library algorithms
 */

#ifndef VECTOR_H
#define VECTOR_H

#include "column_vector.h"
#include "row_vector.h"

namespace cppmatrix {
    template<typename T1, typename T2>
    T1 dot(const RowVector<T1> &left, const ColumnVector<T2> &right) {
        if (left.N() == right.N())
            return std::inner_product(left.data(), left.data() + left.N(), right.data(),
                                      T1(0.0));
        else
            throw std::runtime_error("Shape mismatch for dot(): vector sizes must match.");
    }

    template<>
    inline float dot(const RowVector<float> &left, const ColumnVector<float> &right) {
        if (left.N() == right.N())
            return cblas_sdot(left.N(), left.data(), 1, right.data(), 1);
        else
            throw std::runtime_error("Shape mismatch for dot(): vector sizes must match.");
    }

    template<>
    inline double dot(const RowVector<double> &left, const ColumnVector<double> &right) {
        if (left.N() == right.N())
            return cblas_ddot(left.N(), left.data(), 1, right.data(), 1);
        else
            throw std::runtime_error("Shape mismatch for dot(): vector sizes must match.");
    }

    template<typename T1, typename T2>
    T1 dot(const T1 &left, const T2 &right) {
        return left * right;
    }
}

#endif