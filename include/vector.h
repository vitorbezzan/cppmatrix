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
#include <variant>

// Defines operations for mixed-type vectors

namespace cppmatrix {
    template<typename T1, typename T2>
    T1 dot(RowVector<T1> &left, ColumnVector<T2> &right) {
        if (left.N() == right.N())
            return std::inner_product(left.data(), left.data() + left.N(), right.data(),
                                      0.0);
        else
            throw std::runtime_error("Shape mismatch for dot().");
    }

    template<>
    inline float dot(RowVector<float> &left, ColumnVector<float> &right) {
        if (left.N() == right.N())
            return cblas_sdot(left.N(), left.data(), 1, right.data(), 1);
        else
            throw std::runtime_error("Shape mismatch for dot().");
    }

    template<>
    inline double dot(RowVector<double> &left, ColumnVector<double> &right) {
        if (left.N() == right.N())
            return cblas_ddot(left.N(), left.data(), 1, right.data(), 1);
        else
            throw std::runtime_error("Shape mismatch for dot().");
    }

    template<typename T1, typename T2>
    T1 dot(const T1 &left, const T2 &right) {
        return left * right;
    }
} // namespace cppmatrix

#endif
