/**
 * @file batch_operations.h
 * @brief Batch processing operations for multiple matrices.
 * 
 * This module provides:
 * - Parallel batch matrix multiplication
 * - Batch element-wise operations
 * - Efficient processing of multiple independent operations
 * - Automatic load balancing across threads
 * - Significant speedup for processing arrays of matrices
 */

#ifndef BATCH_OPERATIONS_H
#define BATCH_OPERATIONS_H

#include "matrix.h"
#include <vector>
#include <stdexcept>
#ifdef CPPMATRIX_USE_OPENMP
#include <omp.h>
#endif

namespace cppmatrix {
    namespace batch {
        /**
         * @brief Batch matrix multiplication: C[i] = A[i] * B[i]
         * 
         * Performs multiple independent matrix multiplications in parallel.
         * Each operation is independent, allowing perfect parallelization.
         * 
         * @param lefts Vector of left matrices
         * @param rights Vector of right matrices
         * @return Vector of result matrices
         * @throws std::runtime_error if vector sizes don't match
         */
        template<typename T>
        std::vector<Matrix<T> > multiply(
            const std::vector<Matrix<T> > &lefts,
            const std::vector<Matrix<T> > &rights
        ) {
            if (lefts.size() != rights.size()) {
                throw std::runtime_error(
                    "Batch multiply: left and right vector sizes must match"
                );
            }

            const std::size_t n = lefts.size();
            std::vector<Matrix<T> > results(n);

#ifdef CPPMATRIX_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (std::size_t i = 0; i < n; ++i) {
                results[i] = lefts[i] * rights[i];
            }

            return results;
        }

        /**
         * @brief Batch matrix-scalar multiplication: C[i] = A[i] * scalars[i]
         * 
         * @param matrices Vector of matrices
         * @param scalars Vector of scalars
         * @return Vector of result matrices
         */
        template<typename T>
        std::vector<Matrix<T> > multiply_scalar(
            const std::vector<Matrix<T> > &matrices,
            const std::vector<T> &scalars
        ) {
            if (matrices.size() != scalars.size()) {
                throw std::runtime_error(
                    "Batch multiply_scalar: vector sizes must match"
                );
            }

            const std::size_t n = matrices.size();
            std::vector<Matrix<T> > results(n);

#ifdef CPPMATRIX_USE_OPENMP
#pragma omp parallel for
#endif
            for (std::size_t i = 0; i < n; ++i) {
                results[i] = matrices[i] * scalars[i];
            }

            return results;
        }

        /**
         * @brief Batch matrix addition: C[i] = A[i] + B[i]
         * 
         * @param lefts Vector of left matrices
         * @param rights Vector of right matrices
         * @return Vector of result matrices
         */
        template<typename T>
        std::vector<Matrix<T> > add(
            const std::vector<Matrix<T> > &lefts,
            const std::vector<Matrix<T> > &rights
        ) {
            if (lefts.size() != rights.size()) {
                throw std::runtime_error(
                    "Batch add: vector sizes must match"
                );
            }

            const std::size_t n = lefts.size();
            std::vector<Matrix<T> > results(n);

#ifdef CPPMATRIX_USE_OPENMP
#pragma omp parallel for
#endif
            for (std::size_t i = 0; i < n; ++i) {
                results[i] = lefts[i] + rights[i];
            }

            return results;
        }

        /**
         * @brief Batch matrix subtraction: C[i] = A[i] - B[i]
         * 
         * @param lefts Vector of left matrices
         * @param rights Vector of right matrices
         * @return Vector of result matrices
         */
        template<typename T>
        std::vector<Matrix<T> > subtract(
            const std::vector<Matrix<T> > &lefts,
            const std::vector<Matrix<T> > &rights
        ) {
            if (lefts.size() != rights.size()) {
                throw std::runtime_error(
                    "Batch subtract: vector sizes must match"
                );
            }

            const std::size_t n = lefts.size();
            std::vector<Matrix<T> > results(n);

#ifdef CPPMATRIX_USE_OPENMP
#pragma omp parallel for
#endif
            for (std::size_t i = 0; i < n; ++i) {
                results[i] = lefts[i] - rights[i];
            }

            return results;
        }

        /**
         * @brief Batch matrix transpose: B[i] = A[i]^T
         * 
         * @param matrices Vector of matrices to transpose
         * @return Vector of transposed matrices
         */
        template<typename T>
        std::vector<Matrix<T> > transpose(
            const std::vector<Matrix<T> > &matrices
        ) {
            const std::size_t n = matrices.size();
            std::vector<Matrix<T> > results(n);

#ifdef CPPMATRIX_USE_OPENMP
#pragma omp parallel for
#endif
            for (std::size_t i = 0; i < n; ++i) {
                results[i] = matrices[i].transpose();
            }

            return results;
        }

        /**
         * @brief Apply a function to each matrix in parallel
         * 
         * Generic batch operation that applies a transformation function
         * to each matrix independently.
         * 
         * @param matrices Input matrices
         * @param func Transformation function
         * @return Vector of transformed matrices
         */
        template<typename T, typename Func>
        std::vector<Matrix<T> > transform(
            const std::vector<Matrix<T> > &matrices,
            Func func
        ) {
            const std::size_t n = matrices.size();
            std::vector<Matrix<T> > results(n);

#ifdef CPPMATRIX_USE_OPENMP
#pragma omp parallel for
#endif
            for (std::size_t i = 0; i < n; ++i) {
                results[i] = func(matrices[i]);
            }

            return results;
        }

        /**
         * @brief Batch matrix-vector multiplication: y[i] = A[i] * x[i]
         * 
         * @param matrices Vector of matrices
         * @param vectors Vector of column vectors
         * @return Vector of result vectors
         */
        template<typename T>
        std::vector<ColumnVector<T> > multiply_vector(
            const std::vector<Matrix<T> > &matrices,
            const std::vector<ColumnVector<T> > &vectors
        ) {
            if (matrices.size() != vectors.size()) {
                throw std::runtime_error(
                    "Batch multiply_vector: vector sizes must match"
                );
            }

            const std::size_t n = matrices.size();
            std::vector<ColumnVector<T> > results(n);

#ifdef CPPMATRIX_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (std::size_t i = 0; i < n; ++i) {
                results[i] = matrices[i] * vectors[i];
            }

            return results;
        }

        /**
         * @brief Compute sum of all matrices in batch
         * 
         * Efficiently sums multiple matrices using parallel reduction.
         * All matrices must have the same dimensions.
         * 
         * @param matrices Vector of matrices to sum
         * @return Sum of all matrices
         */
        template<typename T>
        Matrix<T> sum(const std::vector<Matrix<T> > &matrices) {
            if (matrices.empty()) {
                throw std::runtime_error("Batch sum: empty vector");
            }

            Matrix<T> result(matrices[0]);

#ifdef CPPMATRIX_USE_OPENMP
#pragma omp parallel for
#endif
            for (std::size_t i = 1; i < matrices.size(); ++i) {
#ifdef CPPMATRIX_USE_OPENMP
#pragma omp critical
#endif
                result += matrices[i];
            }

            return result;
        }

        /**
         * @brief Compute element-wise mean of all matrices
         * 
         * @param matrices Vector of matrices
         * @return Mean matrix
         */
        template<typename T>
        Matrix<T> mean(const std::vector<Matrix<T> > &matrices) {
            if (matrices.empty()) {
                throw std::runtime_error("Batch mean: empty vector");
            }

            Matrix<T> result = sum(matrices);
            result /= static_cast<T>(matrices.size());

            return result;
        }
    }
}

#endif