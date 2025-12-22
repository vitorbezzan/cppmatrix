# cppmatrix

A modern, high-performance C++ linear algebra library that provides optimized matrix and vector operations. The library
is designed with template metaprogramming for type safety and BLAS integration for optimal performance.

## Features

- **Templated Types**: Generic support for floating-point types with compile-time type checking
- **BLAS Integration**: Optimized implementations for float and double types using BLAS
- **Comprehensive Matrix Operations**:
    - Matrix-matrix multiplication
    - Matrix-vector multiplication
    - Element-wise operations (addition, subtraction)
    - Scalar operations
- **Vector Support**:
    - Row and column vectors with specialized operations
    - Dot product calculations
    - Vector norms and operations
- **N-dimensional Array Base**: Flexible foundation for matrix and vector operations
- **Root Finding Algorithms**:
    - Newton's method for real functions
    - Polyak's method for scalar fields
- **Function Abstractions**: Base classes for implementing mathematical functions

## Requirements

- C++20 or later
- BLAS library (for optimized operations)
- C++ compiler with template metaprogramming support

## Key Components

- `Matrix<T>`: Base matrix class with templated type
- `ColumnVector<T>` and `RowVector<T>`: Specialized vector classes
- `NDArray<T>`: N-dimensional array base class
- `Function` abstractions for mathematical operations
- Newton and Polyak root-finding implementations

## Performance

The library uses BLAS for optimized operations on float and double types, providing high-performance linear algebra
computations. For other types, it falls back to efficient C++ implementations.
