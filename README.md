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

## OpenMP

OpenMP is required and enabled by default. Ensure your toolchain provides OpenMP support. On macOS with Homebrew LLVM,
install `libomp` and configure via `brew install libomp`; CMake already links `OpenMP::OpenMP_CXX`.

## Testing

The project uses GoogleTest with a single test runner (`tests.cpp`) that includes all files in `tests/`.

- Run tests with `ctest --test-dir build --output-on-failure`.
- Or run the binary directly with `./build/run_tests`.

Current suites cover:

- Core linear algebra (`tests/matrix.cpp`, `tests/ndarray.cpp`, `tests/dot.cpp`)
- Numerical algorithms (`tests/integration.cpp`, `tests/newton.cpp`)
- Batch operations and error paths (`tests/batch_operations.cpp`)

Recent additions focus on shape-mismatch behavior, division-by-zero guards, multi-start solver behavior, and zero-width integration intervals.

