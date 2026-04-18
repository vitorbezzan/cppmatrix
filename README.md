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

- C++23 (project is configured for GNU++23 in `CMakeLists.txt`)
- BLAS library (for optimized operations)
- C++ compiler with template metaprogramming support

## Safety and hardening (untrusted inputs)

`cppmatrix` is performance-oriented by default, but `NDArray` has additional hardening to prevent common UB and overflow issues when **shapes/indices come from untrusted sources**.

- **Allocation caps and overflow checks** (in `NDArray::_allocate`):
  - `CPPMATRIX_MAX_NDIM` (default `16`)
  - `CPPMATRIX_MAX_BYTES` (default `1 GiB`)
- **Indexing**:
  - `NDArray::operator()` supports `std::span<const uint64_t>` and the legacy fixed-array overloads forward to it.
  - `NDArray::at(...)` is the **checked accessor** (throws `std::out_of_range` on rank/bounds errors).
  - Even with checks disabled, rank mismatch (index shorter than `ndim`) throws to avoid UB.
- **Optional bounds checks**:
  - Define `CPPMATRIX_ENABLE_BOUNDS_CHECKS=1` to make `operator()` validate full bounds as well (useful for fuzzing/debugging).

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

Recent additions focus on:

- Batch `add`/`subtract`/`multiply_vector`/`transform` behavior and mismatch paths
- Transpose stress around 32x32 block boundaries
- Mixed-precision matrix/vector arithmetic checks
- Newton/Polyak failure-mode behavior (zero derivative / flat gradient / non-convergence)
- Deterministic BLAS-vs-naive parity for both `float` and `double`

