/**
 * @file tests.cpp
 * @brief Main test runner for the C++ Matrix Library test suite
 * 
 * This file serves as the entry point for all unit tests in the C++ Matrix Library.
 * It includes various test suites that verify the functionality of different components:
 * 
 * - dot.cpp: Tests for vector dot product operations
 * - integration.cpp: Tests for numerical integration methods
 * - matrix.cpp: Tests for matrix operations and manipulations
 * - ndarray.cpp: Tests for n-dimensional array functionality
 * - newton.cpp: Tests for Newton's method implementations
 * 
 * The test suite uses Google Test framework for C++ unit testing.
 * 
 * To run the tests, compile this file with Google Test linked and execute
 * the resulting binary.
 */

#include "tests/dot.cpp"
#include "tests/integration.cpp"
#include "tests/matrix.cpp"
#include "tests/ndarray.cpp"
#include "tests/newton.cpp"
#include <gtest/gtest.h>

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
