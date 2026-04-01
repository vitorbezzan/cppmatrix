#include "tests/batch_operations.cpp"
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