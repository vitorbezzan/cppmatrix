/* tests.cpp - main test files */

#include <gtest/gtest.h>
#include "tests/ndarray.cpp"


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
