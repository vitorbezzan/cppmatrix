#include "../include/cppmatrix.h"
#include <gtest/gtest.h>

using namespace cppmatrix;

namespace {
    Matrix<double> make_batch_matrix(const double base) {
        return Matrix<double>(2, 2, [base](const uint64_t &i, const uint64_t &j) {
            return base + static_cast<double>(i * 2 + j);
        });
    }
}

TEST(batch_operations, multiply_sum_mean_match_scalar_equivalents) {
    std::vector<Matrix<double> > lefts = {make_batch_matrix(1.0), make_batch_matrix(5.0)};
    std::vector<Matrix<double> > rights = {make_batch_matrix(2.0), make_batch_matrix(6.0)};

    auto products = batch::multiply(lefts, rights);
    ASSERT_EQ(products.size(), 2U);

    EXPECT_TRUE(products[0] == multiply_naive(lefts[0], rights[0]));
    EXPECT_TRUE(products[1] == multiply_naive(lefts[1], rights[1]));

    auto summed = batch::sum(products);
    auto expected_sum = products[0] + products[1];
    EXPECT_TRUE(summed == expected_sum);

    auto averaged = batch::mean(products);
    auto expected_mean = expected_sum / 2.0;
    EXPECT_TRUE(averaged == expected_mean);
}

TEST(batch_operations, throws_for_invalid_batch_inputs) {
    std::vector<Matrix<double> > lefts = {make_batch_matrix(1.0), make_batch_matrix(2.0)};
    std::vector<Matrix<double> > rights = {make_batch_matrix(3.0)};

    EXPECT_THROW((void) batch::multiply(lefts, rights), std::runtime_error);
    EXPECT_THROW((void) batch::sum(std::vector<Matrix<double> >{}), std::runtime_error);
    EXPECT_THROW((void) batch::mean(std::vector<Matrix<double> >{}), std::runtime_error);
}


