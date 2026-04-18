#include "../include/cppmatrix.h"
#include <gtest/gtest.h>

using namespace cppmatrix;

namespace {
    Matrix<double> make_batch_matrix(const double base) {
        return Matrix<double>(2, 2, [base](const uint64_t& i, const uint64_t& j) {
            return base + static_cast<double>(i * 2 + j);
        });
    }

    ColumnVector<double> make_batch_vector(const double base) {
        return ColumnVector<double>({base, base + 1.0});
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
    EXPECT_THROW((void) batch::sum(std::vector<Matrix<double> > {}), std::runtime_error);
    EXPECT_THROW((void) batch::mean(std::vector<Matrix<double> > {}), std::runtime_error);
}

TEST(batch_operations, add_subtract_transform_match_scalar_equivalents) {
    std::vector<Matrix<double> > lefts = {make_batch_matrix(1.0), make_batch_matrix(5.0)};
    std::vector<Matrix<double> > rights = {make_batch_matrix(10.0), make_batch_matrix(20.0)};

    auto sums = batch::add(lefts, rights);
    auto diffs = batch::subtract(lefts, rights);
    auto scaled = batch::transform(lefts, [](const Matrix<double>& M) { return M * 2.0; });

    ASSERT_EQ(sums.size(), lefts.size());
    ASSERT_EQ(diffs.size(), lefts.size());
    ASSERT_EQ(scaled.size(), lefts.size());

    for (std::size_t i = 0; i < lefts.size(); ++i) {
        EXPECT_TRUE(sums[i] == (lefts[i] + rights[i]));
        EXPECT_TRUE(diffs[i] == (lefts[i] - rights[i]));
        EXPECT_TRUE(scaled[i] == (lefts[i] * 2.0));
    }
}

TEST(batch_operations, multiply_vector_matches_matrix_vector_product) {
    std::vector<Matrix<double> > matrices = {make_batch_matrix(1.0), make_batch_matrix(3.0)};
    std::vector<ColumnVector<double> > vectors = {make_batch_vector(2.0), make_batch_vector(4.0)};

    auto result = batch::multiply_vector(matrices, vectors);
    ASSERT_EQ(result.size(), matrices.size());

    for (std::size_t i = 0; i < matrices.size(); ++i) {
        auto expected = matrices[i] * vectors[i];
        EXPECT_TRUE(result[i] == expected);
    }
}

TEST(batch_operations, add_subtract_and_multiply_vector_throw_for_invalid_inputs) {
    std::vector<Matrix<double> > lefts = {make_batch_matrix(1.0), make_batch_matrix(2.0)};
    std::vector<Matrix<double> > rights_short = {make_batch_matrix(3.0)};
    EXPECT_THROW((void) batch::add(lefts, rights_short), std::runtime_error);
    EXPECT_THROW((void) batch::subtract(lefts, rights_short), std::runtime_error);

    std::vector<Matrix<double> > matrices = {make_batch_matrix(1.0), make_batch_matrix(2.0)};
    std::vector<ColumnVector<double> > vectors_short = {make_batch_vector(1.0)};
    EXPECT_THROW((void) batch::multiply_vector(matrices, vectors_short), std::runtime_error);
}


