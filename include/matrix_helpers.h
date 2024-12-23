#ifndef HELPERS_H
#define HELPERS_H

#include "matrix.h"
#include <iomanip>
#include <iostream>
#include <random>

namespace cppmatrix {

std::random_device dev;

// Some filler functions
template <typename T> T identity(const uint64_t &i, const uint64_t &j) {
  if (i == j)
    return T(1);

  return T(0);
}

template <typename T> T zeros(const uint64_t &i, const uint64_t &j) {
  return T(0);
}

template <typename T> T ones(const uint64_t &i, const uint64_t &j) {
  return T(1);
}

template <typename T> T normal(const uint64_t &i, const uint64_t &j) {
  std::mt19937 rng(dev());
  std::normal_distribution<> dist(0.0, 1.0);

  return dist(rng);
}

template <typename T> T uniform(const uint64_t &i, const uint64_t &j) {
  std::mt19937 rng(dev());
  std::uniform_real_distribution<> dist(0.0, 1.0);

  return dist(rng);
}

// print matrix to stdout
template <typename T> void print(Matrix<T> &M, int digits = 4) {
  std::cout << std::setprecision(digits);
  std::cout << "\n";
  for (uint64_t i = 0; i < M.rows(); i++) {
    for (uint64_t j = 0; j < M.cols(); j++) {
      std::cout << M(i, j) << "\t";
    }
    std::cout << "\n";
  }
}

} // namespace cppmatrix

#endif // HELPERS_H
