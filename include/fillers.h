#ifndef FILLERS_H
#define FILLERS_H

#include <functional>
#include <random>

namespace cppmatrix {

// Some direct filler functions and classes
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

// Base filler for complex fill- matrices
template <typename T> class BaseFill {

public:
  virtual std::function<T(const uint64_t &, const uint64_t &)> filler() {
    return [](const uint64_t &i, const uint64_t &j) { return T(0); };
  }
};

// Base filler for vectors
template <typename T> class VBaseFill {

public:
  virtual std::function<T(const uint64_t &)> filler() {
    return [](const uint64_t &i) { return T(0); };
  }
};

// More fillers, now based in statistical distros
template <typename T> class BaseRandomFill : BaseFill<T> {

public:
  BaseRandomFill(const uint64_t &seed) { this->_rng = std::mt19937_64(seed); }

  virtual ~BaseRandomFill() = default;

  virtual std::function<T(const uint64_t &, const uint64_t &)> filler() {
    return [this](const uint64_t &i, const uint64_t &j) {
      return std::uniform_real_distribution<T>(0.0, 1.0)(this->_rng);
    };
  }

  std::mt19937_64 _rng;
};

template <typename T> class NormalFill : BaseRandomFill<T> {

public:
  NormalFill(const uint64_t &seed, const T &mean, const T &std)
      : BaseRandomFill<T>(seed) {
    this->_mean = mean;
    this->_std = std;
  }

  std::function<T(const uint64_t &, const uint64_t &)> filler() final {
    return [this](const uint64_t &i, const uint64_t &j) {
      return std::normal_distribution<T>(this->_mean, this->_std)(this->_rng);
    };
  }

  T _mean;
  T _std;
};

// More fillers, now based in statistical distros- vectors
template <typename T> class VBaseRandomFill : VBaseFill<T> {

public:
  VBaseRandomFill(const uint64_t &seed) { this->_rng = std::mt19937_64(seed); }

  virtual ~VBaseRandomFill() = default;

  virtual std::function<T(const uint64_t &)> filler() {
    return [this](const uint64_t &i) {
      return std::uniform_real_distribution<T>(0.0, 1.0)(this->_rng);
    };
  }

  std::mt19937_64 _rng;
};

template <typename T> class VNormalFill : VBaseRandomFill<T> {

public:
  VNormalFill(const uint64_t &seed, const T &mean, const T &std)
      : VBaseRandomFill<T>(seed) {
    this->_mean = mean;
    this->_std = std;
  }

  std::function<T(const uint64_t &)> filler() final {
    return [this](const uint64_t &i) {
      return std::normal_distribution<T>(this->_mean, this->_std)(this->_rng);
    };
  }

  T _mean;
  T _std;
};

} // namespace cppmatrix

#endif // FILLERS_H
