#ifndef NEWTON_H
#define NEWTON_H

#include "function.h"

namespace cppmatrix {

// Defines a rootfinding with defined precision and multiple outputs
template <typename _Precision, typename _Input, typename _FOutput>
class BaseRootFind {

public:
  BaseRootFind(const _Precision &precision, const uint64_t &n) {
    this->_precision = precision;
    this->_n = n;
  }

  virtual ~BaseRootFind() = default;

  inline _Precision precision() { return this->_precision; }
  inline uint64_t n() { return this->_n; }

  virtual _FOutput run(const _Input &x0) { return _FOutput(); }

private:
  _Precision _precision;
  uint64_t _n;
};

// Rootfinder for RealFunction
template <_RealFunction _F>
class Newton : public BaseRootFind<PrecisionT<_F>, InputT<_F>, FOutputT<_F>> {

public:
  Newton(const _F &f, const PrecisionT<_F> &precision, const uint64_t &n)
      : BaseRootFind<PrecisionT<_F>, InputT<_F>, FOutputT<_F>>(precision, n) {
    this->_f = f;
  }

  FOutputT<_F> run(const PrecisionT<_F> &x0) {
    InputT<_F> x = x0;
    InputT<_F> x_new;

    for (uint64_t N = 0; N < this->n(); N++) {

      x_new = x - this->_f(x) / this->_f.d1(x);

      if (fabs(x_new - x) <= this->precision())
        break;

      x = x_new;
    }

    return x;
  }

private:
  _F _f;
};

} // namespace cppmatrix

#endif // NEWTON_H
