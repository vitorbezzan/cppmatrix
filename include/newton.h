#ifndef NEWTON_H
#define NEWTON_H

#include "function.h"
#include "matrix.h"
#include <type_traits>

namespace cppmatrix {

// Defines a rootfinding with variable precision and multiple outputs
template <typename _Precision, typename _Input, typename _Output>
  requires std::is_floating_point_v<_Precision>
class BaseRootFind {

public:
  BaseRootFind(const _Precision &precision, const uint64_t &n) {
    this->_precision = precision;
    this->_n = n;
  }

  virtual ~BaseRootFind() = default;

  inline _Precision precision() { return this->_precision; }
  inline uint64_t n() { return this->_n; }

  virtual _Output run(const _Input &x0) { return _Output(); }

private:
  _Precision _precision;
  uint64_t _n;
};

template <class _F>
class Newton : public BaseRootFind<InputT<_F>, InputT<_F>, FOutputT<_F>> {

public:
  Newton(const _F &f, const InputT<_F> &precision, const uint64_t &n)
      : BaseRootFind<InputT<_F>, InputT<_F>, FOutputT<_F>>(precision, n) {
    this->_f = f;
  }

  FOutputT<_F> run(const InputT<_F> &x0) {
    InputT<_F> x = x0;
    InputT<_F> x_new;

    for (uint64_t N = 0; N < this->n(); N++) {

      if (std::is_same<InputT<_F>, FOutputT<_F>>::value)
        x_new = x - this->_f(x) / this->_f.d1(x);
      else
        x_new = x - dot(this->_f(x), this->_f.d1(x)) / dot(x, x);

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
