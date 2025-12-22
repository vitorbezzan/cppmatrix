/**
 * @file ndarray.h
 * @brief Provides the NDArray class, a base N-dimensional array implementation.
 * 
 * This module implements a templated N-dimensional array class that serves as the foundation
 * for the matrix and vector classes. It provides:
 * - Dynamic memory management for N-dimensional data
 * - Basic arithmetic operations (+, -, *) with type checking
 * - Support for both scalar and array operations
 * - Efficient data access through linear memory layout
 * - Template-based type safety and constraints
 * - Move semantics for efficient memory handling
 * - Flexible shape and dimension management
 * - Exception handling for dimension mismatches
 * - Comprehensive operator overloading
 */

#ifndef NDARRAY_H
#define NDARRAY_H

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#ifdef CPPMATRIX_USE_OPENMP
#include <omp.h>
#endif

namespace cppmatrix {
    template<typename T = float>
        requires std::is_floating_point_v<T>
    class NDArray {
    public:
        template<typename U>
            requires std::is_floating_point_v<U>
        friend class NDArray;

        template<typename T1, typename T2>
        friend NDArray<T1> &operator+=(NDArray<T1> &left, const NDArray<T2> &right);

        template<typename T1, typename T2>
        friend NDArray<T1> &operator-=(NDArray<T1> &left, const NDArray<T2> &right);

        template<typename U1, typename U2>
            requires std::is_floating_point_v<U1> && std::is_floating_point_v<U2>
        friend bool operator==(const NDArray<U1> &left, const NDArray<U2> &right);

        NDArray() = default;

        template<uint64_t ndim>
        explicit NDArray (

            
        const uint64_t (&shape)[ndim]
        )
 {
            this->_allocate(ndim, shape);
        }

        template<uint64_t ndim, typename U>
            requires std::is_floating_point_v<U>
        NDArray (

            
        const uint64_t (&shape)[ndim], U &value
        )
 {
            this->_allocate(ndim, shape);
            std::fill(this->_data, this->_data + this->N(), T(value));
        }

        NDArray(NDArray<T> &&right) noexcept {
            this->_ndim = right._ndim;
            this->_shape = right._shape;
            this->_data = right._data;

            right._shape = nullptr;
            right._data = nullptr;
        }

        NDArray(const NDArray<T> &right) {
            this->_allocate(right._ndim, right._shape);

            std::copy(right._shape, right._shape + right._ndim, this->_shape);
            std::copy(right._data, right._data + right.N(), this->_data);
        }

        virtual ~NDArray() {
            delete[] this->_shape;
            delete[] this->_data;
        }

        template<uint64_t ndim>
        T &operator()(uint64_t (&index)[ndim]) {
            uint64_t _index = 0;
            for (uint64_t i = 0; i < this->_ndim; i++) {
                uint64_t _product = 1;
                for (uint64_t j = i + 1; j < this->_ndim; j++) {
                    _product *= this->_shape[j];
                }
                _index += index[i] * _product;
            }

            return _data[_index];
        }

        template<uint64_t ndim>
        const T &operator()(uint64_t (&index)[ndim]) const {
            uint64_t _index = 0;
            for (uint64_t i = 0; i < this->_ndim; i++) {
                uint64_t _product = 1;
                for (uint64_t j = i + 1; j < this->_ndim; j++) {
                    _product *= this->_shape[j];
                }
                _index += index[i] * _product;
            }

            return _data[_index];
        }

        NDArray<T> &operator=(const NDArray<T> &right) {
            if (this != &right) {
                this->_allocate(right._ndim, right._shape);

                std::copy(right._shape, right._shape + right._ndim, this->_shape);
                std::copy(right._data, right._data + right.N(), this->_data);
            }

            return *this;
        }

        NDArray<T> &operator=(NDArray<T> &&right) noexcept {
            if (this != &right) {
                delete[] this->_shape;
                delete[] this->_data;

                this->_ndim = right._ndim;
                this->_shape = right._shape;
                this->_data = right._data;

                right._shape = nullptr;
                right._data = nullptr;
            }

            return *this;
        }

        template<typename T2>
        NDArray<T> &operator*=(const T2 &right) {
#ifdef CPPMATRIX_USE_OPENMP
#pragma omp parallel for
#endif
            for (uint64_t idx = 0; idx < this->N(); ++idx)
                this->_data[idx] = std::multiplies<T>()(this->_data[idx], T(right));
            return *this;
        }

        template<typename T2>
        NDArray<T> operator*(const T2 &right) const {
            NDArray<T> result(*this);
            result *= right;
            return result;
        }

        template<typename T2>
        NDArray<T> &operator/=(const T2 &right) {
            if (right == T2(0))
                throw std::runtime_error("Division by zero.");
#ifdef CPPMATRIX_USE_OPENMP
#pragma omp parallel for
#endif
            for (uint64_t idx = 0; idx < this->N(); ++idx)
                this->_data[idx] = std::divides<T>()(this->_data[idx], T(right));
            return *this;
        }

        template<typename T2>
        NDArray<T> operator/(const T2 &right) const {
            NDArray<T> result(*this);
            result /= right;
            return result;
        }

        [[nodiscard]] virtual uint64_t N() const {
            uint64_t n = std::accumulate(this->_shape, this->_shape + this->_ndim, 1,
                                         std::multiplies());
            return n;
        }

        [[nodiscard]] uint64_t ndim() const { return this->_ndim; }

        [[nodiscard]] uint64_t *shape() const { return this->_shape; }

        T *data() const { return this->_data; }

        template<typename U>
        bool check_sizes(const NDArray<U> &right) const {
            if ((this->_ndim != right._ndim) ||
                (!std::equal(this->_shape, this->_shape + this->_ndim, right.shape())))
                return false;

            return true;
        }

    private:
        uint64_t _ndim = 0;
        uint64_t *_shape = nullptr;
        T *_data = nullptr;

        void _allocate(const uint64_t &ndim, const uint64_t *shape) {
            delete[] this->_shape;
            delete[] this->_data;

            this->_ndim = ndim;
            this->_shape = new uint64_t[ndim];

            std::copy(shape, shape + ndim, this->_shape);
            this->_data = new T[this->N()];
        }
    };

    template<typename T1, typename T2>
    NDArray<T1> &operator+=(NDArray<T1> &left, const NDArray<T2> &right) {
        if (left.check_sizes(right)) {
            std::transform(left._data, left._data + left.N(), right._data, left._data,
                           std::plus<T1>());
            return left;
        }

        throw std::runtime_error("Size mismatch for operator+=().");
    }

    template<typename T1, typename T2>
    NDArray<T1> operator+(const NDArray<T1> &left, const NDArray<T2> &right) {
        auto result = NDArray(left);
        operator+=(result, right);

        return result;
    }

    template<typename T1, typename T2>
    NDArray<T1> &operator+=(NDArray<T1> &left, const T2 &right) {
        std::transform(left.data(), left.data() + left.N(), left.data(),
                       [right](T1 element) { return element + right; });
        return left;
    }

    template<typename T1, typename T2>
    NDArray<T1> operator+(const NDArray<T1> &left, const T2 &right) {
        NDArray<T1> result(left);
        operator+=(result, right);

        return result;
    }

    template<typename T1, typename T2>
    NDArray<T2> operator+(const T1 &left, const NDArray<T2> &right) {
        NDArray<T2> result(right);
        operator+=(result, T2(left));

        return result;
    }

    template<typename T1, typename T2>
    NDArray<T1> &operator-=(NDArray<T1> &left, const NDArray<T2> &right) {
        if (left.check_sizes(right)) {
            std::transform(left._data, left._data + left.N(), right._data, left._data,
                           std::minus<T1>());
            return left;
        }

        throw std::runtime_error("Size mismatch for operator-=().");
    }

    template<typename T1, typename T2>
    NDArray<T1> operator-(const NDArray<T1> &left, const NDArray<T2> &right) {
        NDArray<T1> result(left);
        operator-=(result, right);

        return result;
    }

    template<typename T1, typename T2>
    NDArray<T1> &operator-=(NDArray<T1> &left, const T2 &right) {
        std::transform(left.data(), left.data() + left.N(), left.data(),
                       [right](T1 element) { return element - T1(right); });
        return left;
    }

    template<typename T1, typename T2>
    NDArray<T1> operator-(const NDArray<T1> &left, const T2 &right) {
        NDArray<T1> result(left);
        operator-=(result, right);

        return result;
    }

    template<typename T1, typename T2>
    NDArray<T2> operator-(const T1 &left, NDArray<T2> &right) {
        auto result = NDArray(right) * T2(-1.0);
        operator+=(result, left);

        return result;
    }

    template<typename T1, typename T2>
    NDArray<T2> operator*(const T1 &left, const NDArray<T2> &right) {
        return right * T2(left);
    }

    template<typename T1, typename T2>
        requires std::is_floating_point_v<T1> && std::is_floating_point_v<T2>
    bool operator==(const NDArray<T1> &left, const NDArray<T2> &right) {
        if (!left.check_sizes(right))
            return false;
        for (uint64_t i = 0; i < left.N(); i++)
            if (std::abs(left._data[i] - T1(right._data[i])) > std::numeric_limits<T1>::epsilon())
                return false;
        return true;
    }

    template<typename T1, typename T2>
        requires std::is_floating_point_v<T1> && std::is_floating_point_v<T2>
    bool operator!=(const NDArray<T1> &left, const NDArray<T2> &right) {
        return !(left == right);
    }
}

#endif