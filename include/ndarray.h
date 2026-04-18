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
#include <span>
#include <type_traits>
#include <new>
#include <cstddef>
#include "detail/safe_math.h"
#ifdef CPPMATRIX_USE_OPENMP
#include <omp.h>
#endif

#ifndef CPPMATRIX_MAX_NDIM
#define CPPMATRIX_MAX_NDIM 16
#endif

#ifndef CPPMATRIX_MAX_BYTES
// Default cap intended for untrusted inputs; override in build flags if needed.
#define CPPMATRIX_MAX_BYTES (static_cast<std::size_t>(1) << 30) // 1 GiB
#endif

#ifdef CPPMATRIX_ENABLE_BOUNDS_CHECKS
#define CPPMATRIX_DETAIL_BOUNDS_CHECKS 1
#else
#define CPPMATRIX_DETAIL_BOUNDS_CHECKS 0
#endif

#ifndef CPPMATRIX_RESTRICT
#if defined(__clang__) || defined(__GNUC__)
#define CPPMATRIX_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define CPPMATRIX_RESTRICT __restrict
#else
#define CPPMATRIX_RESTRICT
#endif
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
        friend NDArray<T1>& operator+=(NDArray<T1>& left, const NDArray<T2>& right);

        template<typename T1, typename T2>
        friend NDArray<T1>& operator-=(NDArray<T1>& left, const NDArray<T2>& right);

        template<typename U1, typename U2>
        requires std::is_floating_point_v<U1> && std::is_floating_point_v<U2>
        friend bool operator==(const NDArray<U1>& left, const NDArray<U2>& right);

        NDArray() = default;

        template<uint64_t ndim>
        explicit NDArray (



            const uint64_t (&shape)[ndim]
        ) {
            this->_allocate(ndim, shape);
        }

        template<uint64_t ndim, typename U>
        requires std::is_floating_point_v<U>
        NDArray (



            const uint64_t (&shape)[ndim], U &value
        ) {
            this->_allocate(ndim, shape);
            std::fill(this->_data, this->_data + this->N(), T(value));
        }

        NDArray(NDArray<T> &&right) noexcept {
            this->_ndim = right._ndim;
            this->_shape = right._shape;
            this->_strides = right._strides;
            this->_data = right._data;

            right._shape = nullptr;
            right._strides = nullptr;
            right._data = nullptr;
        }

        NDArray(const NDArray<T>& right) {
            this->_allocate(right._ndim, right._shape);
            std::copy(right._data, right._data + right.N(), this->_data);
        }

        virtual ~NDArray() {
            delete[] this->_shape;
            delete[] this->_strides;
            if (this->_data)
                ::operator delete[](this->_data, std::align_val_t(kAlignment));
        }

        template<uint64_t ndim>
        T& operator()(uint64_t (&index)[ndim]) {
            return this->operator()(std::span<const uint64_t>(index, ndim));
        }

        template<uint64_t ndim>
        const T& operator()(uint64_t (&index)[ndim]) const {
            return this->operator()(std::span<const uint64_t>(index, ndim));
        }

        T& operator()(std::span<const uint64_t> index) {
            // Reuse const logic for index validation/linearization.
            return const_cast<T&>(static_cast<const NDArray&>(*this).operator()(index));
        }

        const T& operator()(std::span<const uint64_t> index) const {
            if (CPPMATRIX_DETAIL_BOUNDS_CHECKS) {
                if (index.size() != this->_ndim) {
                    throw std::out_of_range("NDArray: index rank mismatch");
                }
                for (uint64_t i = 0; i < this->_ndim; ++i) {
                    if (index[static_cast<std::size_t>(i)] >= this->_shape[i]) {
                        throw std::out_of_range("NDArray: index out of bounds");
                    }
                }
            } else {
                // Even in unchecked mode, prevent the specific UB where index is shorter than _ndim.
                if (index.size() < this->_ndim) {
                    throw std::out_of_range("NDArray: index rank mismatch");
                }
            }

            uint64_t linear = 0;
            for (uint64_t i = 0; i < this->_ndim; ++i) {
                linear += index[static_cast<std::size_t>(i)] * this->_strides[i];
            }
            return this->_data[linear];
        }

        T& at(std::span<const uint64_t> index) { return this->checked_at_mut(index); }
        const T& at(std::span<const uint64_t> index) const { return this->checked_at(index); }

        template<uint64_t ndim>
        T& at(uint64_t (&index)[ndim]) {
            return this->at(std::span<const uint64_t>(index, ndim));
        }

        template<uint64_t ndim>
        const T& at(uint64_t (&index)[ndim]) const {
            return this->at(std::span<const uint64_t>(index, ndim));
        }

        NDArray<T>& operator=(const NDArray<T>& right) {
            if (this != &right) {
                this->_allocate(right._ndim, right._shape);
                std::copy(right._data, right._data + right.N(), this->_data);
            }

            return *this;
        }

        NDArray<T>& operator=(NDArray<T> &&right) noexcept {
            if (this != &right) {
                delete[] this->_shape;
                delete[] this->_strides;
                if (this->_data)
                    ::operator delete[](this->_data, std::align_val_t(kAlignment));

                this->_ndim = right._ndim;
                this->_shape = right._shape;
                this->_strides = right._strides;
                this->_data = right._data;

                right._shape = nullptr;
                right._strides = nullptr;
                right._data = nullptr;
            }

            return *this;
        }

        template<typename T2>
        NDArray<T>& operator*=(const T2 &right) {
#ifdef CPPMATRIX_USE_OPENMP
            #pragma omp parallel for simd
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
        NDArray<T>& operator/=(const T2 &right) {
            if (right == T2(0))
                throw std::runtime_error("Division by zero.");
#ifdef CPPMATRIX_USE_OPENMP
            #pragma omp parallel for simd
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

        [[nodiscard]] std::span<const uint64_t> shape() const {
            return std::span<const uint64_t>(this->_shape, static_cast<std::size_t>(this->_ndim));
        }

        T* data() { return this->_data; }
        const T* data() const { return this->_data; }

        template<typename U>
        bool check_sizes(const NDArray<U>& right) const {
            if ((this->_ndim != right._ndim) ||
                    (!std::equal(this->_shape, this->_shape + this->_ndim, right.shape().begin())))
                return false;

            return true;
        }

    private:
        static constexpr std::size_t kAlignment = 64;
        uint64_t _ndim = 0;
        uint64_t* _shape = nullptr;
        uint64_t* _strides = nullptr;
        T* CPPMATRIX_RESTRICT _data = nullptr;

        T& checked_at_mut(std::span<const uint64_t> index) {
            return const_cast<T&>(static_cast<const NDArray&>(*this).checked_at(index));
        }

        const T& checked_at(std::span<const uint64_t> index) const {
            if (index.size() != this->_ndim) {
                throw std::out_of_range("NDArray: index rank mismatch");
            }
            for (uint64_t i = 0; i < this->_ndim; ++i) {
                if (index[static_cast<std::size_t>(i)] >= this->_shape[i]) {
                    throw std::out_of_range("NDArray: index out of bounds");
                }
            }

            uint64_t linear = 0;
            for (uint64_t i = 0; i < this->_ndim; ++i) {
                linear += index[static_cast<std::size_t>(i)] * this->_strides[i];
            }
            return this->_data[linear];
        }

        void _allocate(const uint64_t& ndim, const uint64_t* shape) {
            delete[] this->_shape;
            delete[] this->_strides;
            if (this->_data)
                ::operator delete[](this->_data, std::align_val_t(kAlignment));

            if (ndim > CPPMATRIX_MAX_NDIM) {
                throw std::length_error("NDArray: ndim exceeds CPPMATRIX_MAX_NDIM");
            }

            this->_ndim = ndim;
            this->_shape = new uint64_t[ndim];
            this->_strides = new uint64_t[ndim];

            std::copy(shape, shape + ndim, this->_shape);

            // Validate shape values fit in size_t for allocation math.
            for (uint64_t i = 0; i < ndim; ++i) {
                if (this->_shape[i] > static_cast<uint64_t>((std::numeric_limits<std::size_t>::max)())) {
                    throw std::length_error("NDArray: shape element too large");
                }
            }

            if (ndim > 0) {
                this->_strides[ndim - 1] = 1;
                for (int64_t i = ndim - 2; i >= 0; --i) {
                    uint64_t stride = 0;
                    if (detail::mul_overflow_u64(this->_strides[i + 1], this->_shape[i + 1], stride)) {
                        throw std::overflow_error("NDArray: stride overflow");
                    }
                    this->_strides[i] = stride;
                }
            }

            // Compute element count with overflow protection.
            std::size_t n_elems = 1;
            for (uint64_t i = 0; i < ndim; ++i) {
                std::size_t tmp = 0;
                if (detail::mul_overflow_size(n_elems, static_cast<std::size_t>(this->_shape[i]), tmp)) {
                    throw std::overflow_error("NDArray: element count overflow");
                }
                n_elems = tmp;
            }

            std::size_t n_bytes = 0;
            if (detail::mul_overflow_size(n_elems, sizeof(T), n_bytes)) {
                throw std::overflow_error("NDArray: byte size overflow");
            }
            if (n_bytes > CPPMATRIX_MAX_BYTES) {
                throw std::length_error("NDArray: allocation exceeds CPPMATRIX_MAX_BYTES");
            }

            this->_data = static_cast<T*>(
                              ::operator new[](n_bytes, std::align_val_t(kAlignment))
                          );
        }
    };

    template<typename T1, typename T2>
    NDArray<T1>& operator+=(NDArray<T1>& left, const NDArray<T2>& right) {
        if (left.check_sizes(right)) {
#ifdef CPPMATRIX_USE_OPENMP
            #pragma omp parallel for simd
#endif
            for (uint64_t idx = 0; idx < left.N(); ++idx)
                left._data[idx] = std::plus<T1>()(left._data[idx], T1(right._data[idx]));
            return left;
        }

        throw std::runtime_error("Size mismatch for operator+=().");
    }

    template<typename T1, typename T2>
    NDArray<T1> operator+(const NDArray<T1>& left, const NDArray<T2>& right) {
        auto result = NDArray(left);
        operator+=(result, right);

        return result;
    }

    template<typename T1, typename T2>
    NDArray<T1>& operator+=(NDArray<T1>& left, const T2 &right) {
#ifdef CPPMATRIX_USE_OPENMP
        #pragma omp parallel for simd
#endif
        for (uint64_t idx = 0; idx < left.N(); ++idx)
            left._data[idx] = std::plus<T1>()(left._data[idx], T1(right));
        return left;
    }

    template<typename T1, typename T2>
    NDArray<T1> operator+(const NDArray<T1>& left, const T2 &right) {
        NDArray<T1> result(left);
        operator+=(result, right);

        return result;
    }

    template<typename T1, typename T2>
    NDArray<T2> operator+(const T1 &left, const NDArray<T2>& right) {
        NDArray<T2> result(right);
        operator+=(result, T2(left));

        return result;
    }

    template<typename T1, typename T2>
    NDArray<T1>& operator-=(NDArray<T1>& left, const NDArray<T2>& right) {
        if (left.check_sizes(right)) {
#ifdef CPPMATRIX_USE_OPENMP
            #pragma omp parallel for simd
#endif
            for (uint64_t idx = 0; idx < left.N(); ++idx)
                left._data[idx] = std::minus<T1>()(left._data[idx], T1(right._data[idx]));
            return left;
        }

        throw std::runtime_error("Size mismatch for operator-=().");
    }

    template<typename T1, typename T2>
    NDArray<T1> operator-(const NDArray<T1>& left, const NDArray<T2>& right) {
        NDArray<T1> result(left);
        operator-=(result, right);

        return result;
    }

    template<typename T1, typename T2>
    NDArray<T1>& operator-=(NDArray<T1>& left, const T2 &right) {
#ifdef CPPMATRIX_USE_OPENMP
        #pragma omp parallel for simd
#endif
        for (uint64_t idx = 0; idx < left.N(); ++idx)
            left._data[idx] = std::minus<T1>()(left._data[idx], T1(right));
        return left;
    }

    template<typename T1, typename T2>
    NDArray<T1> operator-(const NDArray<T1>& left, const T2 &right) {
        NDArray<T1> result(left);
        operator-=(result, right);

        return result;
    }

    template<typename T1, typename T2>
    NDArray<T2> operator-(const T1 &left, NDArray<T2>& right) {
        auto result = NDArray(right) * T2(-1.0);
        operator+=(result, left);

        return result;
    }

    template<typename T1, typename T2>
    NDArray<T2> operator*(const T1 &left, const NDArray<T2>& right) {
        return right * T2(left);
    }

    template<typename T1, typename T2>
    requires std::is_floating_point_v<T1> && std::is_floating_point_v<T2>
    bool operator==(const NDArray<T1>& left, const NDArray<T2>& right) {
        if (!left.check_sizes(right))
            return false;

        const T1 epsilon = std::numeric_limits<T1>::epsilon();
        const uint64_t n = left.N();

        for (uint64_t i = 0; i < n; i++) {
            if (std::abs(left._data[i] - T1(right._data[i])) > epsilon)
                return false;
        }
        return true;
    }

    template<typename T1, typename T2>
    requires std::is_floating_point_v<T1> && std::is_floating_point_v<T2>
    bool operator!=(const NDArray<T1>& left, const NDArray<T2>& right) {
        return !(left == right);
    }
}

#endif