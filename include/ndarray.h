#ifndef NDARRAY_H
#define NDARRAY_H

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <stdexcept>


namespace cppmatrix{

    template<typename T = float>
        requires std::is_floating_point_v<T>
    class NDArray{

    public:

        template<typename U>
            requires std::is_floating_point_v<U>
        friend class NDArray;

        template<typename T1, typename T2>
        friend NDArray<T1>& operator+=(NDArray<T1> &left, const NDArray<T2> &right);

        template<typename T1, typename T2>
        friend NDArray<T1>& operator-=(NDArray<T1> &left, const NDArray<T2> &right);

        NDArray() = default;

        template<uint64_t ndim>
        explicit NDArray(const uint64_t (&shape)[ndim]){
            this->_allocate(ndim, shape);
        }

        template<uint64_t ndim, typename U>
            requires std::is_floating_point_v<U>
        NDArray(const uint64_t (&shape)[ndim], U &value){
            this->_allocate(ndim, shape);
            std::fill(this->_data, this->_data + this->N(), T(value));
        }

        NDArray(NDArray<T> &&right) noexcept{
            this->_ndim = right._ndim;
            this->_shape = right._shape;
            this->_data = right._data;

            right._shape = nullptr;
            right._data = nullptr;
        }

        NDArray(const NDArray<T> &right){
            this->_allocate(right._ndim, right._shape);

            std::copy(right._shape, right._shape + right._ndim, this->_shape);
            std::copy(right._data, right._data + right.N(), this->_data);
        }

        virtual ~NDArray(){
            delete[] this->_shape;
            delete[] this->_data;
        }

        inline uint64_t N() const{
            uint64_t n = std::accumulate(this->_shape, this->_shape + this->_ndim, 1, std::multiplies());
            return n;
        }

        inline uint64_t ndim() const{
            return this->_ndim;
        }

        inline uint64_t* shape() const{
            return this->_shape;
        }

        inline T* data() const{
            return this->_data;
        }

        template<uint64_t ndim>
        T& operator()(const uint64_t (&index)[ndim]){
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

        template<typename U>
        inline bool check_sizes(const NDArray<U> &right) const{
            if((this->_ndim != right._ndim) || (!std::equal(this->_shape, this->_shape + this->_ndim, right.shape())))
                return false;

            return true;
        }

        NDArray<T>& operator=(const NDArray<T> &right){
            if(this != &right){
                this->_allocate(right._ndim, right._shape);

                std::copy(right._shape, right._shape + right._ndim, this->_shape);
                std::copy(right._data, right._data + right.N(), this->_data);
            }

            return *this;
        }

        NDArray<T>& operator+=(const NDArray<T> &right){
            if(this->check_sizes(right)){
                std::transform(this->_data, this->_data + this->N(), right._data, this->_data, std::plus<T>());
                return *this;
            }

            throw std::runtime_error("Size mismatch for operator+=");
        }

        NDArray<T> operator+(const NDArray<T> &right){
            return NDArray<T>(*this) += right;
        }

        NDArray<T>& operator-=(const NDArray<T> &right){
            if(this->check_sizes(right)){
                std::transform(this->_data, this->_data + this->N(), right._data, this->_data, std::minus<T>());
                return *this;
            }

            throw std::runtime_error("Size mismatch for operator-=");
        }

        NDArray<T> operator-(const NDArray<T> &right){
            return NDArray<T>(*this) -= right;
        }

    private:
        uint64_t _ndim = 0;
        uint64_t* _shape = nullptr;
        T* _data = nullptr;

        void _allocate(const uint64_t &ndim, const uint64_t* shape){
            if(this->_shape != nullptr)
                delete[] this->_shape;

            if(this->_data != nullptr)
                delete[] this->_data;

            this->_ndim = ndim;
            this->_shape = new uint64_t[ndim];

            std::copy(shape, shape + ndim, this->_shape);
            this->_data = new T[this->N()];
        }

    };

    template<typename T1, typename T2>
    NDArray<T1>& operator+=(NDArray<T1> &left, const NDArray<T2> &right){
        if(left.check_sizes(right)){
            std::transform(left._data, left._data + left.N(), right._data, left._data, std::plus<T1>());
            return left;
        }

        throw std::runtime_error("Size mismatch for operator+=");
    }

    template<typename T1, typename T2>
    NDArray<T1> operator+(const NDArray<T1> &left, const NDArray<T2> &right){
        auto result = NDArray(left);
        operator+=(result, right);

        return result;
    }

    template<typename T1, typename T2>
    NDArray<T1>& operator-=(NDArray<T1> &left, const NDArray<T2> &right){
        if(left.check_sizes(right)){
            std::transform(left._data, left._data + left.N(), right._data, left._data, std::minus<T1>());
            return left;
        }

        throw std::runtime_error("Size mismatch for operator-=");
    }

    template<typename T1, typename T2>
    NDArray<T1> operator-(const NDArray<T1> &left, const NDArray<T2> &right){
        auto result = NDArray(left);
        operator-=(result, right);

        return result;
    }

}

#endif // NDARRAY_H
