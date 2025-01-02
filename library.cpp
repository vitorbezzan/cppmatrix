/* library.cpp - main file to build libcppmatrix */
#include "include/ndarray.h"
#include "include/matrix.h"


template class cppmatrix::NDArray<float>;
template class cppmatrix::NDArray<double>;
template class cppmatrix::Matrix<float>;
template class cppmatrix::Matrix<double>;
template class cppmatrix::ColumnVector<float>;
template class cppmatrix::ColumnVector<double>;
template class cppmatrix::RowVector<float>;
template class cppmatrix::RowVector<double>;
template class cppmatrix::Vector<float>;
template class cppmatrix::Vector<double>;
