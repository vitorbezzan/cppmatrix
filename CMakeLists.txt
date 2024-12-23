cmake_minimum_required(VERSION 3.31.3)

project(cppmatrix LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VERSION_MAJOR 0)
set(VERSION_MINOR 0)
set(VERSION_FEATURE 1)
set(VERSION_PATCH 0)
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}")
set(VERSION "${VERSION}.${VERSION_FEATURE}.${VERSION_PATCH}")


# If building on macos, needs to point to homebrew libraries (or use Accel framework)
if(APPLE)
    include_directories(
            "/opt/homebrew/opt/openblas/include"
            "/opt/homebrew/opt/lapack/include"
    )
endif ()


find_package(GTest REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_executable(run_tests tests.cpp)
target_link_libraries(run_tests GTest::gtest GTest::gtest_main ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} pthread)

add_library(cppmatrix SHARED STATIC library.cpp)
target_link_libraries(cppmatrix ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} pthread)
