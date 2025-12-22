cmake_minimum_required(VERSION 3.31.4)

project(cppmatrix LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VERSION_MAJOR 0)
set(VERSION_MINOR 0)
set(VERSION_FEATURE 1)
set(VERSION_PATCH 0)
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}")
set(VERSION "${VERSION}.${VERSION_FEATURE}.${VERSION_PATCH}")

if (APPLE)
    execute_process(
            COMMAND brew --prefix
            RESULT_VARIABLE HOMEBREW
            OUTPUT_VARIABLE HOMEBREW_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS Homebrew Prefix: ${HOMEBREW_PREFIX})

    set(CMAKE_PREFIX_PATH
            "${HOMEBREW_PREFIX}"
            "${HOMEBREW_PREFIX}/opt/openblas"
            "${HOMEBREW_PREFIX}/opt/lapack"
            "${HOMEBREW_PREFIX}/opt/libomp"
            "/usr/local"
    )

    list(TRANSFORM CMAKE_PREFIX_PATH APPEND "/include"
            OUTPUT_VARIABLE CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES)
    set(CMAKE_C_STANDARD_INCLUDE_DIRECTORIES "${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}")

    message(INCLUDE: "${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}")

    include_directories(
            "${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}"
    )

    set(CMAKE_FIND_FRAMEWORK NEVER)
    set(CMAKE_FIND_APPBUNDLE NEVER)

    set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)
    set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)

    find_package(GTest REQUIRED)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    find_package(OpenMP REQUIRED)

elseif (UNIX)

    find_package(GTest REQUIRED)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    find_package(OpenMP REQUIRED)

endif ()

add_executable(run_tests tests.cpp)
target_link_libraries(run_tests GTest::gtest GTest::gtest_main ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX)
target_compile_definitions(run_tests PRIVATE CPPMATRIX_USE_OPENMP=1)
