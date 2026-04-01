# Preamble
cmake_minimum_required(VERSION 4.2.1)

project(cppmatrix LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Versioning scheme
set(VERSION_MAJOR 0)
set(VERSION_MINOR 0)
set(VERSION_FEATURE 2)
set(VERSION_PATCH 0)
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}")
set(VERSION "${VERSION}.${VERSION_FEATURE}.${VERSION_PATCH}")


# Prefer Release by default and allow LTO/native tuning
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
option(CPPMATRIX_ENABLE_LTO "Enable IPO/LTO when available" ON)
option(CPPMATRIX_NATIVE_OPTIMIZATIONS "Enable -march=native on Release builds" ON)
option(CPPMATRIX_USE_ACCELERATE "Prefer Apple Accelerate over OpenBLAS when available" OFF)
option(CPPMATRIX_VECTORIZE_REPORT "Enable clang vectorization diagnostics" OFF)
option(CPPMATRIX_USE_MEMORY_POOL "Enable thread-local memory pooling for small matrices" ON)
option(CPPMATRIX_FAST_MATH "Enable -ffast-math for aggressive floating-point optimizations" OFF)

if(CPPMATRIX_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "IPO/LTO not supported: ${ipo_output}")
    endif()
endif()

if(CPPMATRIX_NATIVE_OPTIMIZATIONS)
    add_compile_options(
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Release>:-march=native>
            $<$<CONFIG:Release>:-DNDEBUG>
            $<$<AND:$<CONFIG:Release>,$<BOOL:${CPPMATRIX_VECTORIZE_REPORT}>>:-Rpass=loop-vectorize>
            $<$<AND:$<CONFIG:Release>,$<BOOL:${CPPMATRIX_VECTORIZE_REPORT}>>:-Rpass-missed=loop-vectorize>
    )
else()
    add_compile_options(
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Release>:-DNDEBUG>
            $<$<AND:$<CONFIG:Release>,$<BOOL:${CPPMATRIX_VECTORIZE_REPORT}>>:-Rpass=loop-vectorize>
            $<$<AND:$<CONFIG:Release>,$<BOOL:${CPPMATRIX_VECTORIZE_REPORT}>>:-Rpass-missed=loop-vectorize>
    )
endif()

# Additional performance optimizations
if(CPPMATRIX_FAST_MATH)
    add_compile_options(
            $<$<CONFIG:Release>:-ffast-math>
            $<$<CONFIG:Release>:-fno-math-errno>
    )
    message(STATUS "Fast math enabled - may reduce numerical precision")
endif()

# Additional vectorization hints
add_compile_options(
        $<$<CONFIG:Release>:-funroll-loops>
        $<$<CONFIG:Release>:-fvectorize>
        $<$<CONFIG:Release>:-fslp-vectorize>
)


# Build instructions for different platforms
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
    if(CPPMATRIX_USE_ACCELERATE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if(ACCELERATE_FRAMEWORK)
            set(BLAS_LIBRARIES ${ACCELERATE_FRAMEWORK})
            set(LAPACK_LIBRARIES "")
            message(STATUS "Using Accelerate framework for BLAS")
        endif()
    endif()
    if(NOT ACCELERATE_FRAMEWORK)
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
    endif()
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

# Enable memory pooling if requested
if(CPPMATRIX_USE_MEMORY_POOL)
    target_compile_definitions(run_tests PRIVATE CPPMATRIX_USE_MEMORY_POOL=1)
    message(STATUS "Memory pooling enabled for small matrices")
endif()
