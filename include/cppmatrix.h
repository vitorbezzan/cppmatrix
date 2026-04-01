/**
 * @file cppmatrix.h
 * @brief Main header file providing access to all cppmatrix library components.
 * 
 * This module serves as the primary include point for the cppmatrix library and provides:
 * - Access to all matrix and vector operations
 * - Mathematical function implementations and utilities
 * - Numerical integration (RealFunction templates, same functor style as Newton) and root finding
 * - Array initialization and filling utilities
 * - N-dimensional array support
 * - Expression templates for lazy evaluation (NEW)
 * - Memory pooling for small matrices (NEW)
 * - Batch operations for parallel processing (NEW)
 * - Parallel multi-start root finding (NEW)
 */

#ifndef CPPMATRIX_H
#define CPPMATRIX_H

#include "fillers.h"
#include "function.h"
#include "integration.h"
#include "matrix.h"
#include "ndarray.h"
#include "newton.h"
#include "ode.h"
#include "process.h"
#include "vector.h"

#include "expression_templates.h"
#include "memory_pool.h"
#include "batch_operations.h"

#endif