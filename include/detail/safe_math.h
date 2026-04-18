#ifndef CPPMATRIX_DETAIL_SAFE_MATH_H
#define CPPMATRIX_DETAIL_SAFE_MATH_H

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cppmatrix::detail {
    [[nodiscard]] inline bool add_overflow_u64(uint64_t a, uint64_t b, uint64_t &out) {
        out = a + b;
        return out < a;
    }

    [[nodiscard]] inline bool mul_overflow_u64(uint64_t a, uint64_t b, uint64_t &out) {
        if (a == 0 || b == 0) {
            out = 0;
            return false;
        }
        if (a > (std::numeric_limits<uint64_t>::max)() / b) return true;
        out = a * b;
        return false;
    }

    [[nodiscard]] inline bool mul_overflow_size(std::size_t a, std::size_t b, std::size_t &out) {
        if (a == 0 || b == 0) {
            out = 0;
            return false;
        }
        if (a > (std::numeric_limits<std::size_t>::max)() / b) return true;
        out = a * b;
        return false;
    }
}

#endif
