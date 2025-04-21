#pragma once

namespace cute_kernels {
    constexpr bool convert_bool_to_static_bool(const bool &x) {
        if (x) {
            return true;
        } else {
            return false;
        }
    }
}  // namespace cute_kernels
