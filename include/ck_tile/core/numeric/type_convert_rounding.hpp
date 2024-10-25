// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/numeric/type_convert.hpp"

namespace ck_tile {

template <typename Y, typename X, int round>
struct type_convert_rounding {
    constexpr Y operator()(const X& value) const {
        return type_convert<Y>(value);
    }
};

template<int round>
struct type_convert_rounding<bfloat16_t, float, round> {
    constexpr bfloat16_t operator()(const float& value) const {
        return float_to_bf16<int_to_bf16_rounding_mode(round)>(value);
    }
};

} // namespace ck_tile
