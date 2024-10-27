// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/layernorm2d.hpp"
#include <string>

template <typename DataType>
struct LayerNormTypeConfig;

template <>
struct LayerNormTypeConfig<ck_tile::half_t>
{
    using XDataType       = ck_tile::half_t;
    using YDataType       = ck_tile::half_t;
    using GammaDataType   = ck_tile::half_t;
    using BetaDataType    = ck_tile::half_t;
    using MeanDataType    = ck_tile::half_t;
    using InvStdDataType  = ck_tile::half_t;
    using ComputeDataType = float;
};

template <>
struct LayerNormTypeConfig<ck_tile::bf16_t>
{
    using XDataType       = ck_tile::bf16_t;
    using YDataType       = ck_tile::bf16_t;
    using GammaDataType   = ck_tile::bf16_t;
    using BetaDataType    = ck_tile::bf16_t;
    using MeanDataType    = ck_tile::bf16_t;
    using InvStdDataType  = ck_tile::bf16_t;
    using ComputeDataType = float;
};

// runtime args
struct layernorm2d_bwd_args : public ck_tile::Layernorm2dBwdGammaBetaHostArgs
{
};

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <typename DataType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          bool kPadN_>
struct layernorm2d_bwd_traits_
{
    using DataType = ck_tile::remove_cvref_t<DataType_>;

    static constexpr bool is_warp_per_row = ThreadPerBlock_N_ <= warpSize;
    static_assert((ThreadPerBlock_M_ * ThreadPerBlock_N_) % warpSize == 0);
    static constexpr ck_tile::index_t total_warps =
        (ThreadPerBlock_M_ * ThreadPerBlock_N_) / warpSize;

    // num of warps along m
    static constexpr ck_tile::index_t BlockWarps_M = []() {
        if constexpr(is_warp_per_row)
        {
            static_assert(warpSize % ThreadPerBlock_N_ == 0);
            return total_warps * (warpSize / ThreadPerBlock_N_);
        }
        else
        {
            // static_assert(warpSize % ThreadPerBlock_M_ == 0);
            return total_warps / (ThreadPerBlock_N_ / warpSize);
        }
    }();

    // num of warps along n
    static constexpr ck_tile::index_t BlockWarps_N = []() {
        if constexpr(is_warp_per_row)
        {
            static_assert(warpSize % ThreadPerBlock_N_ == 0);
            return 1;
        }
        else
        {
            static_assert(ThreadPerBlock_N_ % warpSize == 0);
            return ThreadPerBlock_N_ / warpSize;
        }
    }();

    static constexpr ck_tile::index_t Repeat_M = Repeat_M_;

    static constexpr ck_tile::index_t Block_M = Repeat_M_ * ThreadPerBlock_M_;
    static constexpr ck_tile::index_t Block_N = ThreadPerBlock_N_;

    static constexpr ck_tile::index_t Warp_M = ThreadPerBlock_M_ / BlockWarps_M;
    static constexpr ck_tile::index_t Warp_N = ThreadPerBlock_N_ / BlockWarps_N;

    using BlockTile  = ck_tile::sequence<Block_M, Block_N>;
    using BlockWarps = ck_tile::sequence<BlockWarps_M, BlockWarps_N>;
    using WarpTile   = ck_tile::sequence<Warp_M, Warp_N>;
    using Vector     = ck_tile::sequence<1, 1>;

    using Shape = ck_tile::Layernorm2dShape<BlockTile, BlockWarps, WarpTile, Vector>;

    static constexpr bool kPadN           = kPadN_;
};

template <typename Traits_>
float layernorm2d_bwd_(const ck_tile::stream_config& s, layernorm2d_bwd_args a);

// This is the public API, will be generated by script
struct layernorm2d_bwd_traits
{
    std::string data_type;
};

float layernorm2d_bwd(layernorm2d_bwd_traits, layernorm2d_bwd_args, const ck_tile::stream_config&);
