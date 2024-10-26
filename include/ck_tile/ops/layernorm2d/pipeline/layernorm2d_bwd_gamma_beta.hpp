// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_default_policy.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename Problem_, typename Policy_ = Layernorm2dBwdGammaBetaPipelineDefaultPolicy>
struct Layernorm2dBwdGammaBetaPipeline
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using GammaDataType   = ck_tile::remove_cvref_t<typename Problem::GammaDataType>;
    using BetaDataType    = ck_tile::remove_cvref_t<typename Problem::BetaDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using MeanDataType    = ck_tile::remove_cvref_t<typename Problem::MeanDataType>;
    using InvStdDataType  = ck_tile::remove_cvref_t<typename Problem::InvStdDataType>;

    static constexpr bool kPadM              = false; // TODO - BlockLayernorm2dBwdGammaBetaProblem::kPadM
    static constexpr bool kPadN              = Problem::kPadN;

    static constexpr const char* name = []() {
        return "bwd_gamma_beta"
    }();

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename DYWindow,
              typename MeanWindow,
              typename InvStdWindow,
              typename DGammaWindow,
              typename DBetaWindow>
    CK_TILE_DEVICE auto operator()(const DYWindow& dy_window_,
                                   const MeanWindow& mean_window_,
                                   const InvStdWindow& inv_std_window_,
                                   DGammaWindow& gamma_window_,
                                   DBetaWindow& beta_window_,
                                   ck_tile::index_t row_size,
                                   void* smem) const
    {
        const auto dy_window =
            make_tile_window(dy_window_, Policy::template MakeDyBlockTileDistribution<Problem>());
        const auto mean_window = make_tile_window(
            mean_window_, Policy::template MakeMeanBlockTileDistribution<Problem>());
        // const auto gamma_window = make_tile_window(
        //     gamma_window_, Policy::template MakeGammaBetaBlockTileDistribution<Problem>());
        // const auto beta_window = make_tile_window(
        //     beta_window_, Policy::template MakeGammaBetaBlockTileDistribution<Problem>());

        const auto dy  = load_tile(dy_window);
        const auto mean = load_tile(mean_window);
        

        // layernorm computation
        // auto y = make_static_distributed_tensor<YDataType>(x.get_tile_distribution());
        // sweep_tile(y, [&, mean_ = mean](auto idx) {
        //     constexpr auto i_idx = make_tuple(idx[number<0>{}]);
        //     constexpr auto j_idx = make_tuple(idx[number<1>{}]);

        //     const auto gamma_ = type_convert<ComputeDataType>(gamma[j_idx]);
        //     const auto beta_  = type_convert<ComputeDataType>(beta[j_idx]);

        //     const auto x_ = type_convert<ComputeDataType>(x[idx]);
        //     auto y_       = (x_ - mean_[i_idx]) * inv_std[i_idx] * gamma_ + beta_;

        //     y(idx) = type_convert<YDataType>(y_);
        // });
        // store_tile(y_window, y);
    }
};
} // namespace ck_tile
