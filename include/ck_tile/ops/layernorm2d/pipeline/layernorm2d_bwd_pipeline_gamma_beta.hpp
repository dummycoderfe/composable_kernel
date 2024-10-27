// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/layernorm2d/pipeline/layernorm2d_bwd_pipeline_default_policy.hpp"
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
        return "bwd_gamma_beta";
    }();

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }
    // template <typename DumpTensor_>
    // CK_TILE_DEVICE void dump(const DumpTensor_& x) const 
    // {
    //     constexpr auto I0 = number<0>{};
    //     constexpr auto I1 = number<1>{};

    //     constexpr auto spans = DumpTensor_::get_distributed_spans();

    //     sweep_tile_span(spans[I1], [&](auto i1) {
    //         sweep_tile_span(spans[I0], [&](auto i0) {
    //             constexpr auto in_dstr_idx  = make_tuple(i0, i1);
    //             auto v = ck_tile::type_convert<float>(x[in_dstr_idx]);
    //             index_t tid =
    //                 (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    //             printf("%d %f\n", tid, v);
                
    //         });
    //     });
    // }
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
        const auto dy_window = make_tile_window(dy_window_, 
            Policy::template MakeDyBlockTileDistribution<Problem>());
        const auto mean_window = make_tile_window(
            mean_window_, Policy::template MakeMeanBlockTileDistribution<Problem>());
        const auto inv_std_window = make_tile_window(
            inv_std_window_, Policy::template MakeMeanBlockTileDistribution<Problem>());
        // const auto gamma_window = make_tile_window(
        //     gamma_window_, Policy::template MakeGammaBetaBlockTileDistribution<Problem>());
        // const auto beta_window = make_tile_window(
        //     beta_window_, Policy::template MakeGammaBetaBlockTileDistribution<Problem>());

        const auto dy  = load_tile(dy_window);
        const auto mean = load_tile(mean_window);
        const auto inv_std = load_tile(inv_std_window);
        
        // auto y = make_static_distributed_tensor<YDataType>(dy.get_tile_distribution());
        sweep_tile(mean, [&](auto idx) {
            constexpr auto i_idx = make_tuple(idx[number<0>{}]);
            // constexpr auto j_idx = make_tuple(idx[number<1>{}]);

            index_t tid = (threadIdx.y * blockDim.x) + threadIdx.x;
            const auto m = type_convert<float>(mean[i_idx]);
            if(blockIdx.x == 0 && blockIdx.y == 0)
                printf("%d %f\n", tid, m);
                
        });
        // dump(dy);
        // dump(mean);
        // dump(inv_std);
        *reinterpret_cast<char *>(smem) = row_size;

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
