#pragma once
#include <torch/torch.h>
#include "LossMetrics.h"

namespace Loss {

struct TripletLossImpl : torch::nn::Module {
    explicit TripletLossImpl(double margin = 0.3) : margin_(margin) {}
    LossMetrics forward(const torch::Tensor& embeddings, const torch::Tensor& labels);
    LossMetrics forward_online_hard(const torch::Tensor& embeddings, const torch::Tensor& labels);
    LossMetrics forward_batch_hard(const torch::Tensor& embeddings, const torch::Tensor& labels);
    double margin_;
};

using TripletLoss = torch::nn::ModuleHolder<TripletLossImpl>;

} // namespace loss
