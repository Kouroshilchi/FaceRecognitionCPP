#pragma once
#include <torch/torch.h>

namespace Loss {

struct LossMetrics {
    torch::Tensor loss;
    double avg_pos_dist;
    double avg_neg_dist;
};

struct TripletLossImpl : torch::nn::Module {
    explicit TripletLossImpl(double margin = 0.3) : margin_(margin) {}
    LossMetrics forward(const torch::Tensor& embeddings, const torch::Tensor& labels);
    double margin_;
};

using TripletLoss = torch::nn::ModuleHolder<TripletLossImpl>;

} // namespace loss
