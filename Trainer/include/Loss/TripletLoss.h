#pragma once
#include <torch/torch.h>

namespace Loss {

struct TripletLossImpl : torch::nn::Module {
    explicit TripletLossImpl(double margin = 0.3) : margin_(margin) {}
    torch::Tensor forward(const torch::Tensor& embeddings, const torch::Tensor& labels);
    double margin_;
};

using TripletLoss = torch::nn::ModuleHolder<TripletLossImpl>;

} // namespace loss
