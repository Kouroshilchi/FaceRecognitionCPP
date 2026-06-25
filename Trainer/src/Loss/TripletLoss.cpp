#include "../include/Loss/TripletLoss.h"
#include <iostream>
namespace Loss 
{
    LossMetrics TripletLossImpl::forward(const torch::Tensor& embeddings, const torch::Tensor& labels) {
    auto N = embeddings.size(0);
    if (N <= 1) {
        return {torch::zeros({}, embeddings.options()), 0.0, 0.0};
    }

    auto device = embeddings.device();
    auto sq = embeddings.pow(2).sum(1);
    auto dist = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * embeddings.matmul(embeddings.t());
    dist = torch::clamp_min(dist, 0.0);
    dist = torch::sqrt(dist + 1e-12);

    auto labels1 = labels.view({-1, 1});
    auto eq = labels1.eq(labels1.t());
    auto neq = labels1.ne(labels1.t());

    auto eye = torch::eye(N, torch::TensorOptions().device(device).dtype(torch::kBool));
    eq = eq.logical_and(eye.logical_not());

    const double NEG_INF = -1e12;
    auto dist_pos = dist.masked_fill(eq.logical_not(), NEG_INF);
    auto hardest_pos = std::get<0>(dist_pos.max(1));

    const double POS_INF = 1e12;
    auto dist_neg = dist.masked_fill(neq.logical_not(), POS_INF);
    auto hardest_neg = std::get<0>(dist_neg.min(1));

    auto losses = torch::relu(hardest_pos - hardest_neg + margin_);

    auto has_pos = eq.any(1);
    auto has_neg = neq.any(1);
    auto valid = has_pos.logical_and(has_neg);
    
    // Calculate average distances
    double avg_pos_dist = 0.0;
    double avg_neg_dist = 0.0;
    
    if (valid.sum().item<int64_t>() > 0) {
        auto hardest_pos_valid = hardest_pos.masked_select(valid);
        auto hardest_neg_valid = hardest_neg.masked_select(valid);
        avg_pos_dist = hardest_pos_valid.mean().item<double>();
        avg_neg_dist = hardest_neg_valid.mean().item<double>();
    }
    
    if (valid.sum().item<int64_t>() == 0) {
        auto loss = torch::zeros({}, embeddings.options()).requires_grad_(true);
        return {loss, 0.0, 0.0};
    }

    auto losses_valid = losses.masked_select(valid);
    return {losses_valid.mean(), avg_pos_dist, avg_neg_dist};
}
}