#include "../include/model/TripletLoss.h"
#include <iostream>

torch::Tensor loss::TripletLossImpl::forward(
    const torch::Tensor& embeddings,
    const torch::Tensor& labels)
{
    auto N = embeddings.size(0);
    if (N <= 1)
        return torch::zeros({}, embeddings.options()).requires_grad_(true);

    auto device = embeddings.device();

    auto dist = torch::cdist(embeddings, embeddings, 2); // [N, N]

    auto labels_col = labels.view({-1, 1});
    auto pos_mask   = labels_col.eq(labels_col.t());   // [N,N] - same class
    auto neg_mask   = labels_col.ne(labels_col.t());   // [N,N] - different class

    auto eye = torch::eye(N, torch::TensorOptions().device(device).dtype(torch::kBool));
    pos_mask = pos_mask.logical_and(eye.logical_not());

    auto dist_pos   = dist * pos_mask.to(embeddings.dtype());
    auto hardest_pos = std::get<0>(dist_pos.max(1)); // [N]
    auto dist_neg    = dist + (~neg_mask).to(embeddings.dtype()) * 1e12;
    auto hardest_neg = std::get<0>(dist_neg.min(1)); // [N]

    auto losses = torch::relu(hardest_pos - hardest_neg + margin_);

    auto has_pos = pos_mask.any(1);
    auto has_neg = neg_mask.any(1);
    auto valid   = has_pos.logical_and(has_neg);

    auto losses_valid = losses.masked_select(valid);
    if (losses_valid.numel() == 0)
        return torch::zeros({}, embeddings.options()).requires_grad_(true);

    return losses_valid.mean();
}