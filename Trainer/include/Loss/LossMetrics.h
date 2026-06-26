#pragma once

#include <torch/torch.h>

namespace Loss {

struct LossMetrics {
    torch::Tensor loss;
    double avg_pos_metric;
    double avg_neg_metric;
    int64_t num_valid_triplets;
    int64_t num_zero_loss_triplets;
};

} // namespace Loss
