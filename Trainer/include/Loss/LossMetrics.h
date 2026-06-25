#pragma once

#include <torch/torch.h>

namespace Loss {

struct LossMetrics {
    torch::Tensor loss;
    double avg_pos_cos;
    double avg_neg_cos;
};

} // namespace Loss
