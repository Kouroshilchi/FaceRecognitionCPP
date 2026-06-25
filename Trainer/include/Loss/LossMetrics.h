#pragma once

#include <torch/torch.h>

namespace Loss {

struct LossMetrics {
    torch::Tensor loss;
    double avg_pos_metric;
    double avg_neg_metric;
};

} // namespace Loss
