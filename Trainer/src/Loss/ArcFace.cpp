#include "../include/Loss/ArcFace.h"
#include <iostream>

namespace Loss {

ArcFaceImpl::ArcFaceImpl(int64_t num_classes,
                         int64_t embedding_dim,
                         double scale,
                         double margin)
    : s_(scale), m_(margin) {

    weight_ = register_parameter("weight",
        torch::randn({num_classes, embedding_dim},
                     torch::TensorOptions().dtype(torch::kFloat32)));
    torch::nn::init::xavier_uniform_(weight_);

    std::cout << "ArcFace initialized: " << num_classes
              << " classes, dim=" << embedding_dim
              << ", scale=" << s_ << ", margin=" << m_ << std::endl;
}

LossMetrics ArcFaceImpl::forward(const torch::Tensor& embeddings,
                                  const torch::Tensor& labels) {

    auto weight_norm = torch::nn::functional::normalize(
        weight_, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    auto cos_theta = embeddings.matmul(weight_norm.t());
    cos_theta = torch::clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7);

    auto one_hot = torch::zeros_like(cos_theta)
                       .scatter_(1, labels.view({-1, 1}), 1.0);

    auto theta       = torch::acos(cos_theta);
    auto cos_theta_m = torch::cos(theta + m_);

    const double threshold    = std::cos(M_PI - m_);
    const double mm           = std::sin(M_PI - m_) * m_;
    auto cos_theta_m_safe = torch::where(
        cos_theta > threshold,
        cos_theta_m,
        cos_theta - mm
    );

    auto output = (one_hot * (cos_theta_m_safe - cos_theta) + cos_theta) * s_;
    auto loss   = torch::nn::functional::cross_entropy(output, labels);

    if (std::isnan(loss.item<double>())) {
        std::cerr << "Warning: NaN loss in ArcFace!" << std::endl;
        loss = torch::tensor(0.0, embeddings.options());
    }

    auto pos_cos           = torch::sum(cos_theta * one_hot, 1);
    double avg_pos_metric  = pos_cos.mean().item<double>();

    auto cos_theta_neg     = cos_theta.masked_fill(one_hot.to(torch::kBool), -1e9);
    auto hardest_neg_cos   = std::get<0>(cos_theta_neg.max(1));
    double avg_neg_metric  = hardest_neg_cos.mean().item<double>();

    return {loss, avg_pos_metric, avg_neg_metric, 0, 0};
}

} 