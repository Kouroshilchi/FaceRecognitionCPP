#include "../include/Loss/ArcFace.h"
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

    auto logits = embeddings.matmul(weight_norm.t());
    logits = torch::clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7);

    auto index = labels.view({-1, 1}); // [N,1]

    auto target_logit = logits.gather(1, index).squeeze(1); // [N] = cos(theta_y)

    // cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
    const double cos_m = std::cos(m_);
    const double sin_m = std::sin(m_);
    const double th    = std::cos(M_PI - m_);          // threshold: cos(pi - m)
    const double mm    = std::sin(M_PI - m_) * m_;       // correction term

    auto sin_theta = torch::sqrt(torch::clamp(1.0 - target_logit * target_logit, 0.0, 1.0));
    auto cos_theta_m = target_logit * cos_m - sin_theta * sin_m;

    // Monotonicity fix: only use cos(theta+m) while theta+m <= pi,
    // otherwise fall back to a linear/easy-margin substitute (cos_theta - mm)
    auto final_target_logit = torch::where(target_logit > th, cos_theta_m, target_logit - mm);

    logits = logits.scatter(1, index, final_target_logit.unsqueeze(1));

    auto output = logits * s_;
    auto loss   = torch::nn::functional::cross_entropy(output, labels);

    if (std::isnan(loss.item<double>())) {
        std::cerr << "Warning: NaN loss in ArcFace!" << std::endl;
        loss = torch::tensor(0.0, embeddings.options());
    }

    auto cos_theta_raw = embeddings.matmul(weight_norm.t()); 
    auto one_hot = torch::zeros_like(cos_theta_raw).scatter_(1, index, 1.0);

    auto pos_cos          = torch::sum(cos_theta_raw * one_hot, 1);
    double avg_pos_metric = pos_cos.mean().item<double>();

    auto cos_theta_neg    = cos_theta_raw.masked_fill(one_hot.to(torch::kBool), -1e9);
    auto hardest_neg_cos  = std::get<0>(cos_theta_neg.max(1));
    double avg_neg_metric = hardest_neg_cos.mean().item<double>();

    return {loss, avg_pos_metric, avg_neg_metric, 0, 0};
}

}