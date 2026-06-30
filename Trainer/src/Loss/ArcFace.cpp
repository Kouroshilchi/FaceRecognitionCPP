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

    auto logits = embeddings.matmul(weight_norm.t());
    logits = torch::clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7);

    auto index = labels.view({-1, 1}); // [N,1]

    auto target_logit = logits.gather(1, index).squeeze(1); // [N]

    torch::Tensor final_target_logit;
    {

        torch::NoGradGuard no_grad;
        auto theta_y = torch::acos(target_logit);          // theta_y_i
        final_target_logit = theta_y + m_;                  // theta_y_i + m
        final_target_logit = torch::cos(final_target_logit); // cos(theta_y_i + m)
    }


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