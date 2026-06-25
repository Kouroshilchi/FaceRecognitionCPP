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
    
    auto batch_size = embeddings.size(0);
    auto device = embeddings.device();

    // Normalize embeddings
    auto embeddings_norm = torch::nn::functional::normalize(
        embeddings, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    // Normalize weights
    auto weight_norm = torch::nn::functional::normalize(
        weight_, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    // Cosine similarity: [batch_size, num_classes]
    auto cos_theta = embeddings_norm.matmul(weight_norm.t());

    cos_theta = torch::clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7);

    // One-hot for ground truth
    auto one_hot = torch::zeros_like(cos_theta)
                    .scatter_(1, labels.view({-1, 1}), 1.0);

    // ArcFace margin
    auto theta = torch::acos(cos_theta);
    auto cos_theta_m = torch::cos(theta + m_);

    auto output = (one_hot * (cos_theta_m - cos_theta) + cos_theta) * s_;

    // Loss
    auto loss = torch::nn::functional::cross_entropy(output, labels);

    // ====================== Metrics ======================
    double avg_pos_cos = 0.0;
    double avg_neg_cos = 0.0;

    // Cosine similarity به کلاس درست (Positive)
    auto pos_cos = torch::sum(cos_theta * one_hot, 1);  // [batch_size]
    avg_pos_cos = pos_cos.mean().item<double>();

    // Hardest Negative: حداکثر cosine similarity به کلاس‌های اشتباه
    auto cos_theta_neg = cos_theta.masked_fill(one_hot.to(torch::kBool), -1e9);
    auto hardest_neg_cos = std::get<0>(cos_theta_neg.max(1));
    avg_neg_cos = hardest_neg_cos.mean().item<double>();

    // NaN check
    if (std::isnan(loss.item<double>())) {
        std::cerr << "Warning: NaN loss in ArcFace!" << std::endl;
        loss = torch::tensor(0.0, embeddings.options());
    }

    return {loss, avg_pos_cos, avg_neg_cos};
}

} // namespace Loss