#include "../include/Model/FaceNet.h"
#include <iostream>

namespace model {

FaceNetImpl::FaceNetImpl(int64_t num_classes,
                         int64_t embedding_dim,
                         double dropout,
                         double scale,
                         double margin) {
    backbone = register_module("backbone", FaceRecognitionModel(3, embedding_dim, dropout));
    arcface  = register_module("arcface",  Loss::ArcFace(num_classes, embedding_dim, scale, margin));
}

Loss::LossMetrics FaceNetImpl::forward(const torch::Tensor& inputs, const torch::Tensor& labels) {
    auto embeddings = backbone->forward(inputs);
    return arcface->forward(embeddings, labels);
}

torch::Tensor FaceNetImpl::embed(const torch::Tensor& inputs) {
    torch::NoGradGuard no_grad;

    this->eval();

    auto emb = backbone->forward(inputs);

    auto norms = emb.norm(2, /*dim=*/1, /*keepdim=*/true).clamp_min(1e-12);
    emb = emb / norms;

    return emb;
}

} // namespace model