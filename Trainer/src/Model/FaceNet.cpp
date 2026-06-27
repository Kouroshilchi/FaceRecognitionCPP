#include "../include/Model/FaceNet.h"
#include <iostream>

namespace model {

FaceNetImpl::FaceNetImpl(int64_t num_classes,
                         int64_t embedding_dim,
                         double dropout,
                         double scale,
                         double margin) {
    backbone = register_module("backbone", FaceRecognitionModel(3, embedding_dim, dropout));
    arcface = register_module("arcface", Loss::ArcFace(num_classes, embedding_dim, scale, margin));
}

Loss::LossMetrics FaceNetImpl::forward(const torch::Tensor& inputs, const torch::Tensor& labels) {
    auto embeddings = backbone->forward(inputs);
    return arcface->forward(embeddings, labels);
}

torch::Tensor FaceNetImpl::embed(const torch::Tensor& inputs) {
    torch::NoGradGuard no_grad;
    backbone->eval();
    auto emb = backbone->forward(inputs);
    return emb;
}

} // namespace model
