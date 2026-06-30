#include "../include/Model/FaceNet.h"
#include <iostream>

namespace model {

FaceNetImpl::FaceNetImpl(int64_t num_classes,
                         int64_t embedding_dim,
                         LossType loss_type,
                         double  scale,
                         double  margin , 
                         bool pretrained_)
    : default_loss_type(loss_type) {
    backbone = register_module("backbone", FaceRecognitionModel(3, embedding_dim , pretrained_));
    arcface  = register_module("arcface",  Loss::ArcFace(num_classes, embedding_dim, scale, margin));
    triplet  = register_module("triplet",  Loss::TripletLoss(0.3));
}

Loss::LossMetrics FaceNetImpl::forward(const torch::Tensor& inputs,
                                        const torch::Tensor& labels,
                                        int64_t epoch) {
    return forward(inputs, labels, default_loss_type, epoch);
}

Loss::LossMetrics FaceNetImpl::forward(const torch::Tensor& inputs,
                                        const torch::Tensor& labels,
                                        LossType loss_type,
                                        int64_t epoch) {
    this->train();
    auto embeddings = backbone->forward(inputs);
    switch (loss_type) {
        case LossType::ArcFace:
            return arcface->forward(embeddings, labels);
        case LossType::TripletSemiHard:
            return triplet->forward_semi_hard(embeddings, labels);
        case LossType::TripletOnlineHard:
            return triplet->forward_online_hard(embeddings, labels);
        default:
            return arcface->forward(embeddings, labels);
    }
}


torch::Tensor FaceNetImpl::embed(const torch::Tensor& inputs) {
    torch::NoGradGuard no_grad;
    this->eval();
    auto emb   = backbone->forward(inputs);
    this->train();
    return emb;
}
}