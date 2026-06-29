#include "../include/Model/FaceNet.h"
#include <iostream>

namespace model {

FaceNetImpl::FaceNetImpl(int64_t num_classes,
                         int64_t embedding_dim,
                         double  dropout,
                         double  scale,
                         double  margin) {
    backbone = register_module("backbone", FaceRecognitionModel(3, embedding_dim, dropout));
    // arcface  = register_module("arcface",  Loss::ArcFace(num_classes, embedding_dim, scale, margin));
    triplet  = register_module("triplet",  Loss::TripletLoss(0.3));
}

Loss::LossMetrics FaceNetImpl::forward(const torch::Tensor& inputs,
                                        const torch::Tensor& labels,
                                        int64_t epoch) {
    auto embeddings = backbone->forward(inputs);

    embeddings = torch::nn::functional::normalize(
        embeddings,
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1)
    );
    Loss::LossMetrics metrics;

    // if (epoch <= 10) {
    //     metrics = triplet->forward_semi_hard(embeddings , labels);  
    // } else {
    //     metrics = triplet->forward_online_hard(embeddings , labels);
    // }
    metrics = triplet->forward_online_hard(embeddings , labels);
    return metrics;
}


torch::Tensor FaceNetImpl::embed(const torch::Tensor& inputs) {
    torch::NoGradGuard no_grad;
    this->eval();
    auto emb   = backbone->forward(inputs);
    emb = torch::nn::functional::normalize(
        emb,
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1)
    );
    this->train();
    return emb;
}
}