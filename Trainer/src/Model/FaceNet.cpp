#include "../include/Model/FaceNet.h"
#include <iostream>

namespace model {

FaceNetImpl::FaceNetImpl(int64_t num_classes,
                         int64_t embedding_dim,
                         double  dropout,
                         double  scale,
                         double  margin) {
    backbone = register_module("backbone", FaceRecognitionModel(3, embedding_dim, dropout));
    arcface  = register_module("arcface",  Loss::ArcFace(num_classes, embedding_dim, scale, margin));
    triplet  = register_module("triplet",  Loss::TripletLoss(0.3));
}

Loss::LossMetrics FaceNetImpl::forward(const torch::Tensor& inputs,
                                        const torch::Tensor& labels,
                                        int64_t epoch) {
    backbone->train();
    auto embeddings = backbone->forward(inputs);
    // std::cout << "Embeddings shape: " << embeddings << std::endl;
    // return triplet->forward_online_hard(embeddings , labels);
    return arcface->forward(embeddings , labels);

}


torch::Tensor FaceNetImpl::embed(const torch::Tensor& inputs) {
    torch::NoGradGuard no_grad;
    backbone->eval();
    auto emb   = backbone->forward(inputs);
    backbone->train();
    return emb;
}
}