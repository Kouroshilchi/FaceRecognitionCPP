#pragma once

#include <torch/torch.h>
#include "Model/Model.h"
#include "Loss/ArcFace.h"
#include "Loss/TripletLoss.h"
#include "Loss/LossMetrics.h"

namespace model {

enum class LossType {
    ArcFace,
    TripletSemiHard,
    TripletOnlineHard
};

struct FaceNetImpl : torch::nn::Module {
    FaceNetImpl(int64_t num_classes,
               int64_t embedding_dim,
               double dropout,
               LossType loss_type = LossType::ArcFace,
               double scale  = 64.0,
               double margin = 0.5);

    
    Loss::LossMetrics forward(const torch::Tensor& inputs,
                              const torch::Tensor& labels,
                              int64_t epoch = 1);
    Loss::LossMetrics forward(const torch::Tensor& inputs,
                              const torch::Tensor& labels,
                              LossType loss_type,
                              int64_t epoch = 1);
    torch::Tensor embed(const torch::Tensor& inputs);
    model::FaceRecognitionModel backbone{nullptr};
    Loss::ArcFace               arcface{nullptr};
    Loss::TripletLoss           triplet{nullptr};
    LossType                    default_loss_type{LossType::ArcFace};
};

TORCH_MODULE(FaceNet);

} 