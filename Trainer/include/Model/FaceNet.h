#pragma once

#include <torch/torch.h>
#include "Model/Model.h"
#include "Loss/ArcFace.h"
#include "Loss/LossMetrics.h"

namespace model {

struct FaceNetImpl : torch::nn::Module {
    FaceNetImpl(int64_t num_classes,
               int64_t embedding_dim,
               double dropout,
               double scale = 64.0,
               double margin = 0.5);

    Loss::LossMetrics forward(const torch::Tensor& inputs, const torch::Tensor& labels);

    torch::Tensor embed(const torch::Tensor& inputs);

    model::FaceRecognitionModel backbone{nullptr};
    Loss::ArcFace arcface{nullptr};
};

TORCH_MODULE(FaceNet);

} // namespace model
