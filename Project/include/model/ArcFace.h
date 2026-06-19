#pragma once
#include <torch/torch.h>

namespace model {

struct ArcFaceImpl : torch::nn::Module {
    ArcFaceImpl(int64_t in_features, int64_t out_features, float s = 64.0, float m = 0.5, bool easy_margin = false);

    torch::Tensor forward(torch::Tensor features, torch::Tensor labels);

    torch::Tensor weight;
    float s;
    float m;
    float cos_m;
    float sin_m;
    float thresh;
    float mm;
    bool easy_margin;
};
TORCH_MODULE(ArcFace);

}