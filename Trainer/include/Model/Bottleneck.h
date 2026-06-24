#pragma once

#include <torch/torch.h>

namespace model {

struct BottleneckImpl : torch::nn::Module {
    BottleneckImpl(int64_t inplanes, int64_t planes, int64_t stride = 1, bool downsample = false);

    torch::Tensor forward(torch::Tensor x);

    static const int expansion = 4;

    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};
    torch::nn::Conv2d conv3{nullptr};
    torch::nn::BatchNorm2d bn3{nullptr};
    torch::nn::Sequential downsample_layer{nullptr};
};

TORCH_MODULE(Bottleneck);

} // namespace model
