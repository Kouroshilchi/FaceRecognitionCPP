#pragma once

#include <torch/torch.h>

namespace model {

struct ResBlockImpl : torch::nn::Module {
    ResBlockImpl(int64_t in_channels, int64_t out_channels, int64_t stride = 1);
    
    torch::Tensor forward(torch::Tensor x);
    
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};
    torch::nn::Conv2d shortcut{nullptr}; 
};

TORCH_MODULE(ResBlock);

} 