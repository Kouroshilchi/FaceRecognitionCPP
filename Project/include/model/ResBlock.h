#pragma once

#include <torch/torch.h>

namespace model {

struct ResBlockImpl : torch::nn::Module {

    ResBlockImpl(int64_t input_dim, 
                 int64_t output_dim, 
                 int64_t stride = 1,
                 bool use_bn = true);
    
    torch::Tensor forward(torch::Tensor x);
    
    torch::nn::Conv1d conv1{nullptr};  
    torch::nn::Conv1d conv2{nullptr};   
    torch::nn::BatchNorm1d bn1{nullptr};  
    torch::nn::BatchNorm1d bn2{nullptr};
    
    torch::nn::Conv1d shortcut{nullptr};   
    torch::nn::BatchNorm1d shortcut_bn{nullptr};
    
    int64_t stride_;
    bool use_bn_;
    bool use_shortcut_;  
};

TORCH_MODULE(ResBlock);

} 