#pragma once
#include <torch/torch.h>


namespace model {
    struct FaceRecognitionHeadImpl : torch::nn::Module {
        FaceRecognitionHeadImpl(
            double output_dim,
            double bottleneck_expansion
        );

        torch::Tensor forward(torch::Tensor x);
        torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
        torch::nn::Linear fc1{nullptr};
        torch::nn::BatchNorm1d bn1_fc1{nullptr};
        torch::nn::Linear fc2{nullptr};
        
    };
    TORCH_MODULE(FaceRecognitionHead);
}