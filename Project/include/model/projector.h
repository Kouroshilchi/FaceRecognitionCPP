#pragma once
#include <torch/torch.h>
#include "ResBlock.h"

namespace model {
    struct FaceRecognitionProjector : torch::nn::Module {
        FaceRecognitionProjector(
            int in_channel, 
            int out_dim,
            int dropout
        );

        torch::Tensor forward(torch::Tensor x);

        ResBlock resblock1{nullptr};
        ResBlock resblock2{nullptr};
        ResBlock resblock3{nullptr};
        ResBlock resblock4{nullptr};
        torch::nn::Linear fc1{nullptr};
        torch::nn::Flatten flatten{nullptr};
        torch::nn::BatchNorm1d bn1{nullptr};
        torch::nn::ReLU relu{nullptr};
        torch::nn::Linear fc2{nullptr};
        torch::nn::BatchNorm1d bn2{nullptr};
        torch::nn::Linear fc3{nullptr};
        torch::nn::BatchNorm1d bn3{nullptr};
        torch::nn::Dropout dropout_layer{nullptr};
        
    };
    TORCH_MODULE(FaceRecognitionProjector);
}