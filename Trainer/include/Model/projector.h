#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include "Bottleneck.h"

namespace model {
    struct FaceRecognitionProjectorImpl : torch::nn::Module {
        FaceRecognitionProjectorImpl(
            int in_channel, 
            int out_dim,
            double dropout
        );

        torch::Tensor forward(torch::Tensor x);
        void load_pretrained_weights(const std::string& weight_path);

        torch::nn::Conv2d conv1{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr};
        torch::nn::ReLU relu{nullptr};
        torch::nn::PReLU prelu{nullptr};
        torch::nn::MaxPool2d maxpool{nullptr};

        torch::nn::Sequential layer1{nullptr};
        torch::nn::Sequential layer2{nullptr};
        torch::nn::Sequential layer3{nullptr};
        torch::nn::Sequential layer4{nullptr};

        torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
        torch::nn::Linear fc1{nullptr};
        torch::nn::BatchNorm1d bn1_fc1{nullptr};
        torch::nn::Linear fc2{nullptr};
        torch::nn::BatchNorm1d bn2_fc2{nullptr};
        // torch::nn::Dropout dropout_layer{nullptr};
    };
    TORCH_MODULE(FaceRecognitionProjector);
}