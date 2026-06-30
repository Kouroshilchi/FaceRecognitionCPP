#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include "Bottleneck.h"

namespace model {
    struct FaceRecognitionProjectorImpl : torch::nn::Module {
        FaceRecognitionProjectorImpl(
            int in_channel, 
            bool pretrained_=true
        );

        torch::Tensor forward(torch::Tensor x);
        void load_pretrained_weights(const std::string& weight_path);

        torch::nn::Conv2d conv1{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr};
        torch::nn::ReLU relu{nullptr};
        torch::nn::MaxPool2d maxpool{nullptr};

        torch::nn::Sequential layer1{nullptr};
        torch::nn::Sequential layer2{nullptr};
        torch::nn::Sequential layer3{nullptr};
        torch::nn::Sequential layer4{nullptr};

    };
    TORCH_MODULE(FaceRecognitionProjector);
}