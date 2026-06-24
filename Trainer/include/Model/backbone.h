#pragma once
#include <torch/torch.h>


namespace model {
    struct FaceRecognitionBackBoneImpl : torch::nn::Module {
        FaceRecognitionBackBoneImpl(
            int num_channel, 
            int output_channel,
            double dropout
        );

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv1{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr};
        torch::nn::Conv2d conv2{nullptr};
        torch::nn::BatchNorm2d bn2{nullptr};
        torch::nn::Conv2d conv3{nullptr};
        torch::nn::BatchNorm2d bn3{nullptr};
        torch::nn::Conv2d conv4{nullptr};
        torch::nn::BatchNorm2d bn4{nullptr};
        torch::nn::Dropout dropout_layer{nullptr};
        
    };
    TORCH_MODULE(FaceRecognitionBackBone);
}