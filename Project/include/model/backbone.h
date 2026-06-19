#pragma once
#include <torch/torch.h>


namespace model {
    struct FaceRecognitionBackBone : torch::nn::Module {
        FaceRecognitionBackBone(
            int num_channel, 
            int output_channel,
            int dropout
        );

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv1{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr};
        torch::nn::Conv2d conv2{nullptr};
        torch::nn::BatchNorm2d bn2{nullptr};
        torch::nn::Dropout dropout_layer{nullptr};
        
    };
    TORCH_MODULE(FaceRecognitionBackBone);
}