#pragma once
#include <torch/torch.h>
#include "ResBlock.h"
#include <string>

namespace model {
    struct FaceRecognitionProjector : torch::nn::Module {
        FaceRecognitionProjector(
            int in_channel, 
            int out_channel,
            int dropout
        );

        torch::Tensor forward(torch::Tensor x);

        ResBlockImpl resblock1{nullptr};
        ResBlockImpl resblock2{nullptr};
        ResBlockImpl resblock3{nullptr};
        ResBlockImpl resblock4{nullptr};
        torch::nn::Dropout dropout_layer{nullptr};
        
    };
    TORCH_MODULE(FaceRecognitionProjector);
}