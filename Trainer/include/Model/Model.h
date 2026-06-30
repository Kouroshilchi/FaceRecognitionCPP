#pragma once
#include <torch/torch.h>
#include "projector.h"
#include "head.h"


namespace model {
    struct FaceRecognitionModelImpl : torch::nn::Module {
        FaceRecognitionModelImpl(
            int num_channel,
            int out_dim,
            bool pretrained_=true
        );

        torch::Tensor forward(torch::Tensor x);
        FaceRecognitionHead head{nullptr};
        FaceRecognitionProjector projector{nullptr};
    };
    TORCH_MODULE(FaceRecognitionModel);
}