#pragma once
#include <torch/torch.h>
#include "backbone.h"
#include "projector.h"


namespace model {
    struct FaceRecognitionModel : torch::nn::Module {
        FaceRecognitionModel(
            int num_channel,
            int out_dim,
            int dropout
        );

        torch::Tensor forward(torch::Tensor x);
        FaceRecognitionBackBone backbone{nullptr};
        FaceRecognitionProjector projector{nullptr};
    };
    TORCH_MODULE(FaceRecognitionModel);
}