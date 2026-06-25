#pragma once
#include <torch/torch.h>
#include "backbone.h"
#include "projector.h"


namespace model {
    struct FaceRecognitionModelImpl : torch::nn::Module {
        FaceRecognitionModelImpl(
            int num_channel,
            int out_dim,
            double dropout
        );

        torch::Tensor forward(torch::Tensor x);
        // FaceRecognitionBackBone backbone{nullptr};
        FaceRecognitionProjector projector{nullptr};
    };
    TORCH_MODULE(FaceRecognitionModel);
}