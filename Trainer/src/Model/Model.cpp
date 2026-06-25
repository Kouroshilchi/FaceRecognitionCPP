#include "../include/Model/Model.h"

namespace model {
    FaceRecognitionModelImpl::FaceRecognitionModelImpl(
        int num_channel,
        int out_dim,
        double dropout
    )
    {
        // backbone = register_module("backbone", FaceRecognitionBackBone(num_channel, 128 , dropout));
        projector = register_module("projector", FaceRecognitionProjector(3, out_dim, dropout));
    }

    torch::Tensor FaceRecognitionModelImpl::forward(torch::Tensor x) {
        // x = backbone->forward(x);
        x = projector->forward(x);
        return x;
    }
}

