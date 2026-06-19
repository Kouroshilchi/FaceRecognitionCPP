#include "../include/model/Model.h"

namespace model {
    FaceRecognitionModelImpl::FaceRecognitionModelImpl(
        int num_channel,
        int out_dim,
        int dropout
    )
    {
        backbone = register_module("backbone", FaceRecognitionBackBone(num_channel, 64 , dropout));
        projector = register_module("projector", FaceRecognitionProjector(64, out_dim, dropout));
    }
}

torch::Tensor FaceRecognitionModelImpl::forward(torch::Tensor x) {
    x = backbone->forward(x);
    x = projector->forward(x);
    return x;
}