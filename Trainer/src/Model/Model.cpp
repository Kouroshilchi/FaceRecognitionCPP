#include "../include/Model/Model.h"
#include "../include/Model/head.h"

namespace model {
    FaceRecognitionModelImpl::FaceRecognitionModelImpl(
        int num_channel,
        int out_dim,
        bool pretrained_
    )
    {
        projector = register_module("projector", FaceRecognitionProjector(3 , pretrained_));
        head = register_module("head", FaceRecognitionHead(out_dim , 4));
    }

    torch::Tensor FaceRecognitionModelImpl::forward(torch::Tensor x) {
        x = projector->forward(x);
        x = head->forward(x);
        return x;
    }
}

