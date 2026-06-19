#include "../include/model/projector.h"

namespace model {
    FaceRecognitionProjector::FaceRecognitionProjector(
        int in_channel,
        int out_channel,
        int dropout
    )
    {
        resblock1 = register_module("resblock1", ResBlockImpl(in_channel, 256));
        resblock2 = register_module("resblock2", ResBlockImpl(256, 512));
        resblock3 = register_module("resblock3", ResBlockImpl(512, 256));
        resblock4 = register_module("resblock4", ResBlockImpl(256, out_channel));
        dropout_layer = register_module("dropout_layer", torch::nn::Dropout(dropout));
    }
}

FaceRecognitionProjector::forward(torch::Tensor x) {
    x = resblock1->forward(x);
    x = dropout_layer->forward(x);
    x = resblock2->forward(x);
    x = resblock3->forward(x);
    x = resblock4->forward(x);
    return x;
}