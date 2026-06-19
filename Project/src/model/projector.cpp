#include "../include/model/projector.h"

namespace model {
    FaceRecognitionProjectorImpl::FaceRecognitionProjectorImpl(
        int in_channel,
        int out_dim,
        double dropout
    )
    {
        resblock1 = register_module("resblock1", ResBlock(in_channel, 128 , 2));
        resblock2 = register_module("resblock2", ResBlock(128, 256, 1));
        resblock3 = register_module("resblock3", ResBlock(256, 512, 2));
        resblock4 = register_module("resblock4", ResBlock(512, 512, 1));
        resblock5 = register_module("resblock5", ResBlock(512, 512, 2));
        flatten = register_module("flatten", torch::nn::Flatten());
        fc1 = register_module("fc1" , torch::nn::Linear(512 * 28 * 28, 512));
        bn1 = register_module("bn1", torch::nn::BatchNorm1d(512));
        relu = register_module("relu", torch::nn::ReLU());  
        fc2 = register_module("fc2", torch::nn::Linear(512, 256));
        bn2 = register_module("bn2", torch::nn::BatchNorm1d(256));
        fc3 = register_module("fc3", torch::nn::Linear(256, out_dim));
        bn3 = register_module("bn3", torch::nn::BatchNorm1d(out_dim));
        dropout_layer = register_module("dropout_layer", torch::nn::Dropout(dropout));
    }
}

torch::Tensor model::FaceRecognitionProjectorImpl::forward(torch::Tensor x) {
    x = resblock1->forward(x);
    x = resblock2->forward(x);
    x = resblock3->forward(x);
    x = resblock4->forward(x);
    x = resblock5->forward(x);
    
    x = flatten->forward(x);
    
    x = fc1->forward(x);
    x = bn1->forward(x);
    x = relu->forward(x);
    x = dropout_layer->forward(x);
    
    x = fc2->forward(x);
    x = bn2->forward(x);
    x = relu->forward(x);
    x = dropout_layer->forward(x);
    
    x = fc3->forward(x);
    x = bn3->forward(x);

    return x;
}