#include "../include/model/backbone.h"

namespace model {
    FaceRecognitionBackBoneImpl::FaceRecognitionBackBoneImpl(
        int num_channel,
        int output_channel,
        int dropout
    ) 
    {
        conv1 = register_module("conv1", 
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(
                    num_channel, 
                    64, 
                    3).stride(1).padding(1)));
        bn1 = register_module("bn1", 
            torch::nn::BatchNorm2d(64));
        conv2 = register_module("conv2", 
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(64, 
                    output_channel, 
                    3).stride(1).padding(1)));
        bn2 = register_module("bn2", 
            torch::nn::BatchNorm2d(output_channel));
        dropout_layer = register_module("dropout_layer", 
            torch::nn::Dropout(dropout));
    }
}
torch::Tensor FaceRecognitionBackBoneImpl::forward(torch::Tensor x) {
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    x = conv2->forward(x);
    x = bn2->forward(x);
    x = torch::relu(x);
    x = dropout_layer->forward(x);
    return x;
}