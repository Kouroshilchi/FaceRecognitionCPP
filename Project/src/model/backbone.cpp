#include "../include/model/backbone.h"

namespace model {
    FaceRecognitionBackBoneImpl::FaceRecognitionBackBoneImpl(
        int num_channel,
        int output_channel,
        double dropout
    ) 
    {
        conv1 = register_module("conv1", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_channel, 32, 4).stride(1).padding(1)));
        bn1 = register_module("bn1", 
            torch::nn::BatchNorm2d(32));
        conv2 = register_module("conv2", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_channel, 64, 3).stride(1).padding(1)));
        bn2 = register_module("bn2", 
            torch::nn::BatchNorm2d(64));
        conv3 = register_module("conv3", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_channel, 128, 4).stride(1).padding(1)));
        bn3 = register_module("bn3", 
            torch::nn::BatchNorm2d(128));
        conv4 = register_module("conv4", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128,output_channel,3).stride(1).padding(1)));
        bn4 = register_module("bn4", 
            torch::nn::BatchNorm2d(output_channel));
        dropout_layer = register_module("dropout_layer", 
            torch::nn::Dropout(dropout));
    }
    torch::Tensor FaceRecognitionBackBoneImpl::forward(torch::Tensor x) {
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    x = dropout_layer->forward(x);
    x = conv2->forward(x);
    x = bn2->forward(x);
    x = torch::relu(x);
    x = conv3->forward(x);
    x = bn3->forward(x);
    x = torch::relu(x);
    x = conv4->forward(x);
    x = bn4->forward(x);
    x = torch::relu(x);
    x = dropout_layer->forward(x);
    return x;
    }   
}
