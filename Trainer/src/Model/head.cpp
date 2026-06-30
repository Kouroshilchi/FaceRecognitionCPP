#include "../include/Model/head.h"

namespace model {
    FaceRecognitionHeadImpl::FaceRecognitionHeadImpl(
        double output_dim,
        double bottleneck_expansion
    ) 
    {
        avgpool       = register_module("avgpool",       torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
        fc1           = register_module("fc1",           torch::nn::Linear(torch::nn::LinearOptions(512 * bottleneck_expansion, 512).bias(false)));
        bn1_fc1        = register_module("bn1_fc1",        torch::nn::BatchNorm1d(512));
        fc2           = register_module("fc2",           torch::nn::Linear(torch::nn::LinearOptions(512, output_dim).bias(false)));
    }
    torch::Tensor FaceRecognitionHeadImpl::forward(torch::Tensor x) {
        x = avgpool->forward(x);
        x = x.view({x.size(0), -1});
        x = fc1->forward(x);
        x = bn1_fc1->forward(x);
        x = fc2->forward(x);
        x = torch::nn::functional::normalize(x, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        return x;
    }   
}
