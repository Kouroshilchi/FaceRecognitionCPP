#include "../include/model/projector.h"

namespace model {
    FaceRecognitionProjectorImpl::FaceRecognitionProjectorImpl(
        int in_channel,
        int out_dim,
        double dropout
    )
    {
        // Initial ResNet-50 stem
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, 64, 7).stride(2).padding(3)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
        relu = register_module("relu", torch::nn::ReLU());
        maxpool = register_module("maxpool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

        int64_t inplanes = 64;

        auto make_layer = [&](int64_t planes, int blocks, int64_t stride) {
            torch::nn::Sequential seq;
            bool need_downsample = (stride != 1) || (inplanes != planes * model::BottleneckImpl::expansion);
            seq->push_back(Bottleneck(inplanes, planes, stride, need_downsample));
            inplanes = planes * model::BottleneckImpl::expansion;
            for (int i = 1; i < blocks; ++i) {
                seq->push_back(Bottleneck(inplanes, planes));
            }
            return seq;
        };

        layer1 = register_module("layer1", make_layer(64, 3, 1));
        layer2 = register_module("layer2", make_layer(128, 4, 2));
        layer3 = register_module("layer3", make_layer(256, 6, 2));
        layer4 = register_module("layer4", make_layer(512, 3, 2));

        avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));

        fc1 = register_module("fc1", torch::nn::Linear(512 * model::BottleneckImpl::expansion, 256));
        bn_fc1 = register_module("bn_fc1", torch::nn::BatchNorm1d(256));
        fc2 = register_module("fc2", torch::nn::Linear(256, 512));
        bn_fc2 = register_module("bn_fc2", torch::nn::BatchNorm1d(512));
        fc3 = register_module("fc3", torch::nn::Linear(512, out_dim));
        bn_fc3 = register_module("bn_fc3", torch::nn::BatchNorm1d(out_dim));
        dropout_layer = register_module("dropout_layer", torch::nn::Dropout(dropout));
    }
}

torch::Tensor model::FaceRecognitionProjectorImpl::forward(torch::Tensor x) {
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = relu->forward(x);
    x = maxpool->forward(x);

    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);

    x = avgpool->forward(x);
    x = x.view({x.size(0), -1});

    x = fc1->forward(x);
    x = bn_fc1->forward(x);
    x = relu->forward(x);
    x = dropout_layer->forward(x);

    x = fc2->forward(x);
    x = bn_fc2->forward(x);
    x = relu->forward(x);
    x = dropout_layer->forward(x);

    x = fc3->forward(x);
    x = bn_fc3->forward(x);

    return x;
}