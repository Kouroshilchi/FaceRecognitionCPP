#include "../include/model/Bottleneck.h"

namespace model {

BottleneckImpl::BottleneckImpl(int64_t inplanes, int64_t planes, int64_t stride, bool downsample) {
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes, 1).stride(1)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(planes));

    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(planes, planes, 3).stride(stride).padding(1)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(planes));

    conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(planes, planes * expansion, 1).stride(1)));
    bn3 = register_module("bn3", torch::nn::BatchNorm2d(planes * expansion));

    if (downsample || inplanes != planes * expansion) {
        downsample_layer = register_module("downsample", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes * expansion, 1).stride(stride)),
            torch::nn::BatchNorm2d(planes * expansion)
        ));
    }
}

torch::Tensor BottleneckImpl::forward(torch::Tensor x) {
    torch::Tensor identity = x;

    torch::Tensor out = conv1->forward(x);
    out = bn1->forward(out);
    out = torch::relu(out);

    out = conv2->forward(out);
    out = bn2->forward(out);
    out = torch::relu(out);

    out = conv3->forward(out);
    out = bn3->forward(out);

    if (!downsample_layer.is_empty()) {
        identity = downsample_layer->forward(identity);
    }

    out += identity;
    out = torch::relu(out);
    return out;
}

} // namespace model
