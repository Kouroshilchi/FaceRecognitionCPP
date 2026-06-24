#include "../include/Model/ResBlock.h"

namespace model {

ResBlockImpl::ResBlockImpl(int64_t in_channels, int64_t out_channels, int64_t stride) {
    conv1 = register_module("conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3)
                              .stride(stride)
                              .padding(1)));
    
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));
    
    conv2 = register_module("conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3)
                              .stride(1)
                              .padding(1)));
    
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
    
    if (in_channels != out_channels || stride != 1) {
        shortcut = register_module("shortcut",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1)
                                  .stride(stride)));
    }
}

torch::Tensor ResBlockImpl::forward(torch::Tensor x) {
    torch::Tensor identity = x;
    

    torch::Tensor out = conv1->forward(x);
    out = bn1->forward(out);
    out = torch::relu(out);
    
    out = conv2->forward(out);
    out = bn2->forward(out);
    
    if (shortcut) {
        identity = shortcut->forward(identity);
    }
    
    out = out + identity;
    out = torch::relu(out);
    
    return out;
}

}