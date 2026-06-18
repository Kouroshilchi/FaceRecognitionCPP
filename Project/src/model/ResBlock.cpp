#include "model/ResBlock.h"
#include <iostream>

namespace model {

ResBlockImpl::ResBlockImpl(int64_t input_dim, 
                           int64_t output_dim, 
                           int64_t stride,
                           bool use_bn)
    : stride_(stride), use_bn_(use_bn) 
{

    use_shortcut_ = (input_dim != output_dim) || (stride != 1);
    
    conv1 = register_module("conv1", 
        torch::nn::Conv1d(
            torch::nn::Conv1dOptions(input_dim, output_dim, 3)
                .stride(stride)
                .padding(1)
        )
    );
    
    if (use_bn_) {
        bn1 = register_module("bn1", 
            torch::nn::BatchNorm1d(output_dim)
        );
    }
    
    conv2 = register_module("conv2", 
        torch::nn::Conv1d(
            torch::nn::Conv1dOptions(output_dim, output_dim, 3)
                .stride(1)
                .padding(1)
        )
    );
    
    if (use_bn_) {
        bn2 = register_module("bn2", 
            torch::nn::BatchNorm1d(output_dim)
        );
    }
    
    if (use_shortcut_) {
        shortcut = register_module("shortcut", 
            torch::nn::Conv1d(
                torch::nn::Conv1dOptions(input_dim, output_dim, 1)
                    .stride(stride)
            )
        );
        
        if (use_bn_) {
            shortcut_bn = register_module("shortcut_bn", 
                torch::nn::BatchNorm1d(output_dim)
            );
        }
    }
    
    std::cout << "ResBlock created: " 
              << input_dim << " -> " << output_dim 
              << " (stride=" << stride 
              << ", shortcut=" << (use_shortcut_ ? "yes" : "no") 
              << ")\n";
}

torch::Tensor ResBlockImpl::forward(torch::Tensor x) {
    torch::Tensor identity = x;
    
    torch::Tensor out = conv1->forward(x);
    if (use_bn_) {
        out = bn1->forward(out);
    }
    out = torch::relu(out);
    
    out = conv2->forward(out);
    if (use_bn_) {
        out = bn2->forward(out);
    }
    
    if (use_shortcut_) {
        identity = shortcut->forward(identity);
        if (use_bn_) {
            identity = shortcut_bn->forward(identity);
        }
    }
    
    out = out + identity;
    out = torch::relu(out);
    
    return out;
}

}