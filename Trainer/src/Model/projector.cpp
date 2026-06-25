#include "../include/Model/projector.h"

namespace model {
    FaceRecognitionProjectorImpl::FaceRecognitionProjectorImpl(
        int in_channel,
        int out_dim,
        double dropout
    )
    {
        // ResNet-50 stem
        conv1   = register_module("conv1",   torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, 64, 7).stride(2).padding(3).bias(false)));
        bn1     = register_module("bn1",     torch::nn::BatchNorm2d(64));
        relu    = register_module("relu",    torch::nn::ReLU());
        maxpool = register_module("maxpool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

        int64_t inplanes = 64;
        auto make_layer = [&](int64_t planes, int blocks, int64_t stride) {
            torch::nn::Sequential seq;
            bool need_downsample = (stride != 1) || (inplanes != planes * model::BottleneckImpl::expansion);
            seq->push_back(Bottleneck(inplanes, planes, stride, need_downsample));
            inplanes = planes * model::BottleneckImpl::expansion;
            for (int i = 1; i < blocks; ++i)
                seq->push_back(Bottleneck(inplanes, planes));
            return seq;
        };

        layer1 = register_module("layer1", make_layer(64,  3, 1));
        layer2 = register_module("layer2", make_layer(128, 4, 2));
        layer3 = register_module("layer3", make_layer(256, 6, 2));
        layer4 = register_module("layer4", make_layer(512, 3, 2));

        avgpool       = register_module("avgpool",       torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
        fc1           = register_module("fc1",           torch::nn::Linear(torch::nn::LinearOptions(512 * model::BottleneckImpl::expansion, out_dim)));
        bn1_fc1        = register_module("bn1_fc1",        torch::nn::BatchNorm1d(out_dim));
        dropout_layer = register_module("dropout_layer", torch::nn::Dropout(dropout));

        load_pretrained_weights("C:\\Users\\kuoro\\Documents\\GitHub\\FaceRecognitionCPP\\models\\resnet50_weights.pt");
    }

    void FaceRecognitionProjectorImpl::load_pretrained_weights(const std::string& weight_path) {
        torch::jit::script::Module pretrained = torch::jit::load(weight_path);
        pretrained.eval();

        std::unordered_map<std::string, torch::Tensor> weight_map;
        for (const auto& p : pretrained.named_parameters()) {
            weight_map[p.name] = p.value;
        }
        for (const auto& b : pretrained.named_buffers()) {
            weight_map[b.name] = b.value;
        }

        auto copy_param = [&](torch::Tensor& dst, const std::string& key) {
            auto it = weight_map.find(key);
            if (it != weight_map.end()) {
                torch::NoGradGuard no_grad;
                dst.copy_(it->second);
            } else {
                std::cerr << "Warning: key not found: " << key << std::endl;
            }
        };

        copy_param(conv1->weight,       "conv1.weight");
        copy_param(bn1->weight,         "bn1.weight");
        copy_param(bn1->bias,           "bn1.bias");
        copy_param(bn1->running_mean,   "bn1.running_mean");
        copy_param(bn1->running_var,    "bn1.running_var");

        auto load_block = [&](BottleneckImpl* block, const std::string& prefix) {
            copy_param(block->conv1->weight,     prefix + ".conv1.weight");
            copy_param(block->bn1->weight,       prefix + ".bn1.weight");
            copy_param(block->bn1->bias,         prefix + ".bn1.bias");
            copy_param(block->bn1->running_mean, prefix + ".bn1.running_mean");
            copy_param(block->bn1->running_var,  prefix + ".bn1.running_var");

            copy_param(block->conv2->weight,     prefix + ".conv2.weight");
            copy_param(block->bn2->weight,       prefix + ".bn2.weight");
            copy_param(block->bn2->bias,         prefix + ".bn2.bias");
            copy_param(block->bn2->running_mean, prefix + ".bn2.running_mean");
            copy_param(block->bn2->running_var,  prefix + ".bn2.running_var");

            copy_param(block->conv3->weight,     prefix + ".conv3.weight");
            copy_param(block->bn3->weight,       prefix + ".bn3.weight");
            copy_param(block->bn3->bias,         prefix + ".bn3.bias");
            copy_param(block->bn3->running_mean, prefix + ".bn3.running_mean");
            copy_param(block->bn3->running_var,  prefix + ".bn3.running_var");

            if (!block->downsample_layer.is_empty()) {
                auto* ds_conv = block->downsample_layer[0]->as<torch::nn::Conv2dImpl>();
                auto* ds_bn   = block->downsample_layer[1]->as<torch::nn::BatchNorm2dImpl>();
                copy_param(ds_conv->weight,    prefix + ".downsample.0.weight");
                copy_param(ds_bn->weight,      prefix + ".downsample.1.weight");
                copy_param(ds_bn->bias,        prefix + ".downsample.1.bias");
                copy_param(ds_bn->running_mean,prefix + ".downsample.1.running_mean");
                copy_param(ds_bn->running_var, prefix + ".downsample.1.running_var");
            }
        };

        auto load_layer = [&](torch::nn::Sequential& layer, const std::string& layer_name) {
            int idx = 0;
            for (auto& m : layer->children()) {
                auto* block = m->as<BottleneckImpl>();
                if (block) {
                    load_block(block, layer_name + "." + std::to_string(idx++));
                }
            }
        };

        load_layer(layer1, "layer1");
        load_layer(layer2, "layer2");
        load_layer(layer3, "layer3");
        load_layer(layer4, "layer4");

        std::cout << "ResNet-50 pretrained weights loaded successfully." << std::endl;
    }

} // namespace model

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
    x = bn1_fc1->forward(x);
    x = dropout_layer->forward(x);

    return torch::nn::functional::normalize(x, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));;
}