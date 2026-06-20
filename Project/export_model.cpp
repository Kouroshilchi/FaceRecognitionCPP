#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include "include/model/Model.h"

int main() {
    try {
        const int64_t embedding_dim = 128;
        const double dropout = 0.1;

        torch::Device device(torch::kCPU);

        std::cout << "Loading model from model.pt ..." << std::endl;
        auto model = model::FaceRecognitionModel(3, embedding_dim, dropout);
        torch::load(model, "../models/model.pt");
        model->to(device);
        model->eval();
        std::cout << "Model loaded!" << std::endl;

        torch::Tensor example_input = torch::zeros({1, 3, 224, 224}, device);

        std::cout << "Tracing model ..." << std::endl;
        auto traced = torch::jit::trace(model, example_input);

        std::cout << "Saving to torchscript.pt ..." << std::endl;
        traced.save("torchscript.pt");

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}