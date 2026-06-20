
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
        torch::load(model, "model.pt");
        model->to(device);
        model->eval();
        std::cout << "Model loaded successfully!" << std::endl;

        auto scripted = torch::jit::script(model);

        std::cout << "Saving to torchscript.pt ..." << std::endl;
        scripted.save("torchscript.pt");

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}