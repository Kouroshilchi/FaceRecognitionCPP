
#include <torch/torch.h>
#include <iostream>
#include "include/model/Model.h"

int main() {
    try {
        const int64_t embedding_dim = 128;
        const double dropout = 0.1;

        std::cout << "Loading model from model.pt ..." << std::endl;
        auto model = model::FaceRecognitionModel(3, embedding_dim, dropout);
        torch::load(model, "model.pt");
        model->eval();
        std::cout << "Model loaded!" << std::endl;

        std::vector<std::pair<std::string, torch::Tensor>> state_dict;
        for (const auto& pair : model->named_parameters()) {
            state_dict.emplace_back(pair.key(), pair.value().cpu().detach());
            std::cout << "  Saving param: " << pair.key() 
                      << " shape: " << pair.value().sizes() << std::endl;
        }
        for (const auto& pair : model->named_buffers()) {
            state_dict.emplace_back(pair.key(), pair.value().cpu().detach());
            std::cout << "  Saving buffer: " << pair.key() 
                      << " shape: " << pair.value().sizes() << std::endl;
        }

        torch::save(state_dict, "model_weights.pt");


        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}