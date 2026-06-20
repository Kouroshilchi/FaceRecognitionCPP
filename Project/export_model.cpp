#include <torch/torch.h>
#include <iostream>
#include "include/model/Model.h"

int main() {
    try {
        const int64_t embedding_dim = 128;
        const double  dropout       = 0.1;

        std::cout << "Loading model from model.pt ..." << std::endl;
        auto model = model::FaceRecognitionModel(3, embedding_dim, dropout);
        torch::load(model, "model.pt");
        model->eval();
        std::cout << "Model loaded!" << std::endl;

        torch::serialize::OutputArchive archive;

        for (const auto& pair : model->named_parameters()) {
            archive.write(pair.key(), pair.value().cpu().detach());
            std::cout << "  param: " << pair.key()
                      << "  shape: " << pair.value().sizes() << std::endl;
        }
        for (const auto& pair : model->named_buffers()) {
            archive.write(pair.key(), pair.value().cpu().detach());
            std::cout << "  buffer: " << pair.key()
                      << "  shape: " << pair.value().sizes() << std::endl;
        }

        archive.save_to("model_weights.pt");
        std::cout << "\nDone! model_weights.pt" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}