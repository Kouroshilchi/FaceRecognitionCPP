#include <torch/torch.h>
#include <iostream>
#include "../Trainer/include/Model/Model.h"
#include <torch/csrc/jit/serialization/pickle.h>
#include <fstream>

int main() {
    try {
        const int64_t embedding_dim = 128;
        const double dropout = 0.1;

        std::cout << "Loading model..." << std::endl;
        
        auto model = model::FaceRecognitionModel(3, embedding_dim, dropout); 
        torch::load(model, "C:\\Users\\kuoro\\Documents\\GitHub\\FaceRecognitionCPP\\models\\model.pt");
        
        
        
        std::cout << "Model loaded successfully!" << std::endl;


        torch::OrderedDict<std::string, torch::Tensor> state_dict;
        for (const auto& p : model->named_parameters()) {
            state_dict.insert(p.key(), p.value());
        }
        for (const auto& b : model->named_buffers()) { 
            state_dict.insert(b.key(), b.value());
        }

        c10::Dict<std::string, at::Tensor> dict;
        for (const auto& item : state_dict) {
            dict.insert(item.key(), item.value());
        }

        auto bytes = torch::pickle_save(dict);

        std::ofstream fout("C:\\Users\\kuoro\\Documents\\GitHub\\FaceRecognitionCPP\\models\\model_weights.pt", std::ios::out | std::ios::binary);
        fout.write(bytes.data(), bytes.size());
        fout.close();
        std::cout << "State dict saved to model_state.pt" << std::endl;

        std::cout << "Done!" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}