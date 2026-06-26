#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include "../Trainer/include/Model/Model.h"
#include <torch/csrc/jit/serialization/pickle.h>
#include <fstream>

int main() {
    try {
        const int64_t embedding_dim = 128;
        const double dropout = 0.1;

        std::cout << "Loading model..." << std::endl;
        
        auto model = model::FaceRecognitionModel(3, embedding_dim, dropout);
        auto repo_root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
        auto model_path = repo_root / "models" / "model.pt";
        torch::load(model, model_path.string());
        
        
        
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

        auto output_path = (repo_root / "models" / "model_weights.pt").string();
        std::ofstream fout(output_path, std::ios::out | std::ios::binary);
        fout.write(bytes.data(), bytes.size());
        fout.close();
        std::cout << "State dict saved to model_weights.pt" << std::endl;

        std::cout << "Done!" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}