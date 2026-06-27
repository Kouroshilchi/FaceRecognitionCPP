#include <torch/torch.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include "Model/FaceNet.h"
#include "Dataset/Dataset.h"

namespace {

std::filesystem::path get_repo_root() {
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
}

std::filesystem::path get_default_dataset_root(const std::filesystem::path& repo_root) {
    return repo_root / "data" / "data_casia";
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const auto repo_root = get_repo_root();
        const std::string dataset_root = (argc > 1)
            ? argv[1]
            : get_default_dataset_root(repo_root).string();

        const int64_t embedding_dim = 256;
        const double dropout = 0.1;
        const double scale = 30.0;
        const double margin = 0.5;
        const cv::Size image_size{112, 112};

        std::cout << "Loading dataset from: " << dataset_root << std::endl;
        auto raw_dataset = dataset::FaceDataset(dataset_root, image_size);
        const int64_t num_classes = raw_dataset.num_classes();
        std::cout << "Detected classes: " << num_classes << std::endl;

        std::cout << "Loading model..." << std::endl;
        auto model = model::FaceNet(num_classes, embedding_dim, dropout, scale, margin);
        const auto model_path = repo_root / "models" / "model.pt";
        torch::load(model, model_path.string());
        model->eval();

        std::cout << "Model loaded successfully." << std::endl;

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

        const auto bytes = torch::pickle_save(dict);
        const auto output_path = repo_root / "models" / "model_weights.pt";
        std::ofstream fout(output_path.string(), std::ios::out | std::ios::binary);
        if (!fout) {
            throw std::runtime_error("Failed to open output file: " + output_path.string());
        }
        fout.write(bytes.data(), bytes.size());
        fout.close();

        std::cout << "State dict saved to: " << output_path.string() << std::endl;
        std::cout << "Export complete." << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}