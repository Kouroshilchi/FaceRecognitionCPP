#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "include/model/Model.h"
#include "include/model/ArcFace.h"
#include "include/dataset/Dataset.h"

int main(int argc, char* argv[]) {
    try {
        const std::string dataset_root = (argc > 1)
            ? argv[1]
            : "../data/cleaned_dataset/train";

        const int64_t batch_size = 16;
        const int64_t embedding_dim = 512;
        const double dropout = 0.5;
        const int64_t epochs = 2;
        const cv::Size image_size{224, 224};

        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << device << std::endl;

        auto face_dataset = dataset::FaceDataset(dataset_root, image_size);
        const int64_t num_classes = face_dataset.num_classes();
        std::cout << "Classes found: " << num_classes << std::endl;

        auto data_loader = torch::data::make_data_loader(
            face_dataset.map(torch::data::transforms::Stack<>()),
            torch::data::DataLoaderOptions().batch_size(batch_size).shuffle(true));

        auto model = model::FaceRecognitionModel(3, embedding_dim, dropout);
        auto arcface = model::ArcFace(embedding_dim, num_classes, 64.0, 0.5, false);

        model->to(device);
        arcface->to(device);

        std::vector<torch::Tensor> parameters;
        for (auto& p : model->parameters()) {
            if (p.requires_grad()) {
                parameters.push_back(p);
            }
        }
        for (auto& p : arcface->parameters()) {
            if (p.requires_grad()) {
                parameters.push_back(p);
            }
        }

        torch::optim::Adam optimizer(parameters, torch::optim::AdamOptions(1e-4));
        torch::nn::CrossEntropyLoss criterion;

        for (int64_t epoch = 1; epoch <= epochs; ++epoch) {
            model->train();
            arcface->train();
            double epoch_loss = 0.0;
            int64_t batch_index = 0;

            for (auto& batch : *data_loader) {
                auto images = batch.data.to(device);
                auto labels = batch.target.to(device).to(torch::kInt64);

                optimizer.zero_grad();
                auto embeddings = model->forward(images);
                auto logits = arcface->forward(embeddings, labels);
                auto loss = criterion(logits, labels);
                loss.backward();
                optimizer.step();

                epoch_loss += loss.item<double>();
                ++batch_index;

                if (batch_index % 10 == 0) {
                    std::cout << "Epoch [" << epoch << "/" << epochs << "] "
                              << "Batch " << batch_index << " "
                              << "Loss: " << loss.item<double>() << std::endl;
                }
            }

            std::cout << "Epoch " << epoch << " finished. "
                      << "Average loss: " << (epoch_loss / std::max<int64_t>(batch_index, 1))
                      << std::endl;
        }

        std::cout << "Training complete." << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}
