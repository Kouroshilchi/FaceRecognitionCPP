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
            : "C:\\Users\\kuoro\\Documents\\GitHub\\FaceRecognitionCPP\\data\\cleaned_dataset\\train";

        const int64_t batch_size = 8;
        const int64_t embedding_dim = 256;
        const double dropout = 0.2;
        const int64_t epochs = 5;
        const cv::Size image_size{224, 224};

        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << device << std::endl;

        auto face_dataset = dataset::FaceDataset(dataset_root, image_size);
        const int64_t num_classes = face_dataset.num_classes();
        const size_t dataset_size = face_dataset.size().value(); 
        
        std::cout << "Classes found: " << num_classes << std::endl;
        std::cout << "Dataset size: " << dataset_size << std::endl;

        auto data_loader = torch::data::make_data_loader(
            face_dataset.map(torch::data::transforms::Stack<>()),
            torch::data::samplers::RandomSampler(dataset_size),
            torch::data::DataLoaderOptions().batch_size(batch_size));

        const size_t total_batches = (dataset_size + batch_size - 1) / batch_size;
        std::cout << "Total batches: " << total_batches << std::endl;

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
                              << "Batch [" << batch_index << "/" << total_batches << "] "
                              << "Loss: " << loss.item<double>() << std::endl;
                }
            }

            std::cout << "Epoch " << epoch << " finished. "
                      << "Average loss: " << (epoch_loss / std::max<int64_t>(batch_index, 1))
                      << std::endl;
        }

        std::cout << "Training complete." << std::endl;
        torch::save(model , "models/model.pt");
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}