#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <torch/optim/schedulers/step_lr.h>
#include "include/Model/Model.h"
#include "include/Dataset/Dataset.h"
#include "include/Loss/TripletLoss.h"

int main(int argc, char* argv[]) {
    try {
        const std::string dataset_root = (argc > 1)
            ? argv[1]
            : "C:\\Users\\kuoro\\Documents\\GitHub\\FaceRecognitionCPP\\data\\data_casia";

        const int64_t batch_size   = 64;      
        const int64_t embedding_dim = 128;
        const double  dropout       = 0.1;
        const int64_t epochs        = 10;
        const cv::Size image_size{112, 112};
        int64_t zero_loss_counter   = 0;

        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << device << std::endl;

        auto raw_dataset = dataset::FaceDataset(dataset_root, image_size);
        const int64_t num_classes  = raw_dataset.num_classes();
        const size_t  dataset_size = raw_dataset.size().value();

        std::cout << "Classes found: " << num_classes << std::endl;
        std::cout << "Dataset size:  " << dataset_size << std::endl;

        auto face_dataset = raw_dataset.map(torch::data::transforms::Stack<>());
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(face_dataset),
            torch::data::DataLoaderOptions().batch_size(batch_size)
        );

        const size_t total_batches = (dataset_size + batch_size - 1) / batch_size;
        std::cout << "Total batches: " << total_batches << std::endl;

        auto model = model::FaceRecognitionModel(3, embedding_dim, dropout);
        model->to(device);

        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
        // auto scheduler = torch::optim::StepLR(optimizer, 5, 0.5);

        const double margin = 0.7;
        auto triplet_loss = Loss::TripletLoss(margin);

        for (int64_t epoch = 1; epoch <= epochs; ++epoch) {
            model->train();
            double   epoch_loss  = 0.0;
            int64_t  batch_index = 0;

            for (auto& batch : *data_loader) {
                optimizer.zero_grad();

                auto inputs = batch.data.to(device);
                auto labels = batch.target.to(device);

                auto embeddings = model->forward(inputs);
                embeddings = torch::nn::functional::normalize(
                    embeddings,
                    torch::nn::functional::NormalizeFuncOptions().p(2).dim(1)
                );

                auto loss = triplet_loss->forward(embeddings, labels);

                if (loss.item<double>() == 0.0 && batch_index > 0) {
                    zero_loss_counter++;
                }

                loss.backward();
                optimizer.step();

                epoch_loss += loss.item<double>();
                ++batch_index;

                if (batch_index % 10 == 0) {
                    std::cout << "Epoch [" << epoch << "/" << epochs << "] "
                              << "Batch [" << batch_index << "/" << total_batches << "] "
                              << "Loss: " << loss.item<double>()
                              << " | zero-loss: " << zero_loss_counter
                              << std::endl;
                }

                if (batch_index % 1000 == 0) {
                    torch::save(model, "C:\\Users\\kuoro\\Documents\\GitHub\\FaceRecognitionCPP\\models\\model.pt");
                }
            }

            // scheduler.step();
            std::cout << "Epoch " << epoch << " done. "
                      << "Avg loss: " << (epoch_loss / std::max<int64_t>(batch_index, 1))
                      << std::endl;

            torch::save(model, "C:\\Users\\kuoro\\Documents\\GitHub\\FaceRecognitionCPP\\models\\model.pt");
            std::cout << "Model saved." << std::endl;
        }

        std::cout << "Training complete." << std::endl;
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}