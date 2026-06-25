#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <torch/optim/schedulers/step_lr.h>
#include "include/Model/Model.h"
#include "include/Dataset/Dataset.h"
#include "include/Loss/TripletLoss.h"
#include <random>
#include <map>

std::vector<size_t> make_balanced_batch(
    const std::vector<std::pair<std::string, int64_t>>& samples,
    int P, int K,
    std::mt19937& rng)
{
    std::map<int64_t, std::vector<size_t>> label_to_indices;
    for (size_t i = 0; i < samples.size(); ++i)
        label_to_indices[samples[i].second].push_back(i);

    std::vector<int64_t> valid_classes;
    for (auto& [label, indices] : label_to_indices)
        if (indices.size() >= 2) valid_classes.push_back(label);

    std::shuffle(valid_classes.begin(), valid_classes.end(), rng);
    int p_actual = std::min(P, (int)valid_classes.size());

    std::vector<size_t> batch_indices;
    for (int i = 0; i < p_actual; ++i) {
        auto& cls_indices = label_to_indices[valid_classes[i]];
        std::shuffle(cls_indices.begin(), cls_indices.end(), rng);
        for (int k = 0; k < K; ++k)
            batch_indices.push_back(cls_indices[k % cls_indices.size()]);
    }
    return batch_indices;
}

int main(int argc, char* argv[]) {
    try {
        const std::string dataset_root = (argc > 1)
            ? argv[1]
            : "C:\\Users\\kuoro\\Documents\\GitHub\\FaceRecognitionCPP\\data\\data_casia";

        const int P = 32;
        const int K = 8; 
        const int64_t batch_size   = P * K;  
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

        const auto& all_samples = raw_dataset.samples();

        const size_t total_batches = dataset_size / batch_size;
        std::cout << "Total batches: " << total_batches << std::endl;

        auto model = model::FaceRecognitionModel(3, embedding_dim, dropout);
        model->to(device);

        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
        auto scheduler = torch::optim::StepLR(optimizer, 5, 0.5);

        const double margin = 0.5;
        auto triplet_loss = Loss::TripletLoss(margin);

        std::mt19937 rng(42);

        for (int64_t epoch = 1; epoch <= epochs; ++epoch) {
            model->train();
            double   epoch_loss  = 0.0;
            int64_t  batch_index = 0;

            for (size_t b = 0; b < total_batches; ++b) {
                auto indices = make_balanced_batch(all_samples, P, K, rng);

                std::vector<torch::Tensor> images, labels_vec;
                for (size_t idx : indices) {
                    auto ex = raw_dataset.get(idx);
                    images.push_back(ex.data);
                    labels_vec.push_back(ex.target);
                }
                auto inputs = torch::stack(images).to(device);
                auto labels = torch::stack(labels_vec).to(device);

                optimizer.zero_grad();
                auto embeddings = model->forward(inputs);
                embeddings = torch::nn::functional::normalize(
                    embeddings,
                    torch::nn::functional::NormalizeFuncOptions().p(2).dim(1)
                );

                auto metrics = triplet_loss->forward(embeddings, labels);
                auto loss = metrics.loss;

                if (loss.item<double>() == 0.0 && batch_index > 0)
                    zero_loss_counter++;

                loss.backward();
                optimizer.step();

                epoch_loss += loss.item<double>();
                ++batch_index;

                if (batch_index % 1 == 0) {
                    std::cout << "Epoch [" << epoch << "/" << epochs << "] "
                              << "Batch [" << batch_index << "/" << total_batches << "] "
                              << "Loss: " << loss.item<double>()
                              << " | Pos-dist: " << metrics.avg_pos_dist
                              << " | Neg-dist: " << metrics.avg_neg_dist
                              << " | zero-loss: " << zero_loss_counter
                              << std::endl;
                }

                if (batch_index % 1000 == 0) {
                    torch::save(model, "C:\\Users\\kuoro\\Documents\\GitHub\\FaceRecognitionCPP\\models\\model.pt");
                }
            }

            scheduler.step();
            std::cout << "Epoch " << epoch << " done. Avg loss: "
                      << (epoch_loss / std::max<int64_t>(batch_index, 1)) << std::endl;
            torch::save(model, "C:\\Users\\kuoro\\Documents\\GitHub\\FaceRecognitionCPP\\models\\model.pt");
        }

        std::cout << "Training complete." << std::endl;
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}