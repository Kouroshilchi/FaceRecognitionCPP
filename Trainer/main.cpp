#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <torch/optim/schedulers/step_lr.h>
#include "include/Model/Model.h"
#include "include/Dataset/Dataset.h"
#include "include/Loss/TripletLoss.h"
#include "include/Loss/ArcFace.h"
#include <random>
#include <map>
#include <cstdlib>
#include <filesystem>

std::string get_home_dir() {
    #ifdef _WIN32
        const char* home = std::getenv("USERPROFILE");
    #else
        const char* home = std::getenv("HOME");
    #endif
    return home ? home : ".";
}

std::string get_model_save_path() {
    return "C:\\Users\\kuoro\\Documents\\GitHub\\FaceRecognitionCPP\\models\\model.pt";
}

std::string get_weights_path() {
    return "C:\\Users\\kuoro\\Documents\\GitHub\\FaceRecognitionCPP\\models\\resnet50_weights.pt";
}

std::vector<size_t> make_balanced_batch(
    const std::vector<std::pair<std::string, int64_t>>& samples,
    int P, int K,
    std::mt19937& rng)
{
    std::map<int64_t, std::vector<size_t>> label_to_indices;
    for (size_t i = 0; i < samples.size(); ++i)
        label_to_indices[samples[i].second].push_back(i);

    std::vector<int64_t> valid_classes;
    for (auto& [label, indices] : label_to_indices) {
        if (indices.size() >= K) {  
            valid_classes.push_back(label);
        }
    }

    if (valid_classes.empty()) {
        throw std::runtime_error("No class has enough samples for batch size P*K");
    }

    std::shuffle(valid_classes.begin(), valid_classes.end(), rng);
    int p_actual = std::min(P, (int)valid_classes.size());

    std::vector<size_t> batch_indices;
    for (int i = 0; i < p_actual; ++i) {
        auto& cls_indices = label_to_indices[valid_classes[i]];
        std::shuffle(cls_indices.begin(), cls_indices.end(), rng);
        for (int k = 0; k < K; ++k) {
            batch_indices.push_back(cls_indices[k % cls_indices.size()]);
        }
    }
    return batch_indices;
}

int main(int argc, char* argv[]) {
    try {
        const std::string dataset_root = (argc > 1)
            ? argv[1]
            : "C:\\Users\\kuoro\\Documents\\GitHub\\FaceRecognitionCPP\\data\\data_casia";

        const int P = 32;
        const int K = 4; 
        const int64_t batch_size   = P * K;          
        const int64_t embedding_dim = 128;
        const double  dropout       = 0.1;
        const int64_t epochs        = 10;
        const cv::Size image_size{112, 112};
        int64_t zero_loss_counter   = 0;
        int64_t nan_loss_counter    = 0;

        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << device << std::endl;

        std::cout << "Loading dataset from: " << dataset_root << std::endl;
        auto raw_dataset = dataset::FaceDataset(dataset_root, image_size);
        const int64_t num_classes  = raw_dataset.num_classes();
        const size_t  dataset_size = raw_dataset.size().value();

        std::cout << "Classes found: " << num_classes << std::endl;
        std::cout << "Dataset size:  " << dataset_size << std::endl;

        if (num_classes < 2) {
            throw std::runtime_error("Dataset must have at least 2 classes");
        }

        const auto& all_samples = raw_dataset.samples();

        const size_t total_batches = dataset_size / batch_size;
        std::cout << "Total batches per epoch: " << total_batches << std::endl;

        auto model = model::FaceRecognitionModel(3, embedding_dim, dropout);
        model->to(device);
        model->train();

        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
        auto scheduler = torch::optim::StepLR(optimizer, 5, 0.5);

        const double margin = 0.0;
        // auto triplet_loss = Loss::TripletLoss(margin);
        auto triplet_loss = Loss::ArcFace(
            num_classes ,
            embedding_dim,
            64, 
            0.5
        );
        triplet_loss->to(device);

        std::mt19937 rng(42);

        for (int64_t epoch = 1; epoch <= epochs; ++epoch) {
            model->train();
            double   epoch_loss  = 0.0;
            int64_t  batch_index = 0;
            int64_t  skip_count   = 0;

            for (size_t b = 0; b < total_batches; ++b) {
                std::vector<size_t> indices;
                try {
                    indices = make_balanced_batch(all_samples, P, K, rng);
                } catch (const std::exception& e) {
                    std::cerr << "Error generating batch: " << e.what() << std::endl;
                    skip_count++;
                    continue;
                }

                if (indices.empty()) {
                    std::cerr << "Warning: Empty batch generated, skipping..." << std::endl;
                    skip_count++;
                    continue;
                }

                std::vector<torch::Tensor> images, labels_vec;
                try {
                    for (size_t idx : indices) {
                        auto ex = raw_dataset.get(idx);
                        images.push_back(ex.data);
                        labels_vec.push_back(ex.target);
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error loading batch data: " << e.what() << std::endl;
                    skip_count++;
                    continue;
                }

                if (images.empty() || labels_vec.empty()) {
                    std::cerr << "Warning: No valid data in batch, skipping..." << std::endl;
                    skip_count++;
                    continue;
                }

                auto inputs = torch::stack(images).to(device);
                auto labels = torch::stack(labels_vec).to(device);

                optimizer.zero_grad();
                
                auto embeddings = model->forward(inputs);
                
                // embeddings = torch::nn::functional::normalize(
                //     embeddings,
                //     torch::nn::functional::NormalizeFuncOptions().p(2).dim(1)
                // );

                auto metrics = triplet_loss->forward(embeddings, labels);
                auto loss = metrics.loss;

                double loss_value = loss.item<double>();
                
                if (std::isnan(loss_value)) {
                    std::cerr << "Warning: NaN loss detected at batch " << batch_index 
                              << " - Skipping update" << std::endl;
                    nan_loss_counter++;
                    continue;
                }

                if (loss_value == 0.0 && batch_index > 0) {
                    zero_loss_counter++;
                }

                loss.backward();
                
                torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
                
                optimizer.step();

                epoch_loss += loss_value;
                ++batch_index;

                if (batch_index % 10 == 0) {
                    std::cout << "Epoch [" << epoch << "/" << epochs << "] "
                              << "Batch [" << batch_index << "/" << total_batches << "] "
                              << "Loss: " << loss_value
                              << " | Pos-dist: " << metrics.avg_pos_cos
                              << " | Neg-dist: " << metrics.avg_neg_cos
                              << " | Zero-loss: " << zero_loss_counter
                              << " | NaN-loss: " << nan_loss_counter
                              << " | Embedding norm mean: " << embeddings.norm(2, 1).mean().item<double>()
                              << std::endl;
                }

                if (batch_index % 1000 == 0) {
                    auto save_path = get_model_save_path();
                    torch::save(model, save_path);
                    std::cout << "Checkpoint saved to: " << save_path << std::endl;
                }
            }

            double avg_loss = batch_index > 0 ? epoch_loss / batch_index : 0.0;
            std::cout << "\n=== Epoch " << epoch << " Summary ===" << std::endl;
            std::cout << "Avg Loss: " << avg_loss << std::endl;
            std::cout << "Batches processed: " << batch_index << std::endl;
            std::cout << "Batches skipped: " << skip_count << std::endl;
            std::cout << "Zero-loss count: " << zero_loss_counter << std::endl;
            std::cout << "NaN-loss count: " << nan_loss_counter << std::endl;

            scheduler.step();
            
            auto save_path = get_model_save_path();
            torch::save(model, save_path);
            std::cout << "Model saved to: " << save_path << std::endl;
            std::cout << "================================\n" << std::endl;
        }

        std::cout << "Training complete." << std::endl;
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "Fatal Error: " << ex.what() << std::endl;
        return 1;
    }
}