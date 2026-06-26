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

std::filesystem::path get_repo_root() {
    std::filesystem::path source_path(__FILE__);
    auto root = source_path.parent_path().parent_path();
    return root;
}

std::string get_model_save_path() {
    return (get_repo_root() / "models" / "model.pt").string();
}

std::string get_weights_path() {
    return (get_repo_root() / "models" / "resnet50_weights.pt").string();
}

std::vector<size_t> make_balanced_batch(
    const std::vector<std::pair<std::string, int64_t>>& samples,
    int P, int K,
    std::mt19937& rng,
    int min_samples_per_class = 4)  
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
        
        if (cls_indices.size() < K) {
            std::cerr << "Warning: Skipping class with insufficient samples: " 
                      << cls_indices.size() << " < " << K << std::endl;
            continue;
        }
        
        std::shuffle(cls_indices.begin(), cls_indices.end(), rng);
        for (int k = 0; k < K; ++k) {
            batch_indices.push_back(cls_indices[k]); 
        }
    }
    
    if (batch_indices.size() < P * K) {
        std::cerr << "Warning: Batch size (" << batch_indices.size() 
                  << ") smaller than P*K (" << P*K << ")" << std::endl;
    }
    
    return batch_indices;
}

int main(int argc, char* argv[]) {
    try {
        const std::filesystem::path repo_root = get_repo_root();
        const std::string dataset_root = (argc > 1)
            ? argv[1]
            : (repo_root / "data" / "data_casia").string();

        const int P = 20;            
        const int K = 10;               
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
        std::cout << "Batch configuration: P=" << P << ", K=" << K 
                  << ", Batch size=" << batch_size << std::endl;

        if (num_classes < 2) {
            throw std::runtime_error("Dataset must have at least 2 classes");
        }

        const auto& all_samples = raw_dataset.samples();

        const size_t total_batches = dataset_size / batch_size;
        std::cout << "Total batches per epoch: " << total_batches << std::endl;

        auto model = model::FaceRecognitionModel(3, embedding_dim, dropout);
        model->to(device);
        model->train();

        torch::optim::Adam optimizer(model->parameters(), 
                                    torch::optim::AdamOptions(1e-4)); 
        auto scheduler = torch::optim::StepLR(optimizer, 5, 0.5);

        const double margin = 0.5;  
        auto triplet_loss = Loss::TripletLoss(margin);
        triplet_loss->to(device);

        std::mt19937 rng(42);

        double best_avg_loss = std::numeric_limits<double>::infinity();
        int no_improve_epochs = 0;
        const int patience = 3; 

        for (int64_t epoch = 1; epoch <= epochs; ++epoch) {
            model->train();
            double   epoch_loss  = 0.0;
            int64_t  batch_index = 0;
            int64_t  skip_count   = 0;
            double   avg_pos_metric_sum = 0.0;
            double   avg_neg_metric_sum = 0.0;
            int64_t  hard_triplets_count = 0;

            for (size_t b = 0; b < total_batches; ++b) {
                std::vector<size_t> indices;
                try {
                    indices = make_balanced_batch(all_samples, P, K, rng, K);
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

                if (indices.size() < batch_size / 2) {
                    std::cerr << "Warning: Batch too small (" << indices.size() 
                              << "), skipping..." << std::endl;
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

                embeddings = torch::nn::functional::normalize(
                    embeddings,
                    torch::nn::functional::NormalizeFuncOptions().p(2).dim(1)
                );

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
                
                torch::nn::utils::clip_grad_norm_(model->parameters(), 5.0);  
                
                optimizer.step();

                epoch_loss += loss_value;
                avg_pos_metric_sum += metrics.avg_pos_metric;
                avg_neg_metric_sum += metrics.avg_neg_metric;
                
                if (loss_value > 1e-6) {
                    hard_triplets_count++;
                }
                
                ++batch_index;

                if (batch_index % 10 == 0) {
                    double margin_gap = metrics.avg_neg_metric - metrics.avg_pos_metric;
                    
                    std::cout << "Epoch [" << epoch << "/" << epochs << "] "
                              << "Batch [" << batch_index << "/" << total_batches << "] "
                              << "Loss: " << loss_value
                              << " | Pos-dist: " << metrics.avg_pos_metric
                              << " | Neg-dist: " << metrics.avg_neg_metric
                              << " | Gap(N-P): " << margin_gap  // ✓ جدید
                              << " | Valid-triplets: " << metrics.num_valid_triplets
                              << " | Zero-triplets: " << metrics.num_zero_loss_triplets << "/" << metrics.num_valid_triplets
                              << " | Emb-norm: " << embeddings.norm(2, 1).mean().item<double>()
                              << std::endl;
                }

                if (batch_index % 1000 == 0) {
                    auto save_path = get_model_save_path();
                    torch::save(model, save_path);
                    std::cout << "Checkpoint saved to: " << save_path << std::endl;
                }
            }

            double avg_loss = batch_index > 0 ? epoch_loss / batch_index : 0.0;
            double avg_pos = batch_index > 0 ? avg_pos_metric_sum / batch_index : 0.0;
            double avg_neg = batch_index > 0 ? avg_neg_metric_sum / batch_index : 0.0;
            
            std::cout << "\n=== Epoch " << epoch << " Summary ===" << std::endl;
            std::cout << "Avg Loss: " << avg_loss << std::endl;
            std::cout << "Avg Pos-dist: " << avg_pos << std::endl;
            std::cout << "Avg Neg-dist: " << avg_neg << std::endl;
            std::cout << "Neg-Pos gap: " << (avg_neg - avg_pos) << std::endl;  
            std::cout << "Batches processed: " << batch_index << std::endl;
            std::cout << "Batches skipped: " << skip_count << std::endl;
            std::cout << "Hard triplets batches: " << hard_triplets_count << std::endl;
            std::cout << "Zero-loss count: " << zero_loss_counter << std::endl;
            std::cout << "NaN-loss count: " << nan_loss_counter << std::endl;

            if (avg_loss < best_avg_loss) {
                best_avg_loss = avg_loss;
                no_improve_epochs = 0;
            } else {
                no_improve_epochs++;
                if (no_improve_epochs >= patience) {
                    std::cout << "No improvement for " << patience 
                              << " epochs. Stopping early." << std::endl;
                    break;
                }
            }

            scheduler.step();
            
            auto save_path = get_model_save_path();
            torch::save(model, save_path);
            std::cout << "Model saved to: " << save_path << std::endl;
            
            if (epoch % 10 == 0) {
                std::string epoch_filename = "model_epoch_" + std::to_string(epoch) + ".pt";
                auto epoch_save_path = get_repo_root() / "models" / epoch_filename;
                
                torch::save(model, epoch_save_path.string());
                std::cout << "Epoch checkpoint saved to: " << epoch_save_path << std::endl;
            }
            
            std::cout << "================================\n" << std::endl;
        }

        std::cout << "Training complete." << std::endl;
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "Fatal Error: " << ex.what() << std::endl;
        return 1;
    }
}