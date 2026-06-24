#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <torch/optim/schedulers/step_lr.h>
#include "include/model/Model.h"
#include "include/dataset/TripletDataset.h"
#include "include/model/TripletLoss.h"
#include "include/model/HardMining.h"

// struct TripletCollate {
//     dataset::TripletBatch operator()(std::vector<dataset::TripletSample> samples) {
//         std::vector<torch::Tensor> anchors, positives, negatives, labels;
//         for (auto& s : samples) {
//             anchors.push_back(s.anchor);
//             positives.push_back(s.positive);
//             negatives.push_back(s.negative);
//             labels.push_back(s.label);
//         }
//         return {
//             torch::stack(anchors),
//             torch::stack(positives),
//             torch::stack(negatives),
//             torch::stack(labels)
//         };
//     }
// };

int main(int argc, char* argv[]) {
    try {
        const std::string dataset_root = (argc > 1)
            ? argv[1]
            : "C:\\Users\\kuoro\\Documents\\GitHub\\FaceRecognitionCPP\\data\\extracted_images";

        const int64_t batch_size   = 64;      
        const int64_t embedding_dim = 128;
        const double  dropout       = 0.1;
        const int64_t epochs        = 10;
        const cv::Size image_size{112, 112};
        int64_t zero_loss_counter   = 0;

        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << device << std::endl;

        auto face_dataset = dataset::TripletDataset(dataset_root, image_size);
        const int64_t num_classes  = face_dataset.num_classes();
        const size_t  dataset_size = face_dataset.size().value();

        std::cout << "Classes found: " << num_classes << std::endl;
        std::cout << "Dataset size:  " << dataset_size << std::endl;

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

        for (int64_t epoch = 1; epoch <= epochs; ++epoch) {
            model->train();
            double   epoch_loss  = 0.0;
            int64_t  batch_index = 0;

            for (auto& batch : *data_loader) {
                std::vector<torch::Tensor> a_list, p_list, n_list, l_list;
                for (auto& s : batch) {
                    a_list.push_back(s.anchor);
                    p_list.push_back(s.positive);
                    n_list.push_back(s.negative);
                    l_list.push_back(s.label);
                }

                auto anchors   = torch::stack(a_list).to(device);
                auto positives = torch::stack(p_list).to(device);
                auto negatives = torch::stack(n_list).to(device);
                auto labels = torch::stack(l_list).to(device);

                optimizer.zero_grad();

                auto emb_a = model->forward(anchors);
                auto emb_p = model->forward(positives);
                auto emb_n = model->forward(negatives);

                emb_a = torch::nn::functional::normalize(emb_a,
                    torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
                emb_p = torch::nn::functional::normalize(emb_p,
                    torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
                emb_n = torch::nn::functional::normalize(emb_n,
                    torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
                
                auto dist_ap = (emb_a - emb_p).pow(2).sum(1);
                auto dist_an = (emb_a - emb_n).pow(2).sum(1);      
                auto all_embeddings = torch::cat({emb_a, emb_p, emb_n}, 0);
                auto all_labels = torch::cat({labels, labels, labels}, 0);
                // auto dist_ap = (emb_a - emb_p).pow(2).sum(1);
                // auto dist_an = (emb_a - emb_n).pow(2).sum(1);
                // auto loss = torch::relu(dist_ap - dist_an + margin).mean();

                auto hard_triplets = HardMining::select_hard_triplets(
                    all_embeddings, 
                    all_labels
                );

                torch::Tensor loss = torch::zeros({}, device);
                if (!hard_triplets.empty()) {
                    for (auto [a, p, n] : hard_triplets) {
                        auto anchor_emb   = all_embeddings[a];
                        auto positive_emb = all_embeddings[p];
                        auto negative_emb = all_embeddings[n];
                        auto triplet_loss = torch::relu(
                            (anchor_emb - positive_emb).pow(2).sum() - 
                            (anchor_emb - negative_emb).pow(2).sum() + 0.2
                        );
                        loss += triplet_loss;
                    }
                    loss = loss / static_cast<double>(hard_triplets.size());
                }

                if (loss.item<double>() == 0.0 && batch_index > 0) {
                    zero_loss_counter++;
                    // ++batch_index;
                    // continue;
                }

                loss.backward();
                // torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
                optimizer.step();

                epoch_loss += loss.item<double>();
                ++batch_index;

                if (batch_index % 1 == 0) {
                    std::cout << "Epoch [" << epoch << "/" << epochs << "] "
                              << "Batch [" << batch_index << "/" << total_batches << "] "
                              << "Loss: " << loss.item<double>()
                              << " | d(a,p): " << dist_ap.mean().item<double>()
                              << " | d(a,n): " << dist_an.mean().item<double>()
                              << "zero-loss" << zero_loss_counter
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