#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <torch/optim/schedulers/step_lr.h>
#include "include/model/Model.h"
#include "include/dataset/Dataset.h" 
#include "include/model/TripletLoss.h"

static constexpr int64_t BATCH_SIZE    = 128;   
static constexpr int64_t EMBEDDING_DIM = 128;
static constexpr double  DROPOUT       = 0.1;
static constexpr int64_t EPOCHS        = 30;
static constexpr double  LR            = 1e-3;
static constexpr double  MARGIN        = 0.3; 
static constexpr int64_t LOG_EVERY     = 1;  
static constexpr int64_t SAVE_EVERY    = 500;

static const cv::Size IMAGE_SIZE{112, 112};
static const std::string MODEL_PATH =
    "/media/kourosh/kuorosh/FaceRecognitionCPP/models/model.pt";

inline torch::Tensor l2_normalize(const torch::Tensor& t) {
    return torch::nn::functional::normalize(
        t, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
}


int main(int argc, char* argv[]) {
    try {
        const std::string dataset_root = (argc > 1)
            ? argv[1]
            : "/media/kourosh/kuorosh/FaceRecognitionCPP/data/extracted_images";

        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << device << "\n";

        auto dataset = dataset::FaceDataset(dataset_root, IMAGE_SIZE)
                           .map(torch::data::transforms::Stack<>());

        const size_t dataset_size  = dataset.size().value();
        const size_t total_batches = (dataset_size + BATCH_SIZE - 1) / BATCH_SIZE;

        std::cout << "Dataset size : " << dataset_size  << "\n";
        std::cout << "Total batches: " << total_batches << "\n";

        auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(dataset),
            torch::data::DataLoaderOptions()
                .batch_size(BATCH_SIZE)
                .workers(4)
                .drop_last(true)  
        );

        auto model = model::FaceRecognitionModel(3, EMBEDDING_DIM, DROPOUT);
        model->to(device);

        if (std::filesystem::exists(MODEL_PATH)) {
            try {
                torch::load(model, MODEL_PATH);
                std::cout << "Loaded checkpoint from: " << MODEL_PATH << "\n";
            } catch (...) {
                std::cout << "Could not load checkpoint, starting fresh.\n";
            }
        }

        auto triplet_loss = loss::TripletLoss(MARGIN);

        torch::optim::Adam optimizer(
            model->parameters(),
            torch::optim::AdamOptions(LR).weight_decay(1e-4)
        );
        auto scheduler = torch::optim::StepLR(optimizer, /*step_size=*/10, /*gamma=*/0.5);

        for (int64_t epoch = 1; epoch <= EPOCHS; ++epoch) {
            model->train();

            double  epoch_loss       = 0.0;
            int64_t batch_index      = 0;
            int64_t non_zero_batches = 0; 

            for (auto& batch : *data_loader) {
                auto images = batch.data.to(device);   
                auto labels = batch.target.to(device); 

                optimizer.zero_grad();

                auto embeddings = model->forward(images); 

                embeddings = l2_normalize(embeddings);

                auto loss = triplet_loss->forward(embeddings, labels);

                double loss_val = loss.item<double>();

                loss.backward();
                torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
                optimizer.step();

                epoch_loss += loss_val;
                if (loss_val > 1e-8) ++non_zero_batches;
                ++batch_index;

                if (batch_index % LOG_EVERY == 0) {
                    torch::NoGradGuard no_grad;
                    auto dist_matrix = torch::cdist(embeddings, embeddings);
                    auto label_col   = labels.view({-1, 1});
                    auto pos_mask    = label_col.eq(label_col.t())
                                           .logical_and(torch::eye(BATCH_SIZE,
                                               torch::TensorOptions().device(device).dtype(torch::kBool)).logical_not());
                    auto neg_mask    = label_col.ne(label_col.t());

                    double avg_pos_dist = dist_matrix.masked_select(pos_mask).mean().item<double>();
                    double avg_neg_dist = dist_matrix.masked_select(neg_mask).mean().item<double>();

                    std::cout
                        << "Epoch [" << epoch << "/" << EPOCHS << "] "
                        << "Batch [" << batch_index << "/" << total_batches << "] "
                        << "Loss: "     << loss_val
                        << " | d+(avg): " << avg_pos_dist
                        << " | d-(avg): " << avg_neg_dist
                        << " | non-zero batches: " << non_zero_batches << "/" << batch_index
                        << "\n";
                }

                if (batch_index % SAVE_EVERY == 0) {
                    torch::save(model, MODEL_PATH);
                    std::cout << "  [checkpoint saved]\n";
                }
            }

            scheduler.step();

            double avg_loss = epoch_loss / std::max<int64_t>(batch_index, 1);
            std::cout
                << "\n=== Epoch " << epoch << " done | "
                << "Avg loss: " << avg_loss << " | "
                << "Non-zero batches: " << non_zero_batches << "/" << batch_index
                << " ===\n\n";

            torch::save(model, MODEL_PATH);
            std::cout << "Model saved to: " << MODEL_PATH << "\n";
        }

        std::cout << "Training complete.\n";
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}