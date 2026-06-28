#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <torch/optim/schedulers/step_lr.h>
#include "include/Model/Model.h"
#include "include/Model/FaceNet.h"
#include "include/Dataset/Dataset.h"
#include "include/Loss/TripletLoss.h"
#include "include/Loss/ArcFace.h"
#include "include/Utils/AccuracyLFW.h"
#include <random>
#include <map>
#include <cstdlib>
#include <filesystem>

static std::filesystem::path g_repo_root;

void init_repo_root(const char* argv0) {
    const char* env = std::getenv("REPO_ROOT");
    if (env && std::filesystem::exists(env)) {
        g_repo_root = std::filesystem::canonical(env);
        return;
    }
    auto bin = std::filesystem::canonical(argv0);
    g_repo_root = bin.parent_path().parent_path();
}

std::filesystem::path get_repo_root() { return g_repo_root; }
std::string get_model_save_path()  { return (g_repo_root / "models" / "model.pt").string(); }
std::string get_weights_path()     { return (g_repo_root / "models" / "resnet50_weights.pt").string(); }

void evaluate_lfw(model::FaceNet& facenet, torch::Device device) {
    try {
        auto lfw_root   = get_repo_root() / "data" / "data_LFW";
        auto pairs_file = lfw_root / "pairs.csv";

        if (!std::filesystem::exists(lfw_root)) {
            std::cerr << "[LFW] WARNING: LFW root not found: " << lfw_root << std::endl;
            return;
        }
        if (!std::filesystem::exists(pairs_file)) {
            std::cerr << "[LFW] WARNING: pairs.csv not found: " << pairs_file << std::endl;
            return;
        }

        Utils::AccuracyLFW lfw_eval(lfw_root, pairs_file, facenet, device);

        auto lfw_metrics = lfw_eval.Evaluate(/*threshold=*/0.0, /*batch_size=*/16, /*num_steps=*/500);

        std::cout << "[LFW] Total pairs    : " << lfw_metrics.total_pairs    << std::endl;
        std::cout << "[LFW] Positive pairs : " << lfw_metrics.positive_pairs << std::endl;
        std::cout << "[LFW] Negative pairs : " << lfw_metrics.negative_pairs << std::endl;
        std::cout << "[LFW] Skipped pairs  : " << lfw_metrics.skipped_pairs  << std::endl;
        std::cout << "[LFW] Best threshold : " << lfw_metrics.best_threshold  << std::endl;

        if (lfw_metrics.total_pairs == 0) {
            std::cerr << "[LFW] ERROR: 0 pairs evaluated! "
                         "Check image paths inside pairs.csv match actual files." << std::endl;
            return;
        }

        double best_thr = lfw_metrics.best_threshold;
        auto final_metrics = lfw_eval.Evaluate(best_thr, 16, 500);
        std::cout << "[LFW] Accuracy (best thr=" << best_thr << "): "
                  << (final_metrics.accuracy * 100.0) << "%" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[LFW] EXCEPTION during evaluation: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    try {
        init_repo_root(argv[0]);
        std::cout << "Repo root: " << get_repo_root() << std::endl;

        const std::string dataset_root = (argc > 1)
            ? argv[1]
            : (get_repo_root() / "data" / "data_casia").string();

        const int64_t batch_size    = 256;
        const int64_t embedding_dim = 256;
        const double  dropout       = 0.1;
        const int64_t epochs        = 100;
        const cv::Size image_size{112, 112};
        int64_t zero_loss_counter = 0;
        int64_t nan_loss_counter  = 0;

        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << device << std::endl;

        std::cout << "Loading dataset from: " << dataset_root << std::endl;
        auto raw_dataset = dataset::FaceDataset(dataset_root, image_size);
        const int64_t num_classes  = raw_dataset.num_classes();
        const size_t  dataset_size = raw_dataset.size().value();
        int64_t total_batches = (dataset_size + batch_size - 1) / batch_size;

        auto dataloader = torch::data::make_data_loader(
            std::move(raw_dataset),
            torch::data::DataLoaderOptions().batch_size(batch_size)
        );

        auto facenet = model::FaceNet(num_classes, embedding_dim, dropout, 64.0, 0.5);
        facenet->to(device);
        facenet->train();

        const double model_lr = 1e-4;
        std::vector<torch::Tensor> facenet_params;
        for (auto& p : facenet->parameters()) facenet_params.push_back(p);

        torch::optim::Adam optimizer_facenet(facenet_params, torch::optim::AdamOptions(model_lr));
        auto scheduler_facenet = torch::optim::StepLR(optimizer_facenet, 20, 0.70);

        for (int64_t epoch = 1; epoch <= epochs; ++epoch) {
            facenet->train();
            double  epoch_loss          = 0.0;
            int64_t batch_index         = 0;
            int64_t skip_count          = 0;
            double  avg_pos_metric_sum  = 0.0;
            double  avg_neg_metric_sum  = 0.0;
            int64_t hard_triplets_count = 0;

            for (auto& batch : *dataloader) {
                std::vector<torch::Tensor> batch_images;
                std::vector<torch::Tensor> batch_labels;
                batch_images.reserve(batch.size());
                batch_labels.reserve(batch.size());

                for (auto& sample : batch) {
                    batch_images.push_back(sample.data);
                    batch_labels.push_back(sample.target.to(torch::kInt64).squeeze());
                }

                auto inputs = torch::stack(batch_images).to(device);
                auto labels = torch::stack(batch_labels).to(device);

                optimizer_facenet.zero_grad();

                auto metrics    = facenet->forward(inputs, labels);
                auto loss       = metrics.loss;
                double loss_val = loss.item<double>();

                if (std::isnan(loss_val)) {
                    std::cerr << "Warning: NaN loss at batch " << batch_index
                              << " - Skipping update" << std::endl;
                    nan_loss_counter++;
                    continue;
                }

                if (loss_val == 0.0 && batch_index > 0) zero_loss_counter++;

                loss.backward();
                torch::nn::utils::clip_grad_norm_(facenet->parameters(), 5.0);
                optimizer_facenet.step();

                epoch_loss         += loss_val;
                avg_pos_metric_sum += metrics.avg_pos_metric;
                avg_neg_metric_sum += metrics.avg_neg_metric;
                if (loss_val > 1e-6) hard_triplets_count++;
                ++batch_index;

                if (batch_index % 10 == 0) {
                    double margin_gap = metrics.avg_neg_metric - metrics.avg_pos_metric;
                    std::cout << "Epoch [" << epoch << "/" << epochs << "] "
                              << "Batch [" << batch_index << "/" << total_batches << "] "
                              << "Loss: "     << loss_val
                              << " | Pos-dist: " << metrics.avg_pos_metric
                              << " | Neg-dist: " << metrics.avg_neg_metric
                              << " | Gap(N-P): " << margin_gap
                              << std::endl;
                }

                if (batch_index % 1000 == 0) {
                    torch::save(facenet, get_model_save_path());
                    std::cout << "Checkpoint saved to: " << get_model_save_path() << std::endl;
                }
            }

            double avg_loss = batch_index > 0 ? epoch_loss / batch_index : 0.0;
            double avg_pos  = batch_index > 0 ? avg_pos_metric_sum / batch_index : 0.0;
            double avg_neg  = batch_index > 0 ? avg_neg_metric_sum / batch_index : 0.0;

            std::cout << "\n=== Epoch " << epoch << " Summary ===" << std::endl;
            std::cout << "Avg Loss: "     << avg_loss             << std::endl;
            std::cout << "Avg Pos-dist: " << avg_pos              << std::endl;
            std::cout << "Avg Neg-dist: " << avg_neg              << std::endl;
            std::cout << "Neg-Pos gap: "  << (avg_neg - avg_pos)  << std::endl;
            std::cout << "Batches processed: "    << batch_index         << std::endl;
            std::cout << "Batches skipped: "      << skip_count          << std::endl;
            std::cout << "Hard triplets batches: "<< hard_triplets_count << std::endl;
            std::cout << "Zero-loss count: "      << zero_loss_counter   << std::endl;
            std::cout << "NaN-loss count: "       << nan_loss_counter    << std::endl;

            scheduler_facenet.step();

            torch::save(facenet, get_model_save_path());
            std::cout << "Model saved to: " << get_model_save_path() << std::endl;

            if (epoch % 10 == 0) {
                auto epoch_path = get_repo_root() / "models" /
                                  ("model_epoch_" + std::to_string(epoch) + ".pt");
                torch::save(facenet, epoch_path.string());
                std::cout << "Epoch checkpoint saved to: " << epoch_path << std::endl;
            }

            std::cout << "================================\n" << std::endl;

            std::cout << "Evaluating LFW accuracy..." << std::endl;
            evaluate_lfw(facenet, device);
        }

        std::cout << "Training complete." << std::endl;
        std::cout << "Final LFW evaluation:" << std::endl;
        evaluate_lfw(facenet, device);

        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "Fatal Error: " << ex.what() << std::endl;
        return 1;
    }
}