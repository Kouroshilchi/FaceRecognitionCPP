#pragma once

#include <filesystem>
#include <tuple>
#include <vector>
#include <torch/torch.h>
#include "../Model/FaceNet.h"

namespace Utils
{
    class AccuracyLFW {
    public:
        struct EvaluationMetrics {
            double accuracy = 0.0;
            double positive_accuracy = 0.0;
            double negative_accuracy = 0.0;
            double best_threshold = 0.0;
            int64_t total_pairs = 0;
            int64_t positive_pairs = 0;
            int64_t negative_pairs = 0;
            int64_t skipped_pairs = 0;
        };

        AccuracyLFW(const std::filesystem::path& lfw_root,
                    const std::filesystem::path& pairs_file,
                    model::FaceNet& model,
                    torch::Device device);

        EvaluationMetrics Evaluate(double threshold = 0.7,
                                   int64_t batch_size = 16,
                                   int num_steps = 200) const;
        double ComputeAcc(double threshold = 0.7) const;
        double BestThreshold(int num_steps = 200) const;
        double ComputeDistence() const;

    private:
        std::vector<std::tuple<std::filesystem::path, std::filesystem::path, int>> read_pairs() const;
        std::filesystem::path resolve_lfw_root() const;
        torch::Tensor load_image_tensor(const std::filesystem::path& image_path) const;
        std::vector<std::pair<std::filesystem::path, torch::Tensor>> build_embeddings(
            const std::vector<std::filesystem::path>& image_paths,
            int64_t batch_size) const;
        std::pair<std::vector<double>, std::vector<int>> compute_similarities_and_labels() const;
        double accuracy_for_threshold(const std::vector<double>& similarities,
                                      const std::vector<int>& labels,
                                      double threshold) const;
        double best_threshold_for(const std::vector<double>& similarities,
                                  const std::vector<int>& labels,
                                  int num_steps) const;

        std::filesystem::path lfw_root_;
        std::filesystem::path pairs_file_;
        model::FaceNet& model_;
        torch::Device device_;
    };
}