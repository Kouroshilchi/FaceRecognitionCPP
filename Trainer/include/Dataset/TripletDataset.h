#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <random>
#include <algorithm>

namespace dataset {

struct TripletSample {
    torch::Tensor anchor;
    torch::Tensor positive;
    torch::Tensor negative;
    torch::Tensor label;  
};

struct TripletBatch {
    torch::Tensor anchors;    // [B, C, H, W]
    torch::Tensor positives;  // [B, C, H, W]
    torch::Tensor negatives;  // [B, C, H, W]
    torch::Tensor labels;     // [B]
};

class TripletDataset : public torch::data::Dataset<TripletDataset, TripletSample> {
public:
    TripletDataset(const std::string& root_dir, const cv::Size& image_size);

    TripletSample get(size_t index) override;
    torch::optional<size_t> size() const override;

    int64_t num_classes() const noexcept;
    const std::vector<std::string>& classes() const noexcept;

    std::vector<int64_t> get_all_labels() const;

private:
    cv::Size image_size_;
    std::vector<std::string> classes_;

    std::map<int64_t, std::vector<std::string>> class_to_paths_;

    std::vector<std::pair<std::string, int64_t>> samples_;

    mutable std::mt19937 rng_;

    void scan_directory(const std::string& root_dir);
    bool is_image_file(const std::filesystem::path& path) const;
    torch::Tensor load_image(const std::string& path) const;

    std::string sample_positive(int64_t label, const std::string& anchor_path) const;

    std::string sample_negative(int64_t anchor_label) const;
};

} // namespace dataset