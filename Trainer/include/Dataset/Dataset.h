#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace dataset {

struct FaceDataset : torch::data::datasets::Dataset<FaceDataset> {
    FaceDataset(const std::string& root_dir, const cv::Size& image_size = {224, 224});

    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;

    const std::vector<std::string>& classes() const noexcept;
    int64_t num_classes() const noexcept;
    const std::vector<std::pair<std::string, int64_t>>& samples() const noexcept {
        return samples_;
    }
    

private:
    void scan_directory(const std::string& root_dir);
    torch::Tensor load_image(const std::string& path) const;
    bool is_image_file(const std::filesystem::path& path) const;

    std::vector<std::pair<std::string, int64_t>> samples_;
    std::vector<std::string> classes_;
    cv::Size image_size_;
};

} // namespace dataset
