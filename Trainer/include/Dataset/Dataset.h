#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

namespace dataset {

class FaceDataset : public torch::data::Dataset<FaceDataset> {
public:
    FaceDataset(const std::string& root_dir, const cv::Size& image_size, bool augment = false);

    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;

    const std::vector<std::string>& classes() const noexcept;
    int64_t num_classes() const noexcept;
    const std::vector<std::pair<std::string, int64_t>>& samples() const noexcept;

    void set_augment(bool augment) noexcept;
    bool is_augment_enabled() const noexcept;

private:
    void scan_directory(const std::string& root_dir);
    bool is_image_file(const std::filesystem::path& path) const;
    torch::Tensor load_image(const std::string& path) const;

    void apply_augmentations(cv::Mat& image) const;
    void random_rotate(cv::Mat& image, float max_angle_deg) const;
    void random_horizontal_flip(cv::Mat& image) const;
    void random_brightness_contrast(cv::Mat& image, float contrast_range, float brightness_range) const;
    void random_translate(cv::Mat& image, float max_shift_ratio) const;

    std::vector<std::pair<std::string, int64_t>> samples_;
    std::vector<std::string> classes_;
    cv::Size image_size_;
    bool augment_ = false; 
};
} // namespace dataset