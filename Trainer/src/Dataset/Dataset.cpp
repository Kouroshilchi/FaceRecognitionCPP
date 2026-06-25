#include "../include/Dataset/Dataset.h"

namespace fs = std::filesystem;

namespace dataset {

FaceDataset::FaceDataset(const std::string& root_dir, const cv::Size& image_size)
    : image_size_(image_size) {
    scan_directory(root_dir);
    
    if (num_classes() < 2) {
        throw std::runtime_error("Dataset must have at least 2 classes for triplet loss");
    }
}

void FaceDataset::scan_directory(const std::string& root_dir) {
    fs::path root(root_dir);
    if (!fs::exists(root) || !fs::is_directory(root)) {
        throw std::runtime_error("Dataset root path does not exist or is not a directory: " + root_dir);
    }

    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_directory()) {
            continue;
        }

        classes_.push_back(entry.path().filename().string());
    }

    std::sort(classes_.begin(), classes_.end());
    classes_.erase(std::unique(classes_.begin(), classes_.end()), classes_.end());

    std::map<int64_t, int> class_sample_count;
    
    for (size_t label = 0; label < classes_.size(); ++label) {
        const auto class_name = classes_[label];
        fs::path class_dir = root / class_name;

        for (const auto& file_entry : fs::directory_iterator(class_dir)) {
            if (!file_entry.is_regular_file()) {
                continue;
            }

            if (!is_image_file(file_entry.path())) {
                continue;
            }

            samples_.emplace_back(file_entry.path().string(), static_cast<int64_t>(label));
            class_sample_count[label]++;
        }
    }

    if (samples_.empty()) {
        throw std::runtime_error("No image files found in dataset root: " + root_dir);
    }
    
    for (const auto& [label, count] : class_sample_count) {
        if (count < 2) {
            throw std::runtime_error("Class " + std::to_string(label) + 
                                   " has only " + std::to_string(count) + 
                                   " samples. Each class needs at least 2 samples for triplet loss.");
        }
    }
}

bool FaceDataset::is_image_file(const fs::path& path) const {
    static const std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return std::find(extensions.begin(), extensions.end(), ext) != extensions.end();
}

torch::Tensor FaceDataset::load_image(const std::string& path) const {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Unable to read image: " + path);
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, image_size_);
    
    image.convertTo(image, CV_32F, 1.0 / 255.0);

    auto tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kFloat32);
    tensor = tensor.permute({2, 0, 1}).clone();

    std::vector<float> means = {0.485f, 0.456f, 0.406f};
    std::vector<float> stds = {0.229f, 0.224f, 0.225f};

    for (int i = 0; i < 3; ++i) {
        tensor[i] = (tensor[i] - means[i]) / stds[i];
    }

    return tensor;
}

torch::data::Example<> FaceDataset::get(size_t index) {
    if (index >= samples_.size()) {
        throw std::out_of_range("Index out of range in FaceDataset: " + std::to_string(index));
    }

    const auto& [path, label] = samples_[index];
    torch::Tensor image_tensor = load_image(path);
    torch::Tensor label_tensor = torch::tensor(label, torch::kInt64);

    return {image_tensor, label_tensor};
}

torch::optional<size_t> FaceDataset::size() const {
    return samples_.size();
}

const std::vector<std::string>& FaceDataset::classes() const noexcept {
    return classes_;
}

int64_t FaceDataset::num_classes() const noexcept {
    return static_cast<int64_t>(classes_.size());
}

}