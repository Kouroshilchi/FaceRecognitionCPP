#include "../include/Dataset/TripletDataset.h"

namespace fs = std::filesystem;

namespace dataset {

TripletDataset::TripletDataset(const std::string& root_dir, const cv::Size& image_size)
    : image_size_(image_size), rng_(std::random_device{}()) {
    scan_directory(root_dir);
}

void TripletDataset::scan_directory(const std::string& root_dir) {
    fs::path root(root_dir);
    if (!fs::exists(root) || !fs::is_directory(root)) {
        throw std::runtime_error("Dataset root does not exist or is not a directory: " + root_dir);
    }

    for (const auto& entry : fs::directory_iterator(root)) {
        if (entry.is_directory()) {
            classes_.push_back(entry.path().filename().string());
        }
    }
    std::sort(classes_.begin(), classes_.end());
    classes_.erase(std::unique(classes_.begin(), classes_.end()), classes_.end());

    for (size_t label = 0; label < classes_.size(); ++label) {
        const auto& class_name = classes_[label];
        fs::path class_dir = root / class_name;

        std::vector<std::string> paths;
        for (const auto& file_entry : fs::directory_iterator(class_dir)) {
            if (!file_entry.is_regular_file()) continue;
            if (!is_image_file(file_entry.path())) continue;
            paths.push_back(file_entry.path().string());
            samples_.emplace_back(file_entry.path().string(), static_cast<int64_t>(label));
        }

        if (paths.size() >= 2) {
            class_to_paths_[static_cast<int64_t>(label)] = std::move(paths);
        }
    }

    if (samples_.empty()) {
        throw std::runtime_error("No image files found in: " + root_dir);
    }

    samples_.erase(
        std::remove_if(samples_.begin(), samples_.end(),
            [&](const std::pair<std::string, int64_t>& s) {
                return class_to_paths_.find(s.second) == class_to_paths_.end();
            }),
        samples_.end()
    );

    if (samples_.empty()) {
        throw std::runtime_error("No classes with >= 2 images found. Triplet loss needs at least 2 images per class.");
    }
}

bool TripletDataset::is_image_file(const fs::path& path) const {
    static const std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return std::find(extensions.begin(), extensions.end(), ext) != extensions.end();
}

torch::Tensor TripletDataset::load_image(const std::string& path) const {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Unable to read image: " + path);
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, image_size_);
    image.convertTo(image, CV_32F, 1.0 / 255.0);

    auto tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kFloat32);
    return tensor.permute({2, 0, 1}).clone();
}

std::string TripletDataset::sample_positive(int64_t label, const std::string& anchor_path) const {
    const auto& paths = class_to_paths_.at(label);
    std::vector<const std::string*> candidates;
    for (const auto& p : paths) {
        if (p != anchor_path) candidates.push_back(&p);
    }
    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    return *candidates[dist(rng_)];
}

std::string TripletDataset::sample_negative(int64_t anchor_label) const {
    std::uniform_int_distribution<size_t> class_dist(0, class_to_paths_.size() - 1);
    
    int64_t neg_label;
    do {
        auto it = class_to_paths_.begin();
        std::advance(it, class_dist(rng_));
        neg_label = it->first;
    } while (neg_label == anchor_label);

    const auto& neg_paths = class_to_paths_.at(neg_label);
    std::uniform_int_distribution<size_t> img_dist(0, neg_paths.size() - 1);
    return neg_paths[img_dist(rng_)];
}

TripletSample TripletDataset::get(size_t index) {
    if (index >= samples_.size()) {
        throw std::out_of_range("Index out of range: " + std::to_string(index));
    }

    const auto& [anchor_path, label] = samples_[index];

    std::string positive_path = sample_positive(label, anchor_path);
    std::string negative_path = sample_negative(label);

    TripletSample sample;
    sample.anchor   = load_image(anchor_path);
    sample.positive = load_image(positive_path);
    sample.negative = load_image(negative_path);
    sample.label    = torch::tensor(label, torch::kInt64);

    return sample;
}

torch::optional<size_t> TripletDataset::size() const {
    return samples_.size();
}

int64_t TripletDataset::num_classes() const noexcept {
    return static_cast<int64_t>(classes_.size());
}

const std::vector<std::string>& TripletDataset::classes() const noexcept {
    return classes_;
}

std::vector<int64_t> TripletDataset::get_all_labels() const {
    std::vector<int64_t> labels;
    labels.reserve(samples_.size());
    for (const auto& [path, label] : samples_) {
        labels.push_back(label);
    }
    return labels;
}

} // namespace dataset