#include "../include/Utils/AccuracyLFW.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <opencv2/opencv.hpp>

namespace {

std::string pad_number(int value) {
    std::ostringstream stream;
    stream << std::setw(4) << std::setfill('0') << value;
    return stream.str();
}

} // namespace

namespace Utils {
namespace {

std::filesystem::path find_lfw_root(const std::filesystem::path& data_dir) {
    std::filesystem::path root = data_dir / "lfw-deepfunneled";
    if (std::filesystem::exists(root)) {
        std::filesystem::path nested = root / "lfw-deepfunneled";
        if (std::filesystem::exists(nested) && std::filesystem::is_directory(nested)) {
            return nested;
        }
        return root;
    }
    throw std::runtime_error("Could not find lfw-deepfunneled under " + data_dir.string());
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string item;
    while (std::getline(stream, item, ',')) {
        fields.push_back(item);
    }
    return fields;
}

} // namespace

AccuracyLFW::AccuracyLFW(const std::filesystem::path& lfw_root,
                         const std::filesystem::path& pairs_file,
                         model::FaceNet& model,
                         torch::Device device)
    : lfw_root_(lfw_root), pairs_file_(pairs_file), model_(model), device_(device) {}

std::filesystem::path AccuracyLFW::resolve_lfw_root() const {
    if (!lfw_root_.empty()) {
        return lfw_root_;
    }
    return find_lfw_root(std::filesystem::path("data") / "data_LFW");
}

std::vector<std::tuple<std::filesystem::path, std::filesystem::path, int>> AccuracyLFW::read_pairs() const {
    std::filesystem::path pairs_path = pairs_file_;
    if (pairs_path.empty()) {
        const auto root = resolve_lfw_root();
        if (std::filesystem::exists(root / "pairs.csv")) {
            pairs_path = root / "pairs.csv";
        } else if (std::filesystem::exists(root.parent_path() / "pairs.csv")) {
            pairs_path = root.parent_path() / "pairs.csv";
        } else if (std::filesystem::exists(root.parent_path().parent_path() / "pairs.csv")) {
            pairs_path = root.parent_path().parent_path() / "pairs.csv";
        }
    }

    if (!std::filesystem::exists(pairs_path)) {
        throw std::runtime_error("Pairs file not found: " + pairs_path.string());
    }

    std::ifstream input(pairs_path);
    if (!input.is_open()) {
        throw std::runtime_error("Unable to open pairs file: " + pairs_path.string());
    }

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path, int>> pairs;
    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        auto fields = split_csv_line(line);
        fields.erase(std::remove_if(fields.begin(), fields.end(), [](const std::string& value) {
            return value.empty();
        }), fields.end());
        if (fields.size() == 3) {
            const std::string& name = fields[0];
            int num1 = std::stoi(fields[1]);
            int num2 = std::stoi(fields[2]);
            pairs.emplace_back(resolve_lfw_root() / name / (name + "_" + pad_number(num1) + ".jpg"),
                               resolve_lfw_root() / name / (name + "_" + pad_number(num2) + ".jpg"),
                               1);
        } else if (fields.size() == 4) {
            const std::string& name1 = fields[0];
            int num1 = std::stoi(fields[1]);
            const std::string& name2 = fields[2];
            int num2 = std::stoi(fields[3]);
            pairs.emplace_back(resolve_lfw_root() / name1 / (name1 + "_" + pad_number(num1) + ".jpg"),
                               resolve_lfw_root() / name2 / (name2 + "_" + pad_number(num2) + ".jpg"),
                               0);
        }
    }
    return pairs;
}

torch::Tensor AccuracyLFW::load_image_tensor(const std::filesystem::path& image_path) const {
    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Unable to read image: " + image_path.string());
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(112, 112));
    image.convertTo(image, CV_32F, 1.0f / 255.0f);

    auto tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kFloat32);
    tensor = tensor.permute({2, 0, 1}).clone();

    const std::vector<float> means = {0.485f, 0.456f, 0.406f};
    const std::vector<float> stds = {0.229f, 0.224f, 0.225f};
    for (int i = 0; i < 3; ++i) {
        tensor[i] = (tensor[i] - means[i]) / stds[i];
    }
    return tensor;
}

std::vector<std::pair<std::filesystem::path, torch::Tensor>> AccuracyLFW::build_embeddings(
    const std::vector<std::filesystem::path>& image_paths,
    int64_t batch_size) const {
    std::vector<std::pair<std::filesystem::path, torch::Tensor>> embeddings;
    std::vector<std::filesystem::path> unique_paths;
    for (const auto& path : image_paths) {
        if (std::find(unique_paths.begin(), unique_paths.end(), path) == unique_paths.end()) {
            unique_paths.push_back(path);
        }
    }

    for (size_t start = 0; start < unique_paths.size(); start += batch_size) {
        std::vector<std::filesystem::path> batch_paths;
        for (size_t i = start; i < std::min<size_t>(start + batch_size, unique_paths.size()); ++i) {
            batch_paths.push_back(unique_paths[i]);
        }

        std::vector<std::filesystem::path> existing_paths;
        std::vector<torch::Tensor> tensors;
        tensors.reserve(batch_paths.size());
        for (const auto& path : batch_paths) {
            if (std::filesystem::exists(path)) {
                existing_paths.push_back(path);
                tensors.push_back(load_image_tensor(path).unsqueeze(0).to(device_));
            }
        }
        if (tensors.empty()) {
            continue;
        }

        torch::Tensor batch = torch::cat(tensors, 0);
        torch::NoGradGuard no_grad;
        model_.to(device_);
        model_->eval();
        auto output = model_->embed(batch);
        auto norms = output.norm(2, 1, true).clamp_min(1e-12);
        output = output / norms;
        auto cpu_output = output.cpu();
        for (size_t i = 0; i < existing_paths.size(); ++i) {
            embeddings.emplace_back(existing_paths[i], cpu_output[i].clone());
        }
    }
    return embeddings;
}

std::pair<std::vector<double>, std::vector<int>> AccuracyLFW::compute_similarities_and_labels() const {
    auto pairs = read_pairs();
    std::vector<std::filesystem::path> image_paths;
    image_paths.reserve(pairs.size() * 2);
    for (const auto& [path1, path2, label] : pairs) {
        image_paths.push_back(path1);
        image_paths.push_back(path2);
    }

    auto embeddings = build_embeddings(image_paths, 16);
    std::vector<double> similarities;
    std::vector<int> labels;
    similarities.reserve(pairs.size());
    labels.reserve(pairs.size());

    std::unordered_map<std::string, torch::Tensor> embedding_map;
    for (const auto& [path, tensor] : embeddings) {
        embedding_map.emplace(path.string(), tensor);
    }

    for (const auto& [path1, path2, label] : pairs) {
        auto it1 = embedding_map.find(path1.string());
        auto it2 = embedding_map.find(path2.string());
        if (it1 == embedding_map.end() || it2 == embedding_map.end()) {
            continue;
        }
        double sim = static_cast<double>((it1->second * it2->second).sum().item<float>());
        similarities.push_back(sim);
        labels.push_back(label);
    }

    return {similarities, labels};
}

double AccuracyLFW::accuracy_for_threshold(const std::vector<double>& similarities,
                                           const std::vector<int>& labels,
                                           double threshold) const {
    if (similarities.empty() || labels.empty()) {
        return 0.0;
    }
    int correct = 0;
    for (size_t i = 0; i < similarities.size(); ++i) {
        bool pred = similarities[i] >= threshold;
        bool label = labels[i] == 1;
        if (pred == label) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(similarities.size());
}

double AccuracyLFW::best_threshold_for(const std::vector<double>& similarities,
                                       const std::vector<int>& labels,
                                       int num_steps) const {
    if (similarities.empty() || labels.empty()) {
        return 0.0;
    }
    double min_sim = *std::min_element(similarities.begin(), similarities.end());
    double max_sim = *std::max_element(similarities.begin(), similarities.end());
    double best_acc = 0.0;
    double best_thr = 0.0;
    for (int step = 0; step <= num_steps; ++step) {
        double threshold = min_sim + (max_sim - min_sim) * (static_cast<double>(step) / static_cast<double>(num_steps));
        double acc = accuracy_for_threshold(similarities, labels, threshold);
        if (acc > best_acc) {
            best_acc = acc;
            best_thr = threshold;
        }
    }
    return best_thr;
}

AccuracyLFW::EvaluationMetrics AccuracyLFW::Evaluate(double threshold, int64_t batch_size, int num_steps) const {
    auto pairs = read_pairs();
    std::vector<std::filesystem::path> image_paths;
    image_paths.reserve(pairs.size() * 2);
    for (const auto& [path1, path2, label] : pairs) {
        image_paths.push_back(path1);
        image_paths.push_back(path2);
    }

    auto embeddings = build_embeddings(image_paths, batch_size);
    std::vector<double> similarities;
    std::vector<int> labels;
    similarities.reserve(pairs.size());
    labels.reserve(pairs.size());

    std::unordered_map<std::string, torch::Tensor> embedding_map;
    for (const auto& [path, tensor] : embeddings) {
        embedding_map.emplace(path.string(), tensor);
    }

    int64_t skipped = 0;
    for (const auto& [path1, path2, label] : pairs) {
        auto it1 = embedding_map.find(path1.string());
        auto it2 = embedding_map.find(path2.string());
        if (it1 == embedding_map.end() || it2 == embedding_map.end()) {
            ++skipped;
            continue;
        }
        double sim = static_cast<double>((it1->second * it2->second).sum().item<float>());
        similarities.push_back(sim);
        labels.push_back(label);
    }

    EvaluationMetrics metrics;
    metrics.total_pairs = static_cast<int64_t>(labels.size());
    metrics.positive_pairs = static_cast<int64_t>(std::count(labels.begin(), labels.end(), 1));
    metrics.negative_pairs = static_cast<int64_t>(std::count(labels.begin(), labels.end(), 0));
    metrics.skipped_pairs = skipped;
    metrics.accuracy = accuracy_for_threshold(similarities, labels, threshold);
    metrics.best_threshold = best_threshold_for(similarities, labels, num_steps);
    return metrics;
}

double AccuracyLFW::ComputeAcc(double threshold) const {
    auto [similarities, labels] = compute_similarities_and_labels();
    return accuracy_for_threshold(similarities, labels, threshold);
}

double AccuracyLFW::BestThreshold(int num_steps) const {
    auto [similarities, labels] = compute_similarities_and_labels();
    return best_threshold_for(similarities, labels, num_steps);
}

double AccuracyLFW::ComputeDistence() const {
    auto [similarities, labels] = compute_similarities_and_labels();
    if (similarities.empty()) {
        return 0.0;
    }
    return std::accumulate(similarities.begin(), similarities.end(), 0.0) / static_cast<double>(similarities.size());
}

} // namespace Utils