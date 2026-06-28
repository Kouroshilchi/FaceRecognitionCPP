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

std::string trim(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

} 

namespace Utils {
namespace {

std::filesystem::path find_lfw_root(const std::filesystem::path& data_dir) {
    {
        auto p = data_dir / "lfw-deepfunneled" / "lfw-deepfunneled";
        if (std::filesystem::exists(p) && std::filesystem::is_directory(p)) return p;
    }
    {
        auto p = data_dir / "lfw-deepfunneled";
        if (std::filesystem::exists(p) && std::filesystem::is_directory(p)) return p;
    }
    if (std::filesystem::exists(data_dir) && std::filesystem::is_directory(data_dir)) {
        return data_dir;
    }
    throw std::runtime_error("Could not find LFW image root under: " + data_dir.string());
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string item;
    while (std::getline(stream, item, ',')) fields.push_back(item);
    return fields;
}

} 

AccuracyLFW::AccuracyLFW(const std::filesystem::path& lfw_root,
                         const std::filesystem::path& pairs_file,
                         model::FaceNet& model,
                         torch::Device device)
    : lfw_root_(lfw_root), pairs_file_(pairs_file), model_(model), device_(device) {}

std::filesystem::path AccuracyLFW::resolve_lfw_root() const {
    if (!lfw_root_.empty() && std::filesystem::exists(lfw_root_)) {
        return find_lfw_root(lfw_root_);
    }
    return find_lfw_root(std::filesystem::path("data") / "data_LFW");
}

std::vector<std::tuple<std::filesystem::path, std::filesystem::path, int>>
AccuracyLFW::read_pairs() const {
    std::filesystem::path pairs_path = pairs_file_;

    if (pairs_path.empty() || !std::filesystem::exists(pairs_path)) {
        std::vector<std::filesystem::path> candidates;
        auto root = lfw_root_;
        candidates.push_back(root / "pairs.csv");
        candidates.push_back(root.parent_path() / "pairs.csv");
        candidates.push_back(root.parent_path().parent_path() / "pairs.csv");
        candidates.push_back(root / "pairs.txt");
        candidates.push_back(root.parent_path() / "pairs.txt");

        for (auto& c : candidates) {
            if (std::filesystem::exists(c)) { pairs_path = c; break; }
        }
    }

    if (!std::filesystem::exists(pairs_path)) {
        throw std::runtime_error("Pairs file not found. Tried: " + pairs_file_.string() +
                                 " and nearby locations.");
    }

    std::cout << "[LFW] Using pairs file: " << pairs_path << std::endl;

    auto img_root = resolve_lfw_root();
    std::cout << "[LFW] Using image root: " << img_root << std::endl;

    std::ifstream input(pairs_path);
    if (!input.is_open())
        throw std::runtime_error("Unable to open pairs file: " + pairs_path.string());

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path, int>> pairs;
    std::string line;
    int line_number = 1;

    bool is_csv = (pairs_path.extension() == ".csv");
    std::getline(input, line); 

    int skipped_lines   = 0;
    int missing_files   = 0;

    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) continue;

        std::vector<std::string> fields;
        if (is_csv) {
            fields = split_csv_line(line);
        } else {
            std::stringstream ss(line);
            std::string token;
            while (std::getline(ss, token, '\t')) fields.push_back(token);
        }

        for (auto& f : fields) f = trim(f);
        fields.erase(std::remove_if(fields.begin(), fields.end(),
            [](const std::string& v){ return v.empty(); }), fields.end());

        auto parse_int = [&](const std::string& value) {
            try {
                size_t pos = 0;
                int result = std::stoi(value, &pos);
                if (pos != value.size()) throw std::invalid_argument("trailing chars");
                return result;
            } catch (...) {
                throw std::runtime_error("Invalid integer in pairs file at line " +
                                         std::to_string(line_number) + ": '" + value + "'");
            }
        };

        std::filesystem::path p1, p2;
        int label = -1;

        if (fields.size() == 3) {
            const std::string& name = fields[0];
            int n1 = parse_int(fields[1]);
            int n2 = parse_int(fields[2]);
            p1 = img_root / name / (name + "_" + pad_number(n1) + ".jpg");
            p2 = img_root / name / (name + "_" + pad_number(n2) + ".jpg");
            label = 1;
        } else if (fields.size() == 4) {
            const std::string& name1 = fields[0];
            int n1 = parse_int(fields[1]);
            const std::string& name2 = fields[2];
            int n2 = parse_int(fields[3]);
            p1 = img_root / name1 / (name1 + "_" + pad_number(n1) + ".jpg");
            p2 = img_root / name2 / (name2 + "_" + pad_number(n2) + ".jpg");
            label = 0;
        } else {
            ++skipped_lines;
            std::cerr << "[LFW] Skipping malformed line " << line_number
                      << " (" << fields.size() << " fields): " << line << std::endl;
            continue;
        }

        bool p1_ok = std::filesystem::exists(p1);
        bool p2_ok = std::filesystem::exists(p2);
        if (!p1_ok || !p2_ok) {
            ++missing_files;
            if (missing_files <= 5) {  
                if (!p1_ok) std::cerr << "[LFW] Missing image: " << p1 << std::endl;
                if (!p2_ok) std::cerr << "[LFW] Missing image: " << p2 << std::endl;
            }
            continue;
        }

        pairs.emplace_back(p1, p2, label);
    }

    if (skipped_lines > 0)
        std::cerr << "[LFW] Total malformed lines skipped: " << skipped_lines << std::endl;
    if (missing_files > 0)
        std::cerr << "[LFW] Total pairs skipped due to missing files: " << missing_files << std::endl;

    std::cout << "[LFW] Loaded " << pairs.size() << " valid pairs." << std::endl;
    return pairs;
}

torch::Tensor AccuracyLFW::load_image_tensor(const std::filesystem::path& image_path) const {
    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (image.empty())
        throw std::runtime_error("Unable to read image: " + image_path.string());

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(112, 112));
    image.convertTo(image, CV_32F, 1.0f / 255.0f);

    auto tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kFloat32);
    tensor = tensor.permute({2, 0, 1}).clone();

    const std::vector<float> means = {0.485f, 0.456f, 0.406f};
    const std::vector<float> stds  = {0.229f, 0.224f, 0.225f};
    for (int i = 0; i < 3; ++i)
        tensor[i] = (tensor[i] - means[i]) / stds[i];

    return tensor;
}

std::vector<std::pair<std::filesystem::path, torch::Tensor>>
AccuracyLFW::build_embeddings(const std::vector<std::filesystem::path>& image_paths,
                               int64_t batch_size) const {
    std::vector<std::filesystem::path> unique_paths;
    {
        std::unordered_map<std::string, bool> seen;
        for (const auto& p : image_paths) {
            if (!seen.count(p.string())) {
                seen[p.string()] = true;
                unique_paths.push_back(p);
            }
        }
    }

    model_->eval();
    torch::NoGradGuard no_grad;

    std::vector<std::pair<std::filesystem::path, torch::Tensor>> embeddings;
    embeddings.reserve(unique_paths.size());

    for (size_t start = 0; start < unique_paths.size(); start += batch_size) {
        std::vector<torch::Tensor>       tensors;
        std::vector<std::filesystem::path> valid_paths;

        for (size_t i = start;
             i < std::min<size_t>(start + batch_size, unique_paths.size()); ++i) {
            if (std::filesystem::exists(unique_paths[i])) {
                try {
                    tensors.push_back(load_image_tensor(unique_paths[i]).unsqueeze(0).to(device_));
                    valid_paths.push_back(unique_paths[i]);
                } catch (const std::exception& e) {
                    std::cerr << "[LFW] Failed to load image: " << unique_paths[i]
                              << " - " << e.what() << std::endl;
                }
            }
        }

        if (tensors.empty()) continue;

        torch::Tensor batch  = torch::cat(tensors, 0);
        torch::Tensor output = model_->embed(batch);

        auto norms = output.norm(2, 1, true).clamp_min(1e-12);
        output = output / norms;

        auto cpu_out = output.cpu();
        for (size_t i = 0; i < valid_paths.size(); ++i)
            embeddings.emplace_back(valid_paths[i], cpu_out[i].clone());
    }

    model_->train();

    return embeddings;
}

std::pair<std::vector<double>, std::vector<int>>
AccuracyLFW::compute_similarities_and_labels() const {
    auto pairs = read_pairs();
    if (pairs.empty()) return {{}, {}};

    std::vector<std::filesystem::path> image_paths;
    image_paths.reserve(pairs.size() * 2);
    for (const auto& [p1, p2, lbl] : pairs) {
        image_paths.push_back(p1);
        image_paths.push_back(p2);
    }

    auto embeddings = build_embeddings(image_paths, 16);

    std::unordered_map<std::string, torch::Tensor> embedding_map;
    for (const auto& [path, tensor] : embeddings)
        embedding_map.emplace(path.string(), tensor);

    std::vector<double> similarities;
    std::vector<int>    labels;
    similarities.reserve(pairs.size());
    labels.reserve(pairs.size());

    for (const auto& [p1, p2, lbl] : pairs) {
        auto it1 = embedding_map.find(p1.string());
        auto it2 = embedding_map.find(p2.string());
        if (it1 == embedding_map.end() || it2 == embedding_map.end()) continue;

        double sim = static_cast<double>((it1->second * it2->second).sum().item<float>());
        similarities.push_back(sim);
        labels.push_back(lbl);
    }

    return {similarities, labels};
}

double AccuracyLFW::accuracy_for_threshold(const std::vector<double>& similarities,
                                            const std::vector<int>&    labels,
                                            double threshold) const {
    if (similarities.empty()) return 0.0;
    int correct = 0;
    for (size_t i = 0; i < similarities.size(); ++i) {
        bool pred  = similarities[i] >= threshold;
        bool truth = labels[i] == 1;
        if (pred == truth) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(similarities.size());
}

double AccuracyLFW::best_threshold_for(const std::vector<double>& similarities,
                                        const std::vector<int>&    labels,
                                        int num_steps) const {
    if (similarities.empty()) return 0.0;

    double min_sim = *std::min_element(similarities.begin(), similarities.end());
    double max_sim = *std::max_element(similarities.begin(), similarities.end());

    if (max_sim - min_sim < 1e-6) {
        std::cerr << "[LFW] WARNING: similarities are nearly constant ("
                  << min_sim << " ~ " << max_sim
                  << "). Model may be in embedding collapse." << std::endl;
        return 0.0;
    }

    double best_acc = -1.0;
    double best_thr = min_sim;

    for (int step = 0; step <= num_steps; ++step) {
        double thr = min_sim + (max_sim - min_sim) *
                     (static_cast<double>(step) / static_cast<double>(num_steps));
        double acc = accuracy_for_threshold(similarities, labels, thr);
        if (acc > best_acc) {
            best_acc = acc;
            best_thr = thr;
        }
    }

    std::cout << "[LFW] Best accuracy=" << (best_acc * 100.0)
              << "% at threshold=" << best_thr
              << " (sim range: [" << min_sim << ", " << max_sim << "])" << std::endl;

    return best_thr;
}

AccuracyLFW::EvaluationMetrics AccuracyLFW::Evaluate(double threshold,
                                                       int64_t batch_size,
                                                       int num_steps) const {
    auto pairs = read_pairs();

    std::vector<std::filesystem::path> image_paths;
    image_paths.reserve(pairs.size() * 2);
    for (const auto& [p1, p2, lbl] : pairs) {
        image_paths.push_back(p1);
        image_paths.push_back(p2);
    }

    auto embeddings = build_embeddings(image_paths, batch_size);

    std::unordered_map<std::string, torch::Tensor> embedding_map;
    for (const auto& [path, tensor] : embeddings)
        embedding_map.emplace(path.string(), tensor);

    std::vector<double> similarities;
    std::vector<int>    labels;
    similarities.reserve(pairs.size());
    labels.reserve(pairs.size());

    int64_t skipped = 0;
    for (const auto& [p1, p2, lbl] : pairs) {
        auto it1 = embedding_map.find(p1.string());
        auto it2 = embedding_map.find(p2.string());
        if (it1 == embedding_map.end() || it2 == embedding_map.end()) {
            ++skipped;
            continue;
        }
        double sim = static_cast<double>((it1->second * it2->second).sum().item<float>());
        similarities.push_back(sim);
        labels.push_back(lbl);
    }

    EvaluationMetrics metrics;
    metrics.total_pairs    = static_cast<int64_t>(labels.size());
    metrics.positive_pairs = static_cast<int64_t>(std::count(labels.begin(), labels.end(), 1));
    metrics.negative_pairs = static_cast<int64_t>(std::count(labels.begin(), labels.end(), 0));
    metrics.skipped_pairs  = skipped;
    metrics.best_threshold = best_threshold_for(similarities, labels, num_steps);

    double eval_thr = (threshold != 0.0) ? threshold : metrics.best_threshold;
    metrics.accuracy = accuracy_for_threshold(similarities, labels, eval_thr);

    return metrics;
}

double AccuracyLFW::ComputeAcc(double threshold) const {
    auto [sims, labels] = compute_similarities_and_labels();
    return accuracy_for_threshold(sims, labels, threshold);
}

double AccuracyLFW::BestThreshold(int num_steps) const {
    auto [sims, labels] = compute_similarities_and_labels();
    return best_threshold_for(sims, labels, num_steps);
}

double AccuracyLFW::ComputeDistence() const {
    auto [sims, labels] = compute_similarities_and_labels();
    if (sims.empty()) return 0.0;
    return std::accumulate(sims.begin(), sims.end(), 0.0) / static_cast<double>(sims.size());
}

} 