#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include "include/model/Model.h"

torch::Tensor load_and_preprocess_image(const std::string& image_path, const cv::Size& image_size) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        throw std::runtime_error("Cannot load image: " + image_path);
    }

    cv::Mat img_resized;
    cv::resize(img, img_resized, image_size);

    cv::Mat img_rgb;
    cv::cvtColor(img_resized, img_rgb, cv::COLOR_BGR2RGB);

    img_rgb.convertTo(img_rgb, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    cv::split(img_rgb, channels);

    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);

    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }

    cv::Mat normalized;
    cv::merge(channels, normalized);

    torch::Tensor tensor = torch::from_blob(
        normalized.data,
        {image_size.height, image_size.width, 3},
        torch::kFloat32
    ).clone().permute({2, 0, 1}).unsqueeze(0); 

    return tensor;
}

double compute_cosine_similarity(const torch::Tensor& emb1, const torch::Tensor& emb2) {
    torch::Tensor e1 = emb1.view(-1);
    torch::Tensor e2 = emb2.view(-1);

    double dot_product = torch::dot(e1, e2).item<double>();
    double norm1 = torch::norm(e1).item<double>();
    double norm2 = torch::norm(e2).item<double>();

    if (norm1 < 1e-8 || norm2 < 1e-8) {
        return 0.0;
    }

    double similarity = dot_product / (norm1 * norm2);
    similarity = std::max(-1.0, std::min(1.0, similarity));
    return similarity;
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " <image1_path> <image2_path>" << std::endl;
            return 1;
        }

        std::string image1_path = argv[1];
        std::string image2_path = argv[2];

        std::cout << "========== Face Image Comparison ==========" << std::endl;
        std::cout << "Image 1: " << image1_path << std::endl;
        std::cout << "Image 2: " << image2_path << std::endl;

        // Configuration
        const int64_t embedding_dim = 128;
        const double dropout = 0.1;
        const cv::Size image_size{224, 224};

        // Device selection
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << device << std::endl;

        // Load model
        std::cout << "\nLoading model from 'models/model.pt'..." << std::endl;
        auto model = model::FaceRecognitionModel(3, embedding_dim, dropout);
        torch::load(model, "models/model.pt");
        model->to(device);
        model->eval();
        std::cout << "Model loaded successfully!" << std::endl;

        std::cout << "\nLoading and preprocessing images..." << std::endl;
        torch::Tensor img1_tensor = load_and_preprocess_image(image1_path, image_size);
        torch::Tensor img2_tensor = load_and_preprocess_image(image2_path, image_size);

        img1_tensor = img1_tensor.to(device);
        img2_tensor = img2_tensor.to(device);

        std::cout << "Image 1 tensor shape: " << img1_tensor.sizes() << std::endl;
        std::cout << "Image 2 tensor shape: " << img2_tensor.sizes() << std::endl;

        std::cout << "\nGenerating embeddings..." << std::endl;
        torch::NoGradGuard no_grad;
        torch::Tensor embedding1 = model->forward(img1_tensor);
        torch::Tensor embedding2 = model->forward(img2_tensor);

        std::cout << "Embedding 1 shape: " << embedding1.sizes() << std::endl;
        std::cout << "Embedding 2 shape: " << embedding2.sizes() << std::endl;

        std::cout << "\nComputing similarity..." << std::endl;
        double similarity = compute_cosine_similarity(embedding1, embedding2);

        double similarity_percentage = (similarity + 1.0) / 2.0 * 100.0;

        std::cout << "\n========== Results ==========" << std::endl;
        std::cout << "Cosine Similarity: " << similarity << std::endl;
        std::cout << "Similarity Percentage: " << similarity_percentage << "%" << std::endl;
        std::cout << "===========================" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
