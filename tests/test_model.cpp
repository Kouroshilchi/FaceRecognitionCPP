#pragma once

#include <torch/torch.h>
#include "../Project/include/model/projector.h"
#include "../Project/include/model/Model.h"
#include <iostream>

bool testFaceRecognitionProjector(int in_channel, int out_dim, double dropout) {
    try {
        int batch_size = 4;
        int img_height = 224;
        int img_width = 224;
        
        auto model = std::make_shared<model::FaceRecognitionProjector>(in_channel, out_dim, dropout);
        model->eval();
        
        std::cout << "✓ Model created successfully!" << std::endl;
        
        auto input = torch::randn({batch_size, in_channel, img_height, img_width});
        std::cout << "✓ Input shape: " << input.sizes() << std::endl;
        
        torch::Tensor output = model->forward(input);
        std::cout << "✓ Output shape: " << output.sizes() << std::endl;
        
        if (output.size(0) != batch_size || output.size(1) != out_dim) {
            std::cerr << "✗ Output dimensions are incorrect!" << std::endl;
            return false;
        }
        
        if (torch::isnan(output).any().item<bool>() || torch::isinf(output).any().item<bool>()) {
            std::cerr << "✗ Output contains NaN or Inf values!" << std::endl;
            return false;
        }
        
        std::cout << "✓ Projector test passed!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Error: " << e.what() << std::endl;
        return false;
    }
}

bool testFaceRecognitionModel(int num_channel, int out_dim, double dropout) {
    try {
        int batch_size = 4;
        int img_height = 224;
        int img_width = 224;
        
        auto model = std::make_shared<model::FaceRecognitionModel>(num_channel, out_dim, dropout);
        model->eval();
        
        std::cout << "\n=== Testing Full FaceRecognitionModel ===" << std::endl;
        std::cout << "✓ Model created successfully!" << std::endl;
        
        auto input = torch::randn({batch_size, num_channel, img_height, img_width});
        std::cout << "✓ Input shape: " << input.sizes() << std::endl;
        
        torch::Tensor output = model->forward(input);
        std::cout << "✓ Output shape: " << output.sizes() << std::endl;
        
        if (output.size(0) != batch_size || output.size(1) != out_dim) {
            std::cerr << "✗ Output dimensions are incorrect!" << std::endl;
            return false;
        }
        
        if (torch::isnan(output).any().item<bool>() || torch::isinf(output).any().item<bool>()) {
            std::cerr << "✗ Output contains NaN or Inf values!" << std::endl;
            return false;
        }
        
        std::cout << "✓ Full model test passed!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Error: " << e.what() << std::endl;
        return false;
    }
}
