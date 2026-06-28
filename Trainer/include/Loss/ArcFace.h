#pragma once

#include <torch/torch.h>
#include "LossMetrics.h"

namespace Loss {

class ArcFaceImpl : public torch::nn::Module {
public:
    ArcFaceImpl(int64_t num_classes, 
                int64_t embedding_dim,
                double scale = 30.0, 
                double margin = 0.5);

    LossMetrics forward(const torch::Tensor& embeddings, 
                         const torch::Tensor& labels);

    
    void set_scale(double s) { s_ = s; }
    void set_margin(double m) { m_ = m; }
    double get_scale() const { return s_; }
    double get_margin() const { return m_; }

private:
    torch::Tensor weight_;  

    double s_;  
    double m_;  
};

TORCH_MODULE(ArcFace);

} 