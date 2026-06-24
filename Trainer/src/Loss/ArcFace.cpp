#include "../include/Loss/ArcFace.h"

namespace Loss {

ArcFaceImpl::ArcFaceImpl(int64_t in_features, int64_t out_features, float s_, float m_, bool easy_margin_)
: s(s_), m(m_), easy_margin(easy_margin_) {
    weight = register_parameter("weight", torch::empty({out_features, in_features}));
    torch::nn::init::xavier_uniform_(weight);
    cos_m = std::cos(m);
    sin_m = std::sin(m);
    thresh = std::cos(M_PI - m);
    mm = std::sin(M_PI - m) * m;
}

torch::Tensor ArcFaceImpl::forward(torch::Tensor features, torch::Tensor labels) {
    auto x = torch::nn::functional::normalize(features, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    auto w = torch::nn::functional::normalize(weight, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    auto cosine = torch::matmul(x, w.t());

    auto sine = torch::sqrt((1.0 - cosine.pow(2)).clamp(0, 1));
    auto phi = cosine * cos_m - sine * sin_m;

    if (easy_margin) {
        phi = torch::where(cosine > 0, phi, cosine);
    } else {
        phi = torch::where(cosine > thresh, phi, cosine - mm);
    }

    auto one_hot = torch::nn::functional::one_hot(labels, w.size(0)).to(cosine.dtype());
    auto output = (one_hot * phi) + ((1 - one_hot) * cosine);
    return output * s;
}

}