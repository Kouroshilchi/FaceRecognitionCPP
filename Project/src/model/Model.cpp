#include "../include/model/Model.h"

namespace model {
    FaceRecognitionModelImpl::FaceRecognitionModelImpl(
        int num_channel,
        int out_dim,
        double dropout
    )
    {
        backbone = register_module("backbone", FaceRecognitionBackBone(num_channel, 128 , dropout));
        projector = register_module("projector", FaceRecognitionProjector(128, out_dim, dropout));
    }

    void FaceRecognitionModelImpl::load_pretrained_weights(const std::string& weights_path) {
        projector->load_pretrained_weights(weights_path);
        
        std::cout << "All pretrained weights loaded successfully!" << std::endl;
    }

    torch::Tensor FaceRecognitionModelImpl::forward(torch::Tensor x) {
        x = backbone->forward(x);
        x = projector->forward(x);
        return x;
    }
}