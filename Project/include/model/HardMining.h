#pragma once
#include <torch/torch.h>
#include <vector>
#include <tuple>

class HardMining {
public:
    static torch::Tensor batch_hard_triplet_loss(
        torch::Tensor embeddings,  
        torch::Tensor labels,     
        double margin = 0.2
    );
    
    static std::vector<std::tuple<int, int, int>> select_hard_triplets(
        torch::Tensor embeddings,
        torch::Tensor labels,
        int num_triplets = -1
    );
    
private:
    static torch::Tensor pairwise_distances(torch::Tensor embeddings);
};