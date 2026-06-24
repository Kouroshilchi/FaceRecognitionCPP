#include "../include/model/HardMining.h"
#include <algorithm>

torch::Tensor HardMining::pairwise_distances(torch::Tensor embeddings) {
    auto dot_product = torch::mm(embeddings, embeddings.transpose(0, 1));
    auto square_norm = torch::diag(dot_product);
    auto distances = square_norm.unsqueeze(1) + square_norm.unsqueeze(0) - 2 * dot_product;
    distances = torch::clamp(distances, 1e-12, 1e12);
    return distances;
}

torch::Tensor HardMining::batch_hard_triplet_loss(
    torch::Tensor embeddings,
    torch::Tensor labels,
    double margin) {
    
    int64_t batch_size = embeddings.size(0);
    auto device = embeddings.device();
    auto dist_mat = pairwise_distances(embeddings);
    
    auto mask_same_class = labels.unsqueeze(1) == labels.unsqueeze(0);
    
    auto same_class_dists = dist_mat.clone();
    same_class_dists.masked_fill_(~mask_same_class, -1e9);
    auto hardest_positive_dist = std::get<0>(same_class_dists.max(1));
    
    auto diff_class_dists = dist_mat.clone();
    diff_class_dists.masked_fill_(mask_same_class, 1e9);
    auto hardest_negative_dist = std::get<0>(diff_class_dists.min(1));
    
    auto loss = torch::relu(hardest_positive_dist - hardest_negative_dist + margin);
    return loss.mean();
}

std::vector<std::tuple<int, int, int>> HardMining::select_hard_triplets(
    torch::Tensor embeddings,
    torch::Tensor labels,
    int num_triplets) {
    
    auto dist_mat = pairwise_distances(embeddings);
    int64_t batch_size = embeddings.size(0);
    std::vector<std::tuple<int, int, int>> triplets;
    
    for (int64_t i = 0; i < batch_size; ++i) {
        std::vector<std::pair<float, int64_t>> positive_dists;
        std::vector<std::pair<float, int64_t>> negative_dists;
        
        for (int64_t j = 0; j < batch_size; ++j) {
            if (i == j) continue;
            float dist = dist_mat[i][j].item<float>();
            
            if (labels[i].item<int64_t>() == labels[j].item<int64_t>()) {
                positive_dists.push_back({dist, j});
            } else {
                negative_dists.push_back({dist, j});
            }
        }
        
        std::sort(positive_dists.begin(), positive_dists.end(), 
            [](auto& a, auto& b) { return a.first > b.first; });
        
        std::sort(negative_dists.begin(), negative_dists.end(),
            [](auto& a, auto& b) { return a.first < b.first; });
        
        if (!positive_dists.empty() && !negative_dists.empty()) {
            int hard_positive = positive_dists[0].second;
            int hard_negative = negative_dists[0].second;
            triplets.push_back({i, hard_positive, hard_negative});
            
            if (num_triplets > 0 && triplets.size() >= (size_t)num_triplets) {
                break;
            }
        }
    }
    
    return triplets;
}