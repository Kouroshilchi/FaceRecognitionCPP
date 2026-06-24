#include "../include/model/HardMining.h"
#include <iostream>

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
    auto inf_mask = torch::full_like(mask_same_class, -1e9, torch::kFloat32);
    auto mask_with_inf = torch::where(mask_same_class, dist_mat, inf_mask);
    auto hardest_positive_dist = mask_with_inf.max(1).values;
    

    auto mask_diff_class = !mask_same_class;
    auto inf_mask_neg = torch::full_like(mask_diff_class, 1e9, torch::kFloat32);
    auto mask_with_inf_neg = torch::where(mask_diff_class, dist_mat, inf_mask_neg);
    auto hardest_negative_dist = mask_with_inf_neg.min(1).values;
    
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