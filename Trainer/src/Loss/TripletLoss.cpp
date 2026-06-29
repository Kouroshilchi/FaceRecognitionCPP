#include "../include/Loss/TripletLoss.h"
#include <iostream>
#include <cmath>

namespace Loss
{

    static torch::Tensor pairwise_distance(const torch::Tensor& emb) {
        auto sq  = emb.pow(2).sum(1);                     
        auto dot = emb.matmul(emb.t());                  
        auto d2  = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * dot;
        d2 = torch::clamp_min(d2, 1e-12);
        return torch::sqrt(d2);           
    }

    static std::pair<torch::Tensor, torch::Tensor>
    build_masks(const torch::Tensor& labels, torch::Device device) {
        auto N      = labels.size(0);
        auto lab    = labels.view({-1, 1});
        auto pos    = lab.eq(lab.t());                       
        auto neg    = lab.ne(lab.t());                        
        auto eye    = torch::eye(N,
            torch::TensorOptions().device(device).dtype(torch::kBool));
        pos = pos.logical_and(eye.logical_not());          
        return {pos, neg};
    }

    LossMetrics TripletLossImpl::forward_semi_hard(
        const torch::Tensor& embeddings,
        const torch::Tensor& labels)
    {
        auto N = embeddings.size(0);
        if (N <= 1)
            return {torch::tensor(0.0, embeddings.options()), 0.0, 0.0, 0, 0};

        auto device = embeddings.device();
        auto dist   = pairwise_distance(embeddings);
        auto [pos_mask, neg_mask] = build_masks(labels, device);

        const double NEG_INF = -1e9;
        auto dist_pos    = dist.masked_fill(pos_mask.logical_not(), NEG_INF);
        auto hardest_pos = std::get<0>(dist_pos.max(1)); 


        auto hp_col    = hardest_pos.unsqueeze(1);
        auto is_semi   = neg_mask
                         .logical_and(dist.gt(hp_col))
                         .logical_and(dist.lt(hp_col + margin_));

        const double POS_INF = 1e9;
        auto dist_semi       = dist.masked_fill(is_semi.logical_not(), POS_INF);
        auto semi_hard_neg   = std::get<0>(dist_semi.min(1));
        auto has_semi        = is_semi.any(1);            

        auto dist_neg_all    = dist.masked_fill(neg_mask.logical_not(), POS_INF);
        auto hard_neg     = std::get<0>(dist_neg_all.min(1));

        auto chosen_neg = torch::where(has_semi, semi_hard_neg, hard_neg);

        auto losses = torch::relu(hardest_pos - chosen_neg + margin_);
        // auto losses = torch::softplus(hardest_pos - chosen_neg);

        auto has_pos  = pos_mask.any(1);
        auto has_neg  = neg_mask.any(1);
        auto valid    = has_pos.logical_and(has_neg);
        int64_t valid_count = valid.sum().item<int64_t>();

        if (valid_count == 0) {
            std::cerr << "[TripletLoss] Warning: No valid anchor found!\n";
            return {torch::tensor(0.0, embeddings.options()), 0.0, 0.0, 0, 0};
        }

        auto losses_valid  = losses.masked_select(valid);
        auto hp_valid      = hardest_pos.masked_select(valid);
        auto hn_valid      = chosen_neg.masked_select(valid);

        double avg_pos_dist = hp_valid.mean().item<double>();
        double avg_neg_dist = hn_valid.mean().item<double>();

        auto loss            = losses_valid.mean();
        int64_t zero_count   = losses_valid.eq(0).sum().item<int64_t>();
        int64_t semi_used    = has_semi.logical_and(valid).sum().item<int64_t>();

        if (std::isnan(loss.item<double>())) {
            std::cerr << "[TripletLoss] Warning: NaN loss!\n";
            loss = torch::tensor(0.0, embeddings.options());
        }

        if (zero_count > valid_count * 0.5) {
            std::cout << "[TripletLoss Semi-Hard] "
                      << (100.0 * zero_count / valid_count)
                      << "% zero-triplets | semi-hard used: "
                      << semi_used << "/" << valid_count << "\n";
        }

        return {loss, avg_pos_dist, avg_neg_dist,
                (int64_t)losses_valid.numel(), zero_count};
    }

    LossMetrics TripletLossImpl::forward_online_hard(
        const torch::Tensor& embeddings,
        const torch::Tensor& labels)
    {
        auto N = embeddings.size(0);
        if (N <= 1)
            return {torch::tensor(0.0, embeddings.options()), 0.0, 0.0, 0, 0};

        auto device = embeddings.device();
        auto dist   = pairwise_distance(embeddings);
        auto [pos_mask, neg_mask] = build_masks(labels, device);

        const double NEG_INF = -1e9;
        auto dist_pos    = dist.masked_fill(pos_mask.logical_not(), NEG_INF);
        auto hardest_pos = std::get<0>(dist_pos.max(1));      // [N]

        const double POS_INF = 1e9;
        auto dist_neg    = dist.masked_fill(neg_mask.logical_not(), POS_INF);
        auto hardest_neg = std::get<0>(dist_neg.min(1));      // [N]

        auto losses = torch::relu(hardest_pos - hardest_neg + margin_);
        // auto losses = torch::softplus(hardest_pos - hardest_neg);

        auto has_pos = pos_mask.any(1);
        auto has_neg = neg_mask.any(1);
        auto valid   = has_pos.logical_and(has_neg);
        int64_t valid_count = valid.sum().item<int64_t>();

        if (valid_count == 0) {
            return {torch::tensor(0.0, embeddings.options()), 0.0, 0.0, 0, 0};
        }

        auto losses_valid = losses.masked_select(valid);
        auto hp_valid     = hardest_pos.masked_select(valid);
        auto hn_valid     = hardest_neg.masked_select(valid);

        double avg_pos_dist = hp_valid.mean().item<double>();
        double avg_neg_dist = hn_valid.mean().item<double>();

        auto loss           = losses_valid.mean();
        int64_t zero_count  = losses_valid.eq(0).sum().item<int64_t>();

        if (std::isnan(loss.item<double>())) {
            loss = torch::tensor(0.0, embeddings.options());
        }

        return {loss, avg_pos_dist, avg_neg_dist,
                (int64_t)losses_valid.numel(), zero_count};
    }

} // namespace Loss