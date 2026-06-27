#include "../include/Loss/TripletLoss.h"
#include <iostream>
#include <cmath>

namespace Loss 
{
    LossMetrics TripletLossImpl::forward(const torch::Tensor& embeddings, const torch::Tensor& labels) {
        auto N = embeddings.size(0);
        if (N <= 1) {
            return {torch::tensor(0.0, embeddings.options()), 0.0, 0.0, 0, 0};
        }

        auto device = embeddings.device();
        
        auto sq = embeddings.pow(2).sum(1);
        auto dist = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * embeddings.matmul(embeddings.t());
        dist = torch::clamp_min(dist, 1e-12);
        dist = torch::sqrt(dist);

        auto labels1 = labels.view({-1, 1});
        auto eq = labels1.eq(labels1.t());
        auto neq = labels1.ne(labels1.t());

        auto eye = torch::eye(N, torch::TensorOptions().device(device).dtype(torch::kBool));
        eq = eq.logical_and(eye.logical_not());

        const double NEG_INF = -1e12;
        auto dist_pos = dist.masked_fill(eq.logical_not(), NEG_INF);
        auto hardest_pos = std::get<0>(dist_pos.max(1));

        auto lower_bound = hardest_pos.unsqueeze(1);  
        auto upper_bound = lower_bound + margin_;     

        auto is_semi_hard = neq.logical_and(
            dist.ge(lower_bound).logical_and(dist.le(upper_bound))
        );

        const double POS_INF = 1e12;
        auto dist_semi_hard = dist.masked_fill(is_semi_hard.logical_not(), POS_INF);
        auto hardest_semi_hard = std::get<0>(dist_semi_hard.min(1));

        auto has_semi_hard = is_semi_hard.any(1);

        auto dist_neg = dist.masked_fill(neq.logical_not(), POS_INF);
        auto hardest_neg_fallback = std::get<0>(dist_neg.min(1));

        auto hardest_neg = torch::where(
            has_semi_hard,
            hardest_semi_hard,
            hardest_neg_fallback
        );

        auto losses = torch::relu(hardest_pos - hardest_neg + margin_);

        auto has_pos = eq.any(1);
        auto has_neg = neq.any(1);
        auto valid = has_pos.logical_and(has_neg);
        
        int64_t valid_count = valid.sum().item<int64_t>();

        if (valid_count == 0) {
            std::cerr << "Warning: No valid triplet samples!" << std::endl;
            auto loss = torch::tensor(0.0, embeddings.options());
            return {loss, 0.0, 0.0, 0, 0};
        }

        double avg_pos_metric = 0.0;
        double avg_neg_metric = 0.0;
        
        auto hardest_pos_valid = hardest_pos.masked_select(valid);
        auto hardest_neg_valid = hardest_neg.masked_select(valid);
        
        if (hardest_pos_valid.numel() > 0) {
            avg_pos_metric = hardest_pos_valid.mean().item<double>();
        }
        if (hardest_neg_valid.numel() > 0) {
            avg_neg_metric = hardest_neg_valid.mean().item<double>();
        }

        auto losses_valid = losses.masked_select(valid);
        auto loss = losses_valid.mean();
        int64_t zero_loss_count = losses_valid.eq(0).sum().item<int64_t>();
        int64_t semi_hard_used = has_semi_hard.sum().item<int64_t>();
        
        valid_count = losses_valid.numel();
        
        if (std::isnan(loss.item<double>())) {
            std::cerr << "Warning: NaN loss detected!" << std::endl;
            loss = torch::tensor(0.0, embeddings.options());
        }

        if (zero_loss_count > valid_count * 0.5) {
            std::cout << "[WARNING] " << (100.0 * zero_loss_count / valid_count) 
                      << "% zero-triplets | " << semi_hard_used 
                      << " semi-hard used" << std::endl;
        }

        return {loss, avg_pos_metric, avg_neg_metric, valid_count, zero_loss_count};
    }

    LossMetrics TripletLossImpl::forward_online_hard(
        const torch::Tensor& embeddings, 
        const torch::Tensor& labels) {
        
        auto N = embeddings.size(0);
        if (N <= 1) {
            return {torch::tensor(0.0, embeddings.options()), 0.0, 0.0, 0, 0};
        }

        auto device = embeddings.device();
        
        auto sq = embeddings.pow(2).sum(1);
        auto dist = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * embeddings.matmul(embeddings.t());
        dist = torch::clamp_min(dist, 1e-12);
        dist = torch::sqrt(dist);

        auto labels1 = labels.view({-1, 1});
        auto eq = labels1.eq(labels1.t());
        auto neq = labels1.ne(labels1.t());
        auto eye = torch::eye(N, torch::TensorOptions().device(device).dtype(torch::kBool));
        eq = eq.logical_and(eye.logical_not());

        const double NEG_INF = -1e12;
        auto dist_pos = dist.masked_fill(eq.logical_not(), NEG_INF);
        auto hardest_pos = std::get<0>(dist_pos.max(1));

        const double POS_INF = 1e12;
        auto dist_neg = dist.masked_fill(neq.logical_not(), POS_INF);
        auto hardest_neg = std::get<0>(dist_neg.min(1));

        auto losses = torch::relu(hardest_pos - hardest_neg + margin_);

        auto has_pos = eq.any(1);
        auto has_neg = neq.any(1);
        auto valid = has_pos.logical_and(has_neg);
        
        int64_t valid_count = valid.sum().item<int64_t>();

        if (valid_count == 0) {
            auto loss = torch::tensor(0.0, embeddings.options());
            return {loss, 0.0, 0.0, 0, 0};
        }

        double avg_pos_metric = 0.0;
        double avg_neg_metric = 0.0;
        
        auto hardest_pos_valid = hardest_pos.masked_select(valid);
        auto hardest_neg_valid = hardest_neg.masked_select(valid);
        
        if (hardest_pos_valid.numel() > 0) {
            avg_pos_metric = hardest_pos_valid.mean().item<double>();
        }
        if (hardest_neg_valid.numel() > 0) {
            avg_neg_metric = hardest_neg_valid.mean().item<double>();
        }

        auto losses_valid = losses.masked_select(valid);
        auto loss = losses_valid.mean();
        int64_t zero_loss_count = losses_valid.eq(0).sum().item<int64_t>();
        
        valid_count = losses_valid.numel();
        
        if (std::isnan(loss.item<double>())) {
            loss = torch::tensor(0.0, embeddings.options());
        }

        return {loss, avg_pos_metric, avg_neg_metric, valid_count, zero_loss_count};
    }

    LossMetrics TripletLossImpl::forward_batch_hard(
        const torch::Tensor& embeddings, 
        const torch::Tensor& labels) {

        auto N = embeddings.size(0);
        if (N <= 1) {
            return {torch::tensor(0.0, embeddings.options()), 0.0, 0.0, 0, 0};
        } 

        auto device = embeddings.device();
        
        auto sq = embeddings.pow(2).sum(1);
        auto dist = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * embeddings.matmul(embeddings.t());
        dist = torch::clamp_min(dist, 1e-12);
        dist = torch::sqrt(dist);

        auto labels1 = labels.view({-1, 1});
        auto eq = labels1.eq(labels1.t());
        auto neq = labels1.ne(labels1.t());
        auto eye = torch::eye(N, torch::TensorOptions().device(device).dtype(torch::kBool));
        eq = eq.logical_and(eye.logical_not());

        const double NEG_INF = -1e12;
        auto dist_pos = dist.masked_fill(eq.logical_not(), NEG_INF);
        auto hardest_pos = std::get<0>(dist_pos.max(1));

        const double POS_INF = 1e12;
        auto dist_neg = dist.masked_fill(neq.logical_not(), POS_INF);
        auto hardest_neg = std::get<0>(dist_neg.min(1));

        auto losses = torch::relu(hardest_pos - hardest_neg + margin_);

        auto valid = eq.any(1).logical_and(neq.any(1));
        
        int64_t valid_count = valid.sum().item<int64_t>();

        if (valid_count == 0) {
            auto loss = torch::tensor(0.0, embeddings.options());
            return {loss, 0.0, 0.0, 0, 0};
        }

        auto losses_valid = losses.masked_select(valid);
        auto loss = losses_valid.mean();
        int64_t zero_loss_count = losses_valid.eq(0).sum().item<int64_t>();

        double avg_pos_metric = hardest_pos.masked_select(valid).mean().item<double>();
        double avg_neg_metric = hardest_neg.masked_select(valid).mean().item<double>();
        
        valid_count = losses_valid.numel();
        
        if (std::isnan(loss.item<double>())) {
            loss = torch::tensor(0.0, embeddings.options());
        }

        return {loss, avg_pos_metric, avg_neg_metric, valid_count, zero_loss_count};
    }
}