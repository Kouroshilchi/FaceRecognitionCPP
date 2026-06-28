#pragma once

#include <torch/torch.h>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include "../Dataset/Dataset.h"

namespace dataset {

struct BalancedBatchSampler {

    BalancedBatchSampler(const dataset::FaceDataset& dataset,
                         int64_t P,   // تعداد کلاس در هر batch
                         int64_t K)   // تعداد تصویر از هر کلاس
        : P_(P), K_(K), rng_(std::random_device{}()) {

        // ساختن نقشه از label به index های تصاویر
        const auto& samples = dataset.samples();
        for (size_t i = 0; i < samples.size(); ++i) {
            int64_t label = samples[i].second;
            class_to_indices_[label].push_back(i);
        }

        // فیلتر کردن کلاس‌هایی که کمتر از K تصویر دارن
        for (auto it = class_to_indices_.begin(); it != class_to_indices_.end(); ) {
            if ((int64_t)it->second.size() < K_) {
                it = class_to_indices_.erase(it);
            } else {
                valid_classes_.push_back(it->first);
                ++it;
            }
        }

        if ((int64_t)valid_classes_.size() < P_) {
            throw std::runtime_error(
                "Not enough classes with >= " + std::to_string(K_) +
                " samples. Have " + std::to_string(valid_classes_.size()) +
                " but need " + std::to_string(P_));
        }

        // *** اصلاح اصلی: num_batches بر اساس کل تصاویر valid ***
        // قبلاً: num_batches_ = valid_classes_.size() / P_
        // این فقط 330 batch میداد (10558/32) و اکثر دیتاست skip میشد
        //
        // الان: کل تصاویر valid رو حساب میکنیم تقسیم بر batch_size
        // این ~1916 batch میده که معادل DataLoader عادیه
        int64_t total_valid_images = 0;
        for (const auto& cls : valid_classes_) {
            total_valid_images += (int64_t)class_to_indices_.at(cls).size();
        }
        num_batches_ = total_valid_images / (P_ * K_);

        // shuffle اولیه
        shuffled_classes_ = valid_classes_;
        std::shuffle(shuffled_classes_.begin(), shuffled_classes_.end(), rng_);
    }

    std::vector<size_t> next_batch() {
        // اگه کلاس‌ها تموم شد، دوباره shuffle کن و از اول شروع کن
        if (class_cursor_ + P_ > (int64_t)shuffled_classes_.size()) {
            shuffled_classes_ = valid_classes_;
            std::shuffle(shuffled_classes_.begin(), shuffled_classes_.end(), rng_);
            class_cursor_ = 0;
        }

        std::vector<size_t> batch_indices;
        batch_indices.reserve(P_ * K_);

        for (int64_t i = 0; i < P_; ++i) {
            int64_t cls = shuffled_classes_[class_cursor_ + i];
            auto& indices = class_to_indices_[cls];

            // shuffle تصاویر این کلاس و انتخاب K تا
            std::shuffle(indices.begin(), indices.end(), rng_);
            for (int64_t k = 0; k < K_; ++k) {
                batch_indices.push_back(indices[k % indices.size()]);
            }
        }

        class_cursor_ += P_;
        return batch_indices;
    }

    int64_t batch_size()   const { return P_ * K_; }
    int64_t num_batches()  const { return num_batches_; }
    int64_t num_classes()  const { return (int64_t)valid_classes_.size(); }
    int64_t P()            const { return P_; }
    int64_t K()            const { return K_; }

private:
    int64_t P_;
    int64_t K_;
    int64_t num_batches_   = 0;
    int64_t class_cursor_  = 0;

    std::map<int64_t, std::vector<size_t>> class_to_indices_;
    std::vector<int64_t> valid_classes_;
    std::vector<int64_t> shuffled_classes_;

    std::mt19937 rng_;
};

} // namespace dataset