#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <torch/optim/schedulers/step_lr.h>
#include "include/Model/Model.h"
#include "include/Model/FaceNet.h"
#include "include/Dataset/Dataset.h"
#include "include/Dataset/BalancedSampler.h"
#include "include/Loss/TripletLoss.h"
#include "include/Loss/ArcFace.h"
#include "include/Utils/AccuracyLFW.h"
#include <random>
#include <map>
#include <cstdlib>
#include <filesystem>

static std::filesystem::path g_repo_root;

std::string getOsName()
{
    #ifdef _WIN32
    return "Windows 32-bit";
    #elif _WIN64
    return "Windows 64-bit";
    #elif __APPLE__ || __MACH__
    return "Mac OSX";
    #elif __linux__
    return "Linux";
    #elif __FreeBSD__
    return "FreeBSD";
    #elif __unix || __unix__
    return "Unix";
    #else
    return "Other";
    #endif
}                      

void init_repo_root(const char* argv0) {
    const char* env = std::getenv("REPO_ROOT");
    if (env && std::filesystem::exists(env)) {
        g_repo_root = std::filesystem::canonical(env);
        return;
    }
    auto bin = std::filesystem::canonical(argv0);
    if (getOsName() == "Windows 32-bit" || getOsName() == "Windows 64-bit") {
        g_repo_root = bin.parent_path().parent_path().parent_path();
    }
    else if (getOsName() == "Linux" || getOsName() == "Mac OSX") {
        g_repo_root = bin.parent_path().parent_path();
    }
    else {
        throw std::runtime_error("Unsupported OS: " + getOsName());
    }
}

std::filesystem::path get_repo_root()  
{ 
    return g_repo_root; 
}
std::string get_model_save_path()      
{ 
    return (g_repo_root / "models" / "model.pt").string(); 
}
std::string get_weights_path()         
{ 
    return (g_repo_root / "models" / "resnet50_weights.pt").string(); 
}

void evaluate_lfw(model::FaceNet& facenet, torch::Device device) {
    std::cout << "=================================================" << std::endl;
    try {
        auto lfw_root   = get_repo_root() / "data" / "data_LFW";
        auto pairs_file = lfw_root / "pairs.csv";
        if (!std::filesystem::exists(lfw_root)) {
            std::cerr << "[LFW] WARNING: LFW root not found: " << lfw_root << std::endl;
            return;
        }
        if (!std::filesystem::exists(pairs_file)) {
            std::cerr << "[LFW] WARNING: pairs.csv not found: " << pairs_file << std::endl;
            return;
        }

        Utils::AccuracyLFW lfw_eval(lfw_root, pairs_file, facenet, device);
        auto lfw_metrics = lfw_eval.Evaluate(0.0, 16, 500);

        std::cout << "[LFW] Total pairs    : " << lfw_metrics.total_pairs    << std::endl;
        std::cout << "[LFW] Accuracy (best thr=" << lfw_metrics.best_threshold << "): "
                  << (lfw_metrics.accuracy * 100.0) << "%" << std::endl;

        if (lfw_metrics.total_pairs == 0) {
            std::cerr << "[LFW] ERROR: 0 pairs evaluated!" << std::endl;
            return;
        }
    } catch (const std::exception& e) {
        std::cerr << "[LFW] EXCEPTION: " << e.what() << std::endl;
    }
    std::cout << "=================================================" << std::endl;

}

int main(int argc, char* argv[]) {
    try {
        init_repo_root(argv[0]);
        std::cout << "Repo root: " << get_repo_root() << std::endl;

        const std::string dataset_root = (get_repo_root() / "data" / "data_vgg2_casia").string();

        
        const int64_t P                 = 32;   
        const int64_t K                 =  4;    
        const cv::Size image_size         {112, 112};
        const int64_t embedding_dim     = 128;
        const int64_t epochs            = 100;
        int64_t log_step                = 100;
        bool resume                     = false;
        double model_lr                 = 1e-4;
        double arcface_lr               = 1e-4;
        double model_lastlayer_lr       = 1e-4;
        model::LossType loss_type       = model::LossType::ArcFace;
        std::string mining_mode         = "ArcFace";
        bool pretrained_resnet          = true;
        int64_t nan_loss_counter        = 0;
        int64_t zero_triplet_counter    = 0;
        std::vector<torch::Tensor> backbone_params, arcface_params, head_params;
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << device << std::endl;

        std::cout << "Loading dataset from: " << dataset_root << std::endl;
        auto raw_dataset = dataset::FaceDataset(dataset_root, image_size, true);
        const int64_t num_classes  = raw_dataset.num_classes();
        const size_t  dataset_size = raw_dataset.size().value();

        
        dataset::BalancedBatchSampler sampler(raw_dataset, P, K);

        std::cout << "Dataset size     : " << dataset_size   << std::endl;
        std::cout << "Total classes    : " << num_classes     << std::endl;
        std::cout << "Valid classes    : " << sampler.num_classes() << " (>= " << K << " images)" << std::endl;
        std::cout << "Batch config     : P=" << P << " classes x K=" << K << " images = " << sampler.batch_size() << " per batch" << std::endl;
        std::cout << "Batches per epoch: " << sampler.num_batches() << std::endl;



        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--resume") {
                resume = true;
            }
            else if (std::string(argv[i]) == "--scratch") {
                pretrained_resnet = false;
            }
            else if (std::string(argv[i]) == "--model_lr") {
                if (i + 1 < argc) {
                    model_lr = std::stod(argv[i + 1]);
                    ++i;
                } else {
                    std::cerr << "Error: --model_lr requires a value" << std::endl;
                    return 1;
                }
            }
            else if (std::string(argv[i]) == "--log_step") {
                if (i + 1 < argc) {
                    log_step = std::stod(argv[i + 1]);
                    ++i;
                } else {
                    std::cerr << "Error: --log_step requires a value" << std::endl;
                    return 1;
                }
            }
            else if (std::string(argv[i]) == "--lastlayer_lr") {
                if (i + 1 < argc) {
                    model_lastlayer_lr = std::stod(argv[i + 1]);
                    ++i;
                } else {
                    std::cerr << "Error: --lastlayer_lr requires a value" << std::endl;
                    return 1;
                }
            }
            else if (std::string(argv[i]) == "--arcface_lr") {
                if (i + 1 < argc) {
                    arcface_lr = std::stod(argv[i + 1]);
                    ++i;
                } else {
                    std::cerr << "Error: --arcface_lr requires a value" << std::endl;
                    return 1;
                }
            }
            else if (std::string(argv[i]) == "--loss") {
                if (i + 1 < argc) {
                    std::string value = argv[i + 1];
                    ++i;
                    if (value == "ArcFace") {
                        loss_type = model::LossType::ArcFace;
                        mining_mode = "arcface";
                    } else if (value == "triplet_semi") {
                        loss_type = model::LossType::TripletSemiHard;
                        mining_mode = "Triplet Semi";
                    } else if (value == "triplet_hard") {
                        loss_type = model::LossType::TripletOnlineHard;
                        mining_mode = "Triplet Hard";
                    } else {
                        std::cerr << "Error: unknown loss type '" << value << "'. "
                                  << "Supported values: arcface, triplet_semi, triplet_hard" << std::endl;
                        return 1;
                    }
                } else {
                    std::cerr << "Error: --loss requires a value" << std::endl;
                    return 1;
                }
            }
        }

        auto facenet = model::FaceNet(num_classes, embedding_dim, loss_type, 64.0, 0.5 , pretrained_resnet);

        if (resume) {
            torch::load(facenet, get_model_save_path());
            std::cout << "Resuming from checkpoint." << std::endl;
        } else {
            std::cout << "Training from scratch with model_lr=" << model_lr
                      << " model_lastlayer_lr=" << model_lastlayer_lr
                      << " arcface_lr=" << arcface_lr
                      << " loss=" << mining_mode << std::endl;
        }
        facenet->to(device);
        facenet->train();

        for (auto& p : facenet->backbone->projector->parameters()) backbone_params.push_back(p);
        
        for (auto& p : facenet->arcface->parameters())  arcface_params.push_back(p);

        for (auto& p : facenet->backbone->head->parameters()) head_params.push_back(p);

        torch::optim::Adam optimizer_facenet({
            torch::optim::OptimizerParamGroup(backbone_params, std::make_unique<torch::optim::AdamOptions>(model_lr)),
            torch::optim::OptimizerParamGroup(head_params, std::make_unique<torch::optim::AdamOptions>(model_lastlayer_lr)),
            torch::optim::OptimizerParamGroup(arcface_params,  std::make_unique<torch::optim::AdamOptions>(arcface_lr))
        });
        auto scheduler_facenet = torch::optim::StepLR(optimizer_facenet, 20, 0.75);

        
        for (int64_t epoch = 1; epoch <= epochs; ++epoch) {
            
            double  epoch_loss       = 0.0;
            int64_t batch_index      = 0;
            double  pos_dist_sum     = 0.0;
            double  neg_dist_sum     = 0.0;
            
            facenet->train();
            std::cout << "\n=== Epoch " << epoch << "/" << epochs
                      << " | Mining: " << mining_mode
                      << " | P=" << P << " K=" << K << " ===" << std::endl;


            int64_t total_batches    = sampler.num_batches();
            for (int64_t b = 0; b < total_batches; ++b) {

                
                auto indices = sampler.next_batch();
                std::vector<torch::Tensor> batch_images;
                std::vector<torch::Tensor> batch_labels;

                
                batch_images.reserve(indices.size());
                batch_labels.reserve(indices.size());

                for (size_t idx : indices) {
                    auto sample = raw_dataset.get(idx);
                    batch_images.push_back(sample.data);
                    batch_labels.push_back(sample.target.to(torch::kInt64).squeeze());
                }

                auto inputs = torch::stack(batch_images).to(device);
                auto labels = torch::stack(batch_labels).to(device);

                optimizer_facenet.zero_grad();

                auto metrics    = facenet->forward(inputs, labels, loss_type, epoch);
                auto loss       = metrics.loss;
                double loss_val = loss.item<double>();

                if (!std::isfinite(loss_val)) {
                    std::cerr << "Warning: non-finite loss at batch " << batch_index << ": " << loss_val << std::endl;
                    nan_loss_counter++;
                    ++batch_index;
                    continue;
                }

                zero_triplet_counter += metrics.num_zero_loss_triplets;

                loss.backward();
                optimizer_facenet.step();

                epoch_loss   += loss_val;
                pos_dist_sum += metrics.avg_pos_metric;
                neg_dist_sum += metrics.avg_neg_metric;
                ++batch_index;

                if (batch_index % log_step == 0) {
                    double gap = metrics.avg_neg_metric - metrics.avg_pos_metric;
                    std::cout << "Epoch [" << epoch << "/" << epochs << "] "
                              << "Batch [" << batch_index << "/" << total_batches << "] "
                              << "(P=" << P << " x K=" << K << ") "
                              << "Loss: "        << loss_val
                              << " | Pos-dist: " << metrics.avg_pos_metric
                              << " | Neg-dist: " << metrics.avg_neg_metric
                              << " | Gap(N-P): " << gap
                              << " | Valid-triplets: " << metrics.num_valid_triplets
                              << " | Zero-triplets: " << metrics.num_zero_loss_triplets
                              << std::endl;
                }

                if (batch_index % 1000 == 0) {
                    torch::save(facenet, get_model_save_path());
                    std::cout << "Checkpoint saved." << std::endl;
                }
            }

            double avg_loss     = batch_index > 0 ? epoch_loss    / batch_index : 0.0;
            double avg_pos_dist = batch_index > 0 ? pos_dist_sum  / batch_index : 0.0;
            double avg_neg_dist = batch_index > 0 ? neg_dist_sum  / batch_index : 0.0;

            std::cout << "\n=== Epoch " << epoch << " Summary ===" << std::endl;
            std::cout << "Mining mode      : " << mining_mode                   << std::endl;
            std::cout << "Batch config     : P=" << P << " x K=" << K           << std::endl;
            std::cout << "Avg Loss         : " << avg_loss                       << std::endl;
            std::cout << "Avg Pos-dist     : " << avg_pos_dist                  << std::endl;
            std::cout << "Avg Neg-dist     : " << avg_neg_dist                  << std::endl;
            std::cout << "Gap (Neg-Pos)    : " << (avg_neg_dist - avg_pos_dist) << std::endl;
            std::cout << "Batches processed: " << batch_index                   << std::endl;
            std::cout << "Zero-triplets    : " << zero_triplet_counter          << std::endl;
            std::cout << "NaN-loss count   : " << nan_loss_counter              << std::endl;

            scheduler_facenet.step();

            torch::save(facenet, get_model_save_path());
            std::cout << "Model saved at : " << get_model_save_path() << std::endl;

            std::cout << "================================\n" << std::endl;

            std::cout << "Evaluating LFW accuracy..." << std::endl;
            evaluate_lfw(facenet, device);
        }

        std::cout << "Training complete." << std::endl;
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "Fatal Error: " << ex.what() << std::endl;
        return 1;
    }
}