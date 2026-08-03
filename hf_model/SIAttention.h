#ifndef SI_ATTENTION_H
#define SI_ATTENTION_H

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <unordered_map>

class SIAttention {
public:
    struct AttentionWeight {
        std::string token;
        double weight;
        size_t position;
    };

private:
    std::vector<std::string> contextBuffer_;
    size_t maxContextSize_;
    std::unordered_map<std::string, double> importanceScores_;
    
    double computeTFIDF(const std::string& token) const {
        if (contextBuffer_.empty()) return 0.0;
        
        size_t termFreq = 0;
        size_t docFreq = 0;
        bool counted = false;
        
        for (const auto& t : contextBuffer_) {
            if (t == token) termFreq++;
        }
        
        double tf = static_cast<double>(termFreq) / contextBuffer_.size();
        double idf = std::log(static_cast<double>(contextBuffer_.size()) / (1.0 + termFreq));
        
        return tf * idf;
    }
    
public:
    SIAttention(size_t maxContext = 1000) : maxContextSize_(maxContext) {}
    
    void addToContext(const std::string& token) {
        contextBuffer_.push_back(token);
        if (contextBuffer_.size() > maxContextSize_) {
            contextBuffer_.erase(contextBuffer_.begin());
        }
        
        double score = computeTFIDF(token);
        importanceScores_[token] = importanceScores_[token] * 0.9 + score * 0.1;
    }
    
    std::vector<AttentionWeight> attend(const std::vector<std::string>& query) {
        std::vector<AttentionWeight> weights;
        
        for (size_t i = 0; i < contextBuffer_.size(); ++i) {
            const auto& token = contextBuffer_[i];
            
            // Position bias - recent tokens get higher weight
            double positionBias = 1.0 - std::exp(-0.1 * (contextBuffer_.size() - i));
            
            // Content matching - query terms get higher weight
            double contentMatch = 0.0;
            for (const auto& q : query) {
                if (token == q) {
                    contentMatch = 1.0;
                    break;
                }
            }
            
            // Importance from frequency
            double importance = importanceScores_[token];
            
            // Combined attention weight
            double weight = 0.4 * positionBias + 0.4 * contentMatch + 0.2 * importance;
            
            weights.push_back({token, weight, i});
        }
        
        // Softmax normalization
        double maxWeight = 0.0;
        for (const auto& w : weights) maxWeight = std::max(maxWeight, w.weight);
        
        double sum = 0.0;
        for (auto& w : weights) {
            w.weight = std::exp(w.weight - maxWeight);
            sum += w.weight;
        }
        
        if (sum > 0.0) {
            for (auto& w : weights) w.weight /= sum;
        }
        
        // Sort by weight descending
        std::sort(weights.begin(), weights.end(),
                  [](const AttentionWeight& a, const AttentionWeight& b) {
                      return a.weight > b.weight;
                  });
        
        return weights;
    }
    
    std::vector<std::string> getFocusedContext(const std::vector<std::string>& query, size_t topK = 10) {
        auto attended = attend(query);
        std::vector<std::string> focused;
        
        for (size_t i = 0; i < std::min(topK, attended.size()); ++i) {
            if (attended[i].weight > 0.01) {
                focused.push_back(attended[i].token);
            }
        }
        
        return focused;
    }
    
    void clear() {
        contextBuffer_.clear();
        importanceScores_.clear();
    }
};

#endif