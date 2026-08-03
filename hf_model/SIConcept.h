#ifndef SI_CONCEPT_H
#define SI_CONCEPT_H

#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <random>
#include <algorithm>

class SIConcept {
public:
    struct Embedding {
        std::vector<double> values;
        
        Embedding(size_t dim = 64) : values(dim, 0.0) {}
        
        double dot(const Embedding& other) const {
            double sum = 0.0;
            for (size_t i = 0; i < values.size(); ++i) {
                sum += values[i] * other.values[i];
            }
            return sum;
        }
        
        double magnitude() const {
            double sum = 0.0;
            for (double v : values) sum += v * v;
            return std::sqrt(sum);
        }
        
        double cosine(const Embedding& other) const {
            double mag = magnitude() * other.magnitude();
            if (mag == 0.0) return 0.0;
            return dot(other) / mag;
        }
        
        Embedding operator+(const Embedding& other) const {
            Embedding result(values.size());
            for (size_t i = 0; i < values.size(); ++i) {
                result.values[i] = values[i] + other.values[i];
            }
            return result;
        }
        
        Embedding operator-(const Embedding& other) const {
            Embedding result(values.size());
            for (size_t i = 0; i < values.size(); ++i) {
                result.values[i] = values[i] - other.values[i];
            }
            return result;
        }
        
        Embedding operator*(double scalar) const {
            Embedding result(values.size());
            for (size_t i = 0; i < values.size(); ++i) {
                result.values[i] = values[i] * scalar;
            }
            return result;
        }
        
        void normalize() {
            double mag = magnitude();
            if (mag > 0.0) {
                for (auto& v : values) v /= mag;
            }
        }
    };

private:
    std::unordered_map<std::string, Embedding> embeddings_;
    size_t dimension_;
    std::mt19937 rng_;
    
public:
    SIConcept(size_t dim = 64, uint32_t seed = 42) : dimension_(dim), rng_(seed) {}
    
    void learnWord(const std::string& word) {
        if (embeddings_.find(word) != embeddings_.end()) return;
        
        Embedding emb(dimension_);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (auto& v : emb.values) v = dist(rng_);
        emb.normalize();
        embeddings_[word] = std::move(emb);
    }
    
    void learnRelationship(const std::string& word1, const std::string& word2, double strength = 0.1) {
        learnWord(word1);
        learnWord(word2);
        
        Embedding& emb1 = embeddings_[word1];
        Embedding& emb2 = embeddings_[word2];
        
        Embedding diff = emb2 - emb1;
        emb1 = emb1 + diff * strength;
        emb2 = emb2 - diff * strength;
        
        emb1.normalize();
        emb2.normalize();
    }
    
    double similarity(const std::string& word1, const std::string& word2) const {
        auto it1 = embeddings_.find(word1);
        auto it2 = embeddings_.find(word2);
        if (it1 == embeddings_.end() || it2 == embeddings_.end()) return 0.0;
        return it1->second.cosine(it2->second);
    }
    
    std::vector<std::pair<std::string, double>> findSimilar(const std::string& word, size_t topN = 5) const {
        auto it = embeddings_.find(word);
        if (it == embeddings_.end()) return {};
        
        std::vector<std::pair<std::string, double>> results;
        for (const auto& [other, emb] : embeddings_) {
            if (other == word) continue;
            double sim = it->second.cosine(emb);
            results.emplace_back(other, sim);
        }
        
        std::partial_sort(results.begin(), 
                         results.begin() + std::min(topN, results.size()),
                         results.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });
        
        if (results.size() > topN) results.resize(topN);
        return results;
    }
    
    void learnFromContext(const std::vector<std::string>& words) {
        for (size_t i = 0; i < words.size(); ++i) {
            learnWord(words[i]);
            size_t window = 3;
            for (size_t j = std::max(0, (int)i - (int)window); j < std::min(words.size(), i + window + 1); ++j) {
                if (i != j) {
                    learnRelationship(words[i], words[j], 0.05 / std::abs((int)i - (int)j));
                }
            }
        }
    }
    
    Embedding getEmbedding(const std::string& word) const {
        auto it = embeddings_.find(word);
        if (it != embeddings_.end()) return it->second;
        return Embedding(dimension_);
    }
};

#endif