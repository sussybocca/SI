#ifndef SI_CURIOSITY_H
#define SI_CURIOSITY_H

#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>

class SICuriosity {
private:
    struct UnknownConcept {
        std::string word;
        size_t encounterCount;
        double curiosityScore;
        std::vector<std::string> contexts;
    };
    
    std::unordered_map<std::string, UnknownConcept> unknownWords_;
    std::vector<std::string> questions_;
    double baseCuriosity_;
    std::mt19937 rng_;
    
public:
    SICuriosity(double curiosity = 0.5, uint32_t seed = 42) 
        : baseCuriosity_(curiosity), rng_(seed) {}
    
    void encounterUnknown(const std::string& word, const std::string& context) {
        auto& unknown = unknownWords_[word];
        unknown.word = word;
        unknown.encounterCount++;
        unknown.contexts.push_back(context);
        
        // More encounters = more curiosity
        unknown.curiosityScore = baseCuriosity_ * std::log(1.0 + unknown.encounterCount);
        
        if (unknown.contexts.size() > 10) {
            unknown.contexts.erase(unknown.contexts.begin());
        }
    }
    
    std::string generateQuestion() {
        if (unknownWords_.empty()) return "";
        
        // Find most curious unknown concept
        auto mostCurious = std::max_element(unknownWords_.begin(), unknownWords_.end(),
            [](const auto& a, const auto& b) {
                return a.second.curiosityScore < b.second.curiosityScore;
            });
        
        if (mostCurious->second.curiosityScore < 0.3) return "";
        
        std::string word = mostCurious->first;
        std::string context;
        if (!mostCurious->second.contexts.empty()) {
            std::uniform_int_distribution<size_t> dist(0, mostCurious->second.contexts.size() - 1);
            context = mostCurious->second.contexts[dist(rng_)];
        }
        
        // Reduce curiosity after asking
        mostCurious->second.curiosityScore *= 0.5;
        
        if (!context.empty()) {
            return "What does '" + word + "' mean in the context of: " + context + "?";
        }
        return "What does '" + word + "' mean?";
    }
    
    void learnAnswer(const std::string& word, const std::string& explanation) {
        unknownWords_.erase(word);
    }
    
    bool hasQuestions() const {
        return !unknownWords_.empty();
    }
    
    void setCuriosity(double level) {
        baseCuriosity_ = std::max(0.0, std::min(1.0, level));
    }
};

#endif