#ifndef SI_REASONING_H
#define SI_REASONING_H

#include "SIConcept.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>

struct Thought {
    std::string concept;
    double activation;
    double relevance;
    std::vector<std::string> relatedConcepts;
};

class SIReasoning {
private:
    SIConcept& concepts_;
    std::vector<Thought> workingMemory_;
    size_t maxThoughts_;
    double attentionThreshold_;
    
    std::unordered_map<std::string, std::vector<std::string>> logicalRelations_;
    
public:
    SIReasoning(SIConcept& concepts, size_t maxThoughts = 20, double threshold = 0.3)
        : concepts_(concepts), maxThoughts_(maxThoughts), attentionThreshold_(threshold) {}
    
    void addLogicalRelation(const std::string& a, const std::string& relation, const std::string& b) {
        logicalRelations_[a + "|" + relation].push_back(b);
    }
    
    std::vector<std::string> reason(const std::vector<std::string>& input, size_t depth = 2) {
        workingMemory_.clear();
        
        // Seed working memory with input concepts
        for (const auto& word : input) {
            Thought t;
            t.concept = word;
            t.activation = 1.0;
            t.relevance = 1.0;
            workingMemory_.push_back(t);
        }
        
        // Spread activation through similar concepts
        for (size_t step = 0; step < depth; ++step) {
            std::vector<Thought> newThoughts;
            
            for (auto& thought : workingMemory_) {
                auto similar = concepts_.findSimilar(thought.concept, 5);
                for (const auto& [word, sim] : similar) {
                    if (sim > attentionThreshold_) {
                        Thought newThought;
                        newThought.concept = word;
                        newThought.activation = thought.activation * sim * 0.7;
                        newThought.relevance = sim;
                        
                        bool exists = false;
                        for (auto& existing : workingMemory_) {
                            if (existing.concept == word) {
                                existing.activation += newThought.activation;
                                exists = true;
                                break;
                            }
                        }
                        if (!exists) newThoughts.push_back(newThought);
                    }
                }
            }
            
            workingMemory_.insert(workingMemory_.end(), newThoughts.begin(), newThoughts.end());
            
            if (workingMemory_.size() > maxThoughts_) {
                std::partial_sort(workingMemory_.begin(), 
                                workingMemory_.begin() + maxThoughts_,
                                workingMemory_.end(),
                                [](const Thought& a, const Thought& b) {
                                    return a.activation > b.activation;
                                });
                workingMemory_.resize(maxThoughts_);
            }
        }
        
        std::vector<std::string> conclusions;
        for (const auto& thought : workingMemory_) {
            if (thought.activation > attentionThreshold_) {
                conclusions.push_back(thought.concept);
            }
        }
        return conclusions;
    }
    
    std::string generateNovelResponse(const std::vector<std::string>& input) {
        auto activated = reason(input);
        
        // Combine activated concepts into a novel response
        std::string response;
        for (const auto& concept : activated) {
            auto similar = concepts_.findSimilar(concept, 1);
            if (!similar.empty() && similar[0].second > 0.5) {
                response += similar[0].first + " ";
            }
        }
        
        return response.empty() ? "I need more examples to understand." : response;
    }
};

#endif