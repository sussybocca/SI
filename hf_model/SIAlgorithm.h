#ifndef SI_ALGORITHM_H
#define SI_ALGORITHM_H

#include "SIBrain.h"
#include "SIConcept.h"
#include "SITrainer.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>

class SIAlgorithm {
public:
    struct IndexedKnowledge {
        std::string text;
        std::vector<std::string> tokens;
        std::set<std::string> keywords;
        double importance;
        int accessCount;
        int positiveFeedback;
        int negativeFeedback;
        std::chrono::system_clock::time_point lastAccessed;
        std::vector<std::string> categories;
    };
    
    struct UserPreference {
        std::string topic;
        double weight;
        int interactionCount;
        std::chrono::system_clock::time_point lastInteraction;
    };
    
private:
    std::vector<IndexedKnowledge> knowledgeIndex_;
    std::unordered_map<std::string, std::vector<size_t>> keywordIndex_;
    std::unordered_map<std::string, std::vector<size_t>> categoryIndex_;
    std::unordered_map<std::string, UserPreference> userPreferences_;
    std::unordered_map<std::string, double> wordImportance_;
    SIBrain* brain_;
    SIConcept* concepts_;
    
    // Common English stop words to filter out
    std::set<std::string> stopWords_ = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "i", "you", "he",
        "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
        "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours",
        "theirs", "this", "that", "these", "those", "am", "at", "by", "for",
        "with", "about", "to", "from", "in", "on", "of", "and", "or", "not",
        "but", "if", "then", "else", "when", "up", "down", "out", "no", "so"
    };
    
    // Question words for intent detection
    std::set<std::string> questionWords_ = {
        "what", "who", "how", "why", "when", "where", "which", "whom",
        "whose", "tell", "explain", "describe", "define", "elaborate"
    };
    
    // Positive/negative sentiment words
    std::set<std::string> positiveWords_ = {
        "good", "great", "awesome", "excellent", "amazing", "love", "like",
        "best", "wonderful", "fantastic", "brilliant", "perfect", "yes",
        "correct", "right", "true", "helpful", "useful", "nice", "cool"
    };
    
    std::set<std::string> negativeWords_ = {
        "bad", "terrible", "awful", "horrible", "hate", "worst", "wrong",
        "false", "incorrect", "no", "not", "never", "useless", "stupid",
        "boring", "slow", "broken", "error", "fail", "failure"
    };
    
    std::vector<std::string> extractKeywords(const std::vector<std::string>& tokens) {
        std::vector<std::string> keywords;
        std::set<std::string> seen;
        
        for (const auto& token : tokens) {
            std::string lower = token;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            
            if (stopWords_.find(lower) == stopWords_.end() && 
                token.size() > 1 &&
                seen.find(lower) == seen.end()) {
                keywords.push_back(token);
                seen.insert(lower);
            }
        }
        
        return keywords;
    }
    
    std::vector<std::string> detectCategories(const std::string& text) {
        std::vector<std::string> categories;
        std::string lower = text;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        
        // Category detection patterns
        std::vector<std::pair<std::string, std::vector<std::string>>> categoryPatterns = {
            {"technology", {"computer", "software", "hardware", "algorithm", "programming", "code", "data", "digital", "tech", "ai", "artificial intelligence", "machine learning"}},
            {"science", {"physics", "chemistry", "biology", "science", "scientific", "experiment", "theory", "quantum", "molecule", "atom", "cell", "organism"}},
            {"history", {"ancient", "medieval", "century", "war", "empire", "kingdom", "revolution", "civilization", "historical"}},
            {"philosophy", {"philosophy", "ethics", "morality", "existence", "consciousness", "reality", "truth", "knowledge", "wisdom", "meaning"}},
            {"art", {"art", "music", "painting", "sculpture", "literature", "poetry", "film", "movie", "theater", "dance", "creative"}},
            {"nature", {"animal", "plant", "forest", "ocean", "mountain", "river", "climate", "weather", "earth", "environment"}},
            {"human", {"human", "person", "people", "society", "culture", "language", "psychology", "emotion", "behavior", "mind", "brain"}},
            {"space", {"space", "planet", "star", "galaxy", "universe", "cosmos", "astronomy", "nasa", "orbit", "moon", "sun", "mars"}},
            {"math", {"math", "mathematics", "number", "equation", "calculus", "algebra", "geometry", "statistics", "probability"}},
            {"health", {"health", "medicine", "disease", "doctor", "hospital", "treatment", "therapy", "diet", "exercise", "fitness"}}
        };
        
        for (const auto& [category, keywords] : categoryPatterns) {
            for (const auto& keyword : keywords) {
                if (lower.find(keyword) != std::string::npos) {
                    categories.push_back(category);
                    break;
                }
            }
        }
        
        return categories;
    }
    
    double calculateRelevance(const IndexedKnowledge& knowledge, const std::vector<std::string>& queryTokens) {
        double relevance = 0.0;
        std::set<std::string> querySet(queryTokens.begin(), queryTokens.end());
        
        // Keyword matching
        for (const auto& kw : knowledge.keywords) {
            std::string lower = kw;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            if (querySet.find(kw) != querySet.end() || querySet.find(lower) != querySet.end()) {
                relevance += 3.0;
            }
        }
        
        // Token overlap
        for (const auto& kt : knowledge.tokens) {
            for (const auto& qt : queryTokens) {
                if (kt == qt) relevance += 1.0;
            }
        }
        
        // Importance bonus
        relevance += knowledge.importance;
        
        // Recency bonus - exponential decay
        auto now = std::chrono::system_clock::now();
        auto ageInHours = std::chrono::duration_cast<std::chrono::hours>(now - knowledge.lastAccessed).count();
        double recencyBonus = std::exp(-ageInHours / 24.0);
        relevance += recencyBonus * 2.0;
        
        // User preference bonus
        for (const auto& cat : knowledge.categories) {
            auto prefIt = userPreferences_.find(cat);
            if (prefIt != userPreferences_.end()) {
                relevance += prefIt->second.weight * 2.0;
            }
        }
        
        // Feedback score
        int totalFeedback = knowledge.positiveFeedback + knowledge.negativeFeedback;
        if (totalFeedback > 0) {
            double feedbackRatio = (double)knowledge.positiveFeedback / totalFeedback;
            relevance += feedbackRatio * 3.0;
        }
        
        // Access count bonus (popular knowledge)
        relevance += std::log(1.0 + knowledge.accessCount) * 0.5;
        
        return relevance;
    }
    
    std::string detectIntent(const std::string& input) {
        auto tokens = SITrainer::tokenize(input);
        if (tokens.empty()) return "statement";
        
        std::string first = tokens[0];
        std::transform(first.begin(), first.end(), first.begin(), ::tolower);
        
        if (questionWords_.find(first) != questionWords_.end()) return "question";
        if (input.back() == '?') return "question";
        if (first == "::teach") return "teach_command";
        if (first == "::save" || first == "::quit" || first == "::info") return "command";
        
        // Check for commands/requests
        if (first == "show" || first == "find" || first == "search" || first == "get") return "request";
        if (first == "hi" || first == "hello" || first == "hey") return "greeting";
        if (first == "thanks" || first == "thank") return "gratitude";
        
        return "statement";
    }
    
    double detectSentiment(const std::string& input) {
        auto tokens = SITrainer::tokenize(input);
        double sentiment = 0.0;
        int wordCount = 0;
        
        for (const auto& token : tokens) {
            std::string lower = token;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            
            if (positiveWords_.find(lower) != positiveWords_.end()) {
                sentiment += 1.0;
                wordCount++;
            }
            if (negativeWords_.find(lower) != negativeWords_.end()) {
                sentiment -= 1.0;
                wordCount++;
            }
        }
        
        if (wordCount > 0) return sentiment / wordCount;
        return 0.0;
    }
    
public:
    SIAlgorithm(SIBrain* brain = nullptr, SIConcept* concepts = nullptr) 
        : brain_(brain), concepts_(concepts) {}
    
    void setBrain(SIBrain* brain) { brain_ = brain; }
    void setConcepts(SIConcept* concepts) { concepts_ = concepts; }
    
    void index(const std::string& text) {
        auto tokens = SITrainer::tokenize(text);
        if (tokens.empty()) return;
        
        IndexedKnowledge knowledge;
        knowledge.text = text;
        knowledge.tokens = tokens;
        
        auto keywords = extractKeywords(tokens);
        for (const auto& kw : keywords) {
            knowledge.keywords.insert(kw);
        }
        
        knowledge.categories = detectCategories(text);
        knowledge.importance = 1.0;
        knowledge.accessCount = 0;
        knowledge.positiveFeedback = 0;
        knowledge.negativeFeedback = 0;
        knowledge.lastAccessed = std::chrono::system_clock::now();
        
        size_t index = knowledgeIndex_.size();
        knowledgeIndex_.push_back(knowledge);
        
        // Build keyword index
        for (const auto& kw : knowledge.keywords) {
            std::string lower = kw;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            keywordIndex_[lower].push_back(index);
        }
        
        // Build category index
        for (const auto& cat : knowledge.categories) {
            categoryIndex_[cat].push_back(index);
        }
    }
    
    std::vector<std::string> search(const std::string& query, size_t maxResults = 5) {
        auto queryTokens = SITrainer::tokenize(query);
        if (queryTokens.empty()) return {};
        
        std::vector<std::pair<size_t, double>> scoredResults;
        std::set<size_t> candidateSet;
        
        // Find candidates through keyword index
        for (const auto& qt : queryTokens) {
            std::string lower = qt;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            
            auto it = keywordIndex_.find(lower);
            if (it != keywordIndex_.end()) {
                for (size_t idx : it->second) {
                    candidateSet.insert(idx);
                }
            }
        }
        
        // Score each candidate
        for (size_t idx : candidateSet) {
            knowledgeIndex_[idx].accessCount++;
            knowledgeIndex_[idx].lastAccessed = std::chrono::system_clock::now();
            
            double relevance = calculateRelevance(knowledgeIndex_[idx], queryTokens);
            scoredResults.push_back({idx, relevance});
        }
        
        // Sort by relevance
        std::sort(scoredResults.begin(), scoredResults.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Get top results
        std::vector<std::string> results;
        for (size_t i = 0; i < std::min(maxResults, scoredResults.size()); ++i) {
            if (scoredResults[i].second > 0.5) {
                results.push_back(knowledgeIndex_[scoredResults[i].first].text);
            }
        }
        
        return results;
    }
    
    std::string getBestResponse(const std::string& input) {
        std::string intent = detectIntent(input);
        double sentiment = detectSentiment(input);
        
        // Update user preferences based on topic
        auto tokens = SITrainer::tokenize(input);
        auto categories = detectCategories(input);
        for (const auto& cat : categories) {
            auto& pref = userPreferences_[cat];
            pref.topic = cat;
            pref.weight += 0.1;
            pref.interactionCount++;
            pref.lastInteraction = std::chrono::system_clock::now();
        }
        
        // Handle different intents
        if (intent == "greeting") {
            return getGreeting(sentiment);
        }
        
        if (intent == "gratitude") {
            return getGratitudeResponse();
        }
        
        // Search for relevant knowledge
        auto results = search(input);
        
        if (!results.empty()) {
            // Give feedback based on sentiment
            if (sentiment > 0.3) {
                for (size_t i = 0; i < std::min(results.size(), (size_t)3); ++i) {
                    // Find and boost the best results
                    for (auto& knowledge : knowledgeIndex_) {
                        if (knowledge.text == results[i]) {
                            knowledge.positiveFeedback++;
                            break;
                        }
                    }
                }
            } else if (sentiment < -0.3) {
                for (auto& knowledge : knowledgeIndex_) {
                    if (knowledge.text == results[0]) {
                        knowledge.negativeFeedback++;
                        break;
                    }
                }
            }
            
            return getBestMatch(results, tokens);
        }
        
        // No results - ask for teaching
        if (intent == "question") {
            std::string keyTerms;
            for (const auto& t : tokens) {
                std::string lower = t;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                if (stopWords_.find(lower) == stopWords_.end() && t.size() > 2) {
                    keyTerms += t + " ";
                }
            }
            return "I don't know about " + keyTerms + "yet. Can you teach me? Use ::teach <word> <meaning>";
        }
        
        // Echo back for statements
        if (brain_) {
            auto generated = brain_->generate(tokens);
            if (!generated.empty() && generated.size() > 2) {
                std::string response;
                for (const auto& t : generated) response += t + " ";
                if (!response.empty()) response.pop_back();
                return response;
            }
        }
        
        return "Interesting. Tell me more.";
    }
    
    std::string getBestMatch(const std::vector<std::string>& results, const std::vector<std::string>& queryTokens) {
        if (results.empty()) return "";
        
        // Score each result
        std::vector<std::pair<std::string, double>> scored;
        for (const auto& result : results) {
            double score = 0.0;
            auto resultTokens = SITrainer::tokenize(result);
            
            // Prefer results that contain query words
            for (const auto& rt : resultTokens) {
                for (const auto& qt : queryTokens) {
                    if (rt == qt) score += 2.0;
                }
            }
            
            // Prefer longer, more informative results
            score += std::log(resultTokens.size() + 1);
            
            // Prefer results with positive feedback
            for (const auto& knowledge : knowledgeIndex_) {
                if (knowledge.text == result) {
                    score += knowledge.importance;
                    score += knowledge.positiveFeedback * 0.5;
                    score -= knowledge.negativeFeedback * 0.5;
                    break;
                }
            }
            
            scored.push_back({result, score});
        }
        
        std::sort(scored.begin(), scored.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        
        return scored[0].first;
    }
    
    std::string getGreeting(double sentiment) {
        std::vector<std::string> greetings;
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto localTime = *std::localtime(&time_t);
        int hour = localTime.tm_hour;
        
        if (hour < 12) {
            greetings = {"Good morning!", "Hello! Nice to see you this morning.", "Good morning! Ready to learn?"};
        } else if (hour < 17) {
            greetings = {"Good afternoon!", "Hello there!", "Good afternoon! How can I help?"};
        } else {
            greetings = {"Good evening!", "Hello! Good evening.", "Evening! What shall we discuss?"};
        }
        
        if (sentiment > 0.5) {
            greetings.push_back("You seem happy today! That's great!");
        }
        
        return greetings[rand() % greetings.size()];
    }
    
    std::string getGratitudeResponse() {
        std::vector<std::string> responses = {
            "You're welcome!",
            "Glad I could help!",
            "Happy to assist!",
            "Anytime!",
            "My pleasure!"
        };
        return responses[rand() % responses.size()];
    }
    
    void provideFeedback(const std::string& text, bool positive) {
        for (auto& knowledge : knowledgeIndex_) {
            if (knowledge.text == text) {
                if (positive) knowledge.positiveFeedback++;
                else knowledge.negativeFeedback++;
                
                // Update importance
                int total = knowledge.positiveFeedback + knowledge.negativeFeedback;
                knowledge.importance = 1.0 + (double)knowledge.positiveFeedback / (total + 1);
                break;
            }
        }
    }
    
    std::vector<std::string> getSuggestedTopics() {
        std::vector<std::pair<std::string, double>> suggestions;
        
        for (const auto& [topic, pref] : userPreferences_) {
            double score = pref.weight * pref.interactionCount;
            auto ageInHours = std::chrono::duration_cast<std::chrono::hours>(
                std::chrono::system_clock::now() - pref.lastInteraction).count();
            score *= std::exp(-ageInHours / 168.0); // Weekly decay
            suggestions.push_back({topic, score});
        }
        
        std::sort(suggestions.begin(), suggestions.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::vector<std::string> topics;
        for (size_t i = 0; i < std::min(suggestions.size(), (size_t)5); ++i) {
            topics.push_back(suggestions[i].first);
        }
        
        return topics;
    }
    
    void save(const std::string& path) {
        std::ofstream out(path, std::ios::binary);
        if (!out) return;
        
        size_t knowledgeSize = knowledgeIndex_.size();
        out.write(reinterpret_cast<const char*>(&knowledgeSize), sizeof(knowledgeSize));
        
        for (const auto& k : knowledgeIndex_) {
            size_t textSize = k.text.size();
            out.write(reinterpret_cast<const char*>(&textSize), sizeof(textSize));
            out.write(k.text.c_str(), textSize);
            out.write(reinterpret_cast<const char*>(&k.importance), sizeof(k.importance));
            out.write(reinterpret_cast<const char*>(&k.positiveFeedback), sizeof(k.positiveFeedback));
            out.write(reinterpret_cast<const char*>(&k.negativeFeedback), sizeof(k.negativeFeedback));
        }
    }
    
    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) return;
        
        knowledgeIndex_.clear();
        keywordIndex_.clear();
        
        size_t knowledgeSize;
        in.read(reinterpret_cast<char*>(&knowledgeSize), sizeof(knowledgeSize));
        
        for (size_t i = 0; i < knowledgeSize; ++i) {
            size_t textSize;
            in.read(reinterpret_cast<char*>(&textSize), sizeof(textSize));
            std::string text(textSize, '\0');
            in.read(&text[0], textSize);
            
            index(text);
            
            double importance;
            int pos, neg;
            in.read(reinterpret_cast<char*>(&importance), sizeof(importance));
            in.read(reinterpret_cast<char*>(&pos), sizeof(pos));
            in.read(reinterpret_cast<char*>(&neg), sizeof(neg));
            
            knowledgeIndex_.back().importance = importance;
            knowledgeIndex_.back().positiveFeedback = pos;
            knowledgeIndex_.back().negativeFeedback = neg;
        }
    }
    
    size_t getKnowledgeCount() const { return knowledgeIndex_.size(); }
    size_t getCategoryCount() const { return categoryIndex_.size(); }
};

#endif