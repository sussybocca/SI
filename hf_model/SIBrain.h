#ifndef SI_BRAIN_H
#define SI_BRAIN_H

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct SIBrainNode {
    std::unordered_map<std::string, std::unique_ptr<SIBrainNode>> children;
    uint64_t count = 0;
    uint64_t totalOutCount = 0;

    void save(std::ostream& os) const;
    static std::unique_ptr<SIBrainNode> load(std::istream& is, uint32_t version);
    size_t subtreeNodeCount() const;
    void recomputeTotalOutCount();
};

class SIBrain {
public:
    struct Config {
        size_t maxNodes;
        size_t maxContextDepth;
        double dopamineBoost;
        size_t maxGenerateTokens;
        uint32_t randomSeed;
        double temperature;
        size_t topK;
        
        Config();
    };

    explicit SIBrain(const Config& cfg = Config());
    ~SIBrain() = default;

    void learnSequence(const std::vector<std::string>& tokens);
    std::vector<std::string> generate(const std::vector<std::string>& prompt) const;
    void rewardSequence(const std::vector<std::string>& tokens);

    static constexpr uint32_t BRAIN_FILE_MAGIC   = 0x53494252;
    static constexpr uint32_t BRAIN_FILE_VERSION = 1;

    void save(const std::string& filepath) const;
    void load(const std::string& filepath);

    size_t totalNodes() const;
    size_t memoryUsageEstimate() const;

    static void writePortable(std::ostream& os, uint64_t val);
    static uint64_t readPortable(std::istream& is);

private:
    mutable std::shared_mutex mutex_;
    Config cfg_;
    std::unique_ptr<SIBrainNode> root_;
    size_t nodeCount_ = 0;
    mutable std::mt19937_64 rng_;

    void addTokenSequence(const std::vector<std::string>& tokens, uint64_t countInc);
    std::string sampleNext(const std::vector<std::string>& context) const;
    void enforceMemoryLimit();
};

#endif