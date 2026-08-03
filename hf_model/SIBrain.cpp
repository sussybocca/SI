#include "SIBrain.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>
#include <vector>
// Config constructor - MUST be here
SIBrain::Config::Config() 
    : maxNodes(10000000)
    , maxContextDepth(8)
    , dopamineBoost(10.0)
    , maxGenerateTokens(200)
    , randomSeed(42)
    , temperature(1.0)
    , topK(0) 
{}
// ========== SIBrainNode ==========
void SIBrainNode::save(std::ostream& os) const {
    SIBrain::writePortable(os, count);
    SIBrain::writePortable(os, static_cast<uint64_t>(children.size()));
    for (const auto& [token, child] : children) {
        SIBrain::writePortable(os, token.size());
        os.write(token.data(), token.size());
        child->save(os);
    }
}

std::unique_ptr<SIBrainNode> SIBrainNode::load(std::istream& is, uint32_t version) {
    auto node = std::make_unique<SIBrainNode>();
    node->count = SIBrain::readPortable(is);
    uint64_t numChildren = SIBrain::readPortable(is);
    for (uint64_t i = 0; i < numChildren; ++i) {
        size_t len = static_cast<size_t>(SIBrain::readPortable(is));
        std::string token(len, '\0');
        is.read(&token[0], len);
        node->children[token] = load(is, version);
    }
    node->recomputeTotalOutCount();
    return node;
}

size_t SIBrainNode::subtreeNodeCount() const {
    size_t total = 1; // this node
    for (const auto& [_, child] : children) {
        total += child->subtreeNodeCount();
    }
    return total;
}

void SIBrainNode::recomputeTotalOutCount() {
    totalOutCount = 0;
    for (const auto& [_, child] : children) {
        totalOutCount += child->count;
    }
}

// ========== SIBrain ==========
SIBrain::SIBrain(const Config& cfg)
    : cfg_(cfg), root_(std::make_unique<SIBrainNode>()), nodeCount_(1), rng_(cfg.randomSeed) {}

size_t SIBrain::totalNodes() const {
    std::shared_lock lock(mutex_);
    return nodeCount_;
}

size_t SIBrain::memoryUsageEstimate() const {
    std::shared_lock lock(mutex_);
    // Rough: each node ~ sizeof(SIBrainNode) + unordered_map overhead + children pointers
    return nodeCount_ * (sizeof(SIBrainNode) + 64);
}

// Portable big‑endian serialization
void SIBrain::writePortable(std::ostream& os, uint64_t val) {
    uint8_t buf[8];
    for (int i = 7; i >= 0; --i) {
        buf[i] = static_cast<uint8_t>(val & 0xFF);
        val >>= 8;
    }
    os.write(reinterpret_cast<const char*>(buf), sizeof(buf));
}

uint64_t SIBrain::readPortable(std::istream& is) {
    uint8_t buf[8];
    is.read(reinterpret_cast<char*>(buf), sizeof(buf));
    if (!is) throw std::runtime_error("Unexpected end of file during read");
    uint64_t val = 0;
    for (int i = 0; i < 8; ++i) {
        val = (val << 8) | buf[i];
    }
    return val;
}

void SIBrain::addTokenSequence(const std::vector<std::string>& tokens, uint64_t countInc) {
    if (tokens.empty()) return;
    std::unique_lock lock(mutex_);

    // Insert all suffixes up to maxContextDepth length.
    // To avoid repeated traversals, we use a sliding window approach,
    // keeping an array of pointers to the nodes at each depth.
    const size_t n = tokens.size();
    for (size_t start = 0; start < n; ++start) {
        SIBrainNode* current = root_.get();
        size_t depth = 0;
        for (size_t i = start; i < n && depth < cfg_.maxContextDepth; ++i, ++depth) {
            const std::string& token = tokens[i];
            auto it = current->children.find(token);
            if (it == current->children.end()) {
                // Enforce memory cap before allocation
                if (nodeCount_ >= cfg_.maxNodes) {
                    enforceMemoryLimit();
                }
                auto newNode = std::make_unique<SIBrainNode>();
                SIBrainNode* raw = newNode.get();
                current->children[token] = std::move(newNode);
                ++nodeCount_;
                current->totalOutCount += countInc; // will be updated later
                current = raw;
            } else {
                current = it->second.get();
                current->totalOutCount += countInc; // maintain totalOutCount
            }
            current->count += countInc;
        }
    }
}

void SIBrain::learnSequence(const std::vector<std::string>& tokens) {
    addTokenSequence(tokens, 1);
}

void SIBrain::rewardSequence(const std::vector<std::string>& tokens) {
    if (tokens.empty()) return;
    uint64_t boost = static_cast<uint64_t>(cfg_.dopamineBoost);
    if (boost == 0) boost = 1;
    addTokenSequence(tokens, boost);
}

std::string SIBrain::sampleNext(const std::vector<std::string>& context) const {
    std::shared_lock lock(mutex_);
    for (size_t len = std::min(context.size(), cfg_.maxContextDepth); len > 0; --len) {
        const SIBrainNode* node = root_.get();
        bool match = true;
        for (size_t i = context.size() - len; i < context.size(); ++i) {
            auto it = node->children.find(context[i]);
            if (it == node->children.end()) {
                match = false;
                break;
            }
            node = it->second.get();
        }
        if (match && !node->children.empty()) {
            // Collect weighted samples considering temperature and top‑k
            std::vector<std::pair<std::string, double>> candidates;
            candidates.reserve(node->children.size());
            for (const auto& [tok, child] : node->children) {
                candidates.emplace_back(tok, static_cast<double>(child->count));
            }

            // Top‑K filtering
            if (cfg_.topK > 0 && candidates.size() > cfg_.topK) {
                std::nth_element(candidates.begin(),
                                 candidates.begin() + cfg_.topK - 1,
                                 candidates.end(),
                                 [](const auto& a, const auto& b) { return a.second > b.second; });
                candidates.resize(cfg_.topK);
            }

            // Temperature scaling
            if (cfg_.temperature <= 0.0) {
                // deterministic: choose max
                return std::max_element(candidates.begin(), candidates.end(),
                                        [](const auto& a, const auto& b) { return a.second < b.second; })->first;
            }

            double totalWeight = 0.0;
            for (auto& [tok, w] : candidates) {
                w = std::pow(w, 1.0 / cfg_.temperature);
                totalWeight += w;
            }
            if (totalWeight == 0.0) return "";

            std::uniform_real_distribution<double> dist(0.0, totalWeight);
            double pick = dist(rng_);
            double cumulative = 0.0;
            for (const auto& [tok, w] : candidates) {
                cumulative += w;
                if (cumulative >= pick) {
                    return tok;
                }
            }
            // fallback
            return candidates.back().first;
        }
    }
    return "";
}

std::vector<std::string> SIBrain::generate(const std::vector<std::string>& prompt) const {
    std::vector<std::string> result;
    std::vector<std::string> ctx = prompt;
    result.reserve(cfg_.maxGenerateTokens);
    for (size_t i = 0; i < cfg_.maxGenerateTokens; ++i) {
        std::string next = sampleNext(ctx);
        if (next.empty()) break;
        result.push_back(next);
        ctx.push_back(next);
        if (ctx.size() > cfg_.maxContextDepth) {
            ctx.erase(ctx.begin());
        }
    }
    return result;
}

void SIBrain::enforceMemoryLimit() {
    // Aggressive pruning: repeatedly remove the child of root with smallest
    // totalOutCount until nodeCount_ < maxNodes (or root has no children).
    if (!root_ || root_->children.empty()) return;

    while (nodeCount_ >= cfg_.maxNodes && !root_->children.empty()) {
        // Find child with smallest totalOutCount (least used)
        auto minIt = std::min_element(root_->children.begin(), root_->children.end(),
            [](const auto& a, const auto& b) {
                return a.second->count < b.second->count;
            });
        if (minIt == root_->children.end()) break;

        SIBrainNode* branch = minIt->second.get();
        size_t freed = branch->subtreeNodeCount();
        nodeCount_ -= freed;
        root_->children.erase(minIt);
        // root's totalOutCount not needed for sampling; we can leave it stale
    }
}

void SIBrain::save(const std::string& filepath) const {
    std::shared_lock lock(mutex_);
    std::ofstream out(filepath, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open file for writing: " + filepath);

    // Write magic and version
    writePortable(out, BRAIN_FILE_MAGIC);
    writePortable(out, BRAIN_FILE_VERSION);

    // Write Config (all fields)
    writePortable(out, cfg_.maxNodes);
    writePortable(out, cfg_.maxContextDepth);
    // dopamineBoost as double -> store as fixed‑point? We'll write raw bytes for simplicity,
    // but ensure portability: write as uint64_t representing bits.
    uint64_t dopamineBits;
    std::memcpy(&dopamineBits, &cfg_.dopamineBoost, sizeof(dopamineBits));
    writePortable(out, dopamineBits);
    writePortable(out, cfg_.maxGenerateTokens);
    writePortable(out, cfg_.randomSeed);
    uint64_t tempBits;
    std::memcpy(&tempBits, &cfg_.temperature, sizeof(tempBits));
    writePortable(out, tempBits);
    writePortable(out, cfg_.topK);

    writePortable(out, nodeCount_);
    root_->save(out);
}

void SIBrain::load(const std::string& filepath) {
    std::unique_lock lock(mutex_);
    std::ifstream in(filepath, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file for reading: " + filepath);

    uint64_t magic = readPortable(in);
    if (magic != BRAIN_FILE_MAGIC) throw std::runtime_error("Invalid brain file (bad magic)");
    uint64_t version = readPortable(in);
    if (version != BRAIN_FILE_VERSION) throw std::runtime_error("Unsupported brain file version");

    cfg_.maxNodes = static_cast<size_t>(readPortable(in));
    cfg_.maxContextDepth = static_cast<size_t>(readPortable(in));
    uint64_t dopamineBits = readPortable(in);
    std::memcpy(&cfg_.dopamineBoost, &dopamineBits, sizeof(cfg_.dopamineBoost));
    cfg_.maxGenerateTokens = static_cast<size_t>(readPortable(in));
    cfg_.randomSeed = static_cast<uint32_t>(readPortable(in));
    uint64_t tempBits = readPortable(in);
    std::memcpy(&cfg_.temperature, &tempBits, sizeof(cfg_.temperature));
    cfg_.topK = static_cast<size_t>(readPortable(in));

    nodeCount_ = static_cast<size_t>(readPortable(in));
    root_ = SIBrainNode::load(in, static_cast<uint32_t>(version));
    rng_.seed(cfg_.randomSeed);
}