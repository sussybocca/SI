#ifndef SI_BRAIN_PRUNETOOL_H
#define SI_BRAIN_PRUNETOOL_H

#include <string>
#include <cstdint>
#include <functional>

/**
 * @brief Advanced pruning policies for a saved SI brain file.
 *        No need to modify SIBrain – operates directly on the binary.
 */
enum class PrunePolicy : uint8_t {
    LFU,        // remove subtrees with lowest total usage count
    MinCount,   // remove all nodes below an absolute count threshold
    Adaptive    // iterative LFU until target node count, keeping high-count paths
};

/**
 * @brief Configuration for a pruning run.
 */
struct PruneConfig {
    PrunePolicy policy          = PrunePolicy::Adaptive;
    size_t targetMaxNodes       = 1'000'000;   // desired node count (0 = no limit)
    uint64_t minCountThreshold  = 5;           // only for MinCount policy
    double keepRatio            = 0.8;         // for Adaptive: fraction of nodes to keep (if target=0)
    bool verbose                = false;
    std::function<void(size_t, size_t)> progressCallback; // current nodes, target (if set)
};

/**
 * @brief Loads a brain file, applies a pruning policy, and writes the result.
 * @param inputFile   path to existing brain file (binary).
 * @param outputFile  path to pruned brain file (overwritten).
 * @param config      pruning configuration.
 * @return true on success.
 * @throws std::runtime_error on I/O / format failure.
 */
bool pruneBrainFile(const std::string& inputFile,
                    const std::string& outputFile,
                    const PruneConfig& config);

#endif // SI_BRAIN_PRUNETOOL_H