#ifndef SI_BENCHMARK_H
#define SI_BENCHMARK_H

#include "SIBrain.h"
#include <string>
#include <chrono>
#include <vector>
#include <functional>

/**
 * @brief Benchmarking utility for SI training and generation.
 */
class SIBenchmark {
public:
    struct Result {
        double trainTimeSec       = 0.0;
        size_t totalTokensTrained = 0;
        double tokensPerSecTrain  = 0.0;

        size_t queriesRun         = 0;
        double avgGenLatencyMs    = 0.0;
        double tokensPerSecGen    = 0.0;
        size_t totalGenTokens     = 0;

        size_t peakNodes          = 0;
        size_t peakMemoryMB       = 0;
    };

    /**
     * @brief Run training benchmark: trains on all .txt files in a directory,
     *        then evaluates generation speed with sample prompts.
     * @param brain       brain instance (will be modified).
     * @param trainDir    directory containing .txt files.
     * @param prompts     prompts used for generation test.
     * @param genCount    number of generation calls per prompt for averaging.
     * @return Result struct with all metrics.
     */
    static Result run(SIBrain& brain,
                      const std::string& trainDir,
                      const std::vector<std::string>& prompts = {"the", "I"},
                      size_t genCount = 10);
};

#endif // SI_BENCHMARK_H