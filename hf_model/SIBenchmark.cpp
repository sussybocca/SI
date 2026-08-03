#include "SIBenchmark.h"
#include "SITrainer.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <numeric>

namespace fs = std::filesystem;
using namespace std::chrono;

SIBenchmark::Result SIBenchmark::run(SIBrain& brain,
                                     const std::string& trainDir,
                                     const std::vector<std::string>& prompts,
                                     size_t genCount) {
    Result res;

    // ---- Training ----
    auto t1 = high_resolution_clock::now();
    // Train all .txt files
    SITrainer::trainFromDirectory(trainDir, brain);
    auto t2 = high_resolution_clock::now();
    res.trainTimeSec = duration<double>(t2 - t1).count();

    // Count total tokens trained is tricky; we approximate from node counts and input size.
    // We'll compute from the number of tokens processed by scanning the files again.
    size_t tokenCount = 0;
    for (auto& entry : fs::directory_iterator(trainDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::ifstream file(entry.path());
            std::string line;
            while (getline(file, line)) {
                auto tokens = SITrainer::tokenize(line);
                tokenCount += tokens.size();
            }
        }
    }
    res.totalTokensTrained = tokenCount;
    if (res.trainTimeSec > 0.0)
        res.tokensPerSecTrain = tokenCount / res.trainTimeSec;

    // ---- Generation ----
    size_t totalGenTokens = 0;
    double totalLatency = 0.0;
    for (const auto& prompt : prompts) {
        auto promptToks = SITrainer::tokenize(prompt);
        for (size_t i = 0; i < genCount; ++i) {
            auto start = high_resolution_clock::now();
            auto generated = brain.generate(promptToks);
            auto end = high_resolution_clock::now();
            totalLatency += duration<double, std::milli>(end - start).count();
            totalGenTokens += generated.size();
        }
    }
    res.queriesRun = prompts.size() * genCount;
    if (res.queriesRun > 0) {
        res.avgGenLatencyMs = totalLatency / res.queriesRun;
        double totalGenTimeSec = totalLatency / 1000.0;
        if (totalGenTimeSec > 0.0)
            res.tokensPerSecGen = totalGenTokens / totalGenTimeSec;
    }
    res.totalGenTokens = totalGenTokens;

    res.peakNodes = brain.totalNodes();
    res.peakMemoryMB = brain.memoryUsageEstimate() / (1024 * 1024);

    return res;
}