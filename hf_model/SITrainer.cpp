#include "SITrainer.h"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

std::vector<std::string> SITrainer::tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::string current;
    bool insideWord = false;

    auto flushCurrent = [&]() {
        if (!current.empty()) {
            tokens.push_back(current);
            current.clear();
        }
        insideWord = false;
    };

    for (size_t i = 0; i < line.size(); ++i) {
        char ch = line[i];
        unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isalpha(uch) || std::isdigit(uch) || ch == '\'' || ch == '-') {
            // Part of a word/number/contraction/hyphenation
            current += ch;
            insideWord = true;
        } else {
            flushCurrent();
            if (!std::isspace(uch)) {
                // Punctuation – emit as separate token
                tokens.emplace_back(1, ch);
            }
        }
    }
    flushCurrent();
    return tokens;
}

void SITrainer::trainFromDirectory(const std::string& dirPath,
                                   SIBrain& brain,
                                   std::function<void(size_t, size_t)> progressCallback,
                                   bool recursive) {
    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        throw std::runtime_error("Invalid directory: " + dirPath);
    }

    std::vector<fs::path> txtFiles;
    if (recursive) {
        for (const auto& entry : fs::recursive_directory_iterator(dirPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                txtFiles.push_back(entry.path());
            }
        }
    } else {
        for (const auto& entry : fs::directory_iterator(dirPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                txtFiles.push_back(entry.path());
            }
        }
    }

    if (txtFiles.empty()) {
        std::cout << "No .txt files found in " << dirPath << std::endl;
        return;
    }

    size_t total = txtFiles.size();
    for (size_t i = 0; i < total; ++i) {
        std::ifstream file(txtFiles[i], std::ios::in);
        if (!file) {
            std::cerr << "Warning: could not open " << txtFiles[i] << std::endl;
            continue;
        }
        // Process line by line; avoid excessive memory usage for huge files.
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            auto tokens = tokenize(line);
            if (!tokens.empty()) {
                brain.learnSequence(tokens);
            }
        }
        if (progressCallback) {
            progressCallback(i + 1, total);
        }
    }
}