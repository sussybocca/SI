#ifndef SI_TRAINER_H
#define SI_TRAINER_H

#include "SIBrain.h"
#include <functional>
#include <string>
#include <vector>

/**
 * @brief Handles ingestion of .txt files and robust tokenisation.
 */
class SITrainer {
public:
    /**
     * @brief Tokeniser that splits on whitespace, separates punctuation,
     *        and treats numbers, hyphenated words, and contractions as single tokens.
     * @param line raw text line.
     * @return vector of tokens.
     */
    static std::vector<std::string> tokenize(const std::string& line);

    /**
     * @brief Train the brain on all .txt files in a directory.
     *        Supports recursive directory traversal (optional).
     * @param dirPath          path to directory containing .txt files.
     * @param brain            the SI brain to train.
     * @param progressCallback optional callback (files processed, total files).
     * @param recursive        if true, subdirectories are searched.
     * @throws std::runtime_error on I/O failure or invalid path.
     */
    static void trainFromDirectory(const std::string& dirPath,
                                   SIBrain& brain,
                                   std::function<void(size_t, size_t)> progressCallback = nullptr,
                                   bool recursive = false);
};

#endif // SI_TRAINER_H