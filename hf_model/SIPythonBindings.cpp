#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "SIIntelligence.h"
#include <filesystem>

namespace py = pybind11;

// Wrapper class to handle Python-friendly interface
class SIPython {
private:
    std::unique_ptr<SIIntelligence> si_;
    std::string brainPath_;
    
public:
    SIPython(const std::string& brainPath = "si_brain") 
        : si_(std::make_unique<SIIntelligence>())
        , brainPath_(brainPath) {
        // Try to load existing brain
        std::ifstream test(brainPath + ".brain");
        if (test.good()) {
            test.close();
            si_->load(brainPath);
        }
    }
    
    // Train on a single text string
    void train(const std::string& text) {
        si_->learn(text);
    }
    
    // Train on a list of texts
    void trainBatch(const std::vector<std::string>& texts) {
        for (const auto& text : texts) {
            si_->learn(text);
        }
    }
    
    // Train from a directory of .txt files
    void trainFromDirectory(const std::string& directoryPath) {
        if (!std::filesystem::exists(directoryPath) || !std::filesystem::is_directory(directoryPath)) {
            throw std::runtime_error("Invalid directory: " + directoryPath);
        }
        
        size_t fileCount = 0;
        for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
            if (entry.path().extension() == ".txt") {
                std::ifstream file(entry.path());
                std::string line;
                while (std::getline(file, line)) {
                    if (!line.empty()) {
                        si_->learn(line);
                    }
                }
                fileCount++;
            }
        }
        std::cout << "Trained on " << fileCount << " files" << std::endl;
    }
    
    // Ask a question / think about input
    std::string ask(const std::string& question) {
        return si_->think(question);
    }
    
    // Teach a specific concept
    void teach(const std::string& word, const std::string& meaning) {
        si_->answerQuestion(word, meaning);
    }
    
    // Generate text from a prompt (uses brain directly)
    std::string generate(const std::string& prompt) {
        return si_->think(prompt);
    }
    
    // Chat interaction - returns response for a message
    std::string chat(const std::string& message) {
        return si_->think(message);
    }
    
    // Save the brain
    void save(const std::string& path = "") {
        std::string savePath = path.empty() ? brainPath_ : path;
        si_->save(savePath);
        std::cout << "Brain saved to " << savePath << std::endl;
    }
    
    // Load a brain
    void load(const std::string& path) {
        brainPath_ = path;
        si_->load(path);
        std::cout << "Brain loaded from " << path << std::endl;
    }
    
    // Get statistics
    py::dict getStats() {
        py::dict stats;
        stats["vocabulary_size"] = si_->vocabularySize();
        stats["knowledge_base_size"] = si_->knowledgeBaseSize();
        stats["web_search_count"] = si_->getWebSearchCount();
        return stats;
    }
    
    // Enable/disable web search
    void enableWebSearch(bool enable) {
        si_->enableWeb(enable);
    }
    
    // Get suggested topics based on learned preferences
    std::vector<std::string> getSuggestedTopics() {
        return si_->getSuggestedTopics();
    }
    
    // Set curiosity level (0.0 to 1.0)
    void setCuriosity(double level) {
        si_->setCuriosity(level);
    }
    
    // Batch question answering (useful for evaluation)
    std::vector<std::string> askBatch(const std::vector<std::string>& questions) {
        std::vector<std::string> answers;
        for (const auto& q : questions) {
            answers.push_back(si_->think(q));
        }
        return answers;
    }
};

// HuggingFace-compatible model interface
class HuggingFaceModel {
private:
    SIPython model_;
    
public:
    HuggingFaceModel(const std::string& modelPath = "si_brain") 
        : model_(modelPath) {}
    
    // HuggingFace pipeline interface
    py::dict __call__(const std::string& inputs, py::kwargs kwargs) {
        std::string response = model_.ask(inputs);
        
        py::dict result;
        result["generated_text"] = response;
        result["input"] = inputs;
        return result;
    }
    
    // Train method compatible with HuggingFace datasets
    void train(const std::vector<std::string>& dataset) {
        model_.trainBatch(dataset);
    }
    
    void save_pretrained(const std::string& path) {
        model_.save(path);
    }
    
    static HuggingFaceModel from_pretrained(const std::string& path) {
        return HuggingFaceModel(path);
    }
};

// Python module definition
PYBIND11_MODULE(si_engine, m) {
    m.doc() = "SI (Synthetic Intelligence) Engine - A new form of intelligence that learns instantly";
    
    // Main SI Python class
    py::class_<SIPython>(m, "SyntheticIntelligence")
        .def(py::init<const std::string&>(), 
             py::arg("brain_path") = "si_brain",
             "Initialize SI with optional brain path")
        .def("train", &SIPython::train,
             py::arg("text"),
             "Train SI on a single text string")
        .def("train_batch", &SIPython::trainBatch,
             py::arg("texts"),
             "Train SI on a batch of text strings")
        .def("train_from_directory", &SIPython::trainFromDirectory,
             py::arg("directory_path"),
             "Train SI on all .txt files in a directory")
        .def("ask", &SIPython::ask,
             py::arg("question"),
             "Ask SI a question - it will search the web if needed")
        .def("teach", &SIPython::teach,
             py::arg("word"), py::arg("meaning"),
             "Teach SI about a specific concept")
        .def("generate", &SIPython::generate,
             py::arg("prompt"),
             "Generate text from a prompt")
        .def("chat", &SIPython::chat,
             py::arg("message"),
             "Send a chat message and get a response")
        .def("save", &SIPython::save,
             py::arg("path") = "",
             "Save the SI brain to disk")
        .def("load", &SIPython::load,
             py::arg("path"),
             "Load an SI brain from disk")
        .def("get_stats", &SIPython::getStats,
             "Get brain statistics")
        .def("enable_web_search", &SIPython::enableWebSearch,
             py::arg("enable"),
             "Enable or disable web search")
        .def("get_suggested_topics", &SIPython::getSuggestedTopics,
             "Get topics SI suggests based on your interactions")
        .def("set_curiosity", &SIPython::setCuriosity,
             py::arg("level"),
             "Set curiosity level (0.0 to 1.0)")
        .def("ask_batch", &SIPython::askBatch,
             py::arg("questions"),
             "Ask multiple questions at once");
    
    // HuggingFace-compatible model class
    py::class_<HuggingFaceModel>(m, "HuggingFaceModel")
        .def(py::init<const std::string&>(),
             py::arg("model_path") = "si_brain",
             "Initialize HuggingFace-compatible SI model")
        .def("__call__", &HuggingFaceModel::__call__,
             py::arg("inputs"),
             "Run inference (HuggingFace pipeline interface)")
        .def("train", &HuggingFaceModel::train,
             py::arg("dataset"),
             "Train on a dataset")
        .def("save_pretrained", &HuggingFaceModel::save_pretrained,
             py::arg("path"),
             "Save the pretrained model")
        .def_static("from_pretrained", &HuggingFaceModel::from_pretrained,
             py::arg("path"),
             "Load a pretrained model");
    
    // Module constants
    m.attr("__version__") = "2.0.0";
    m.attr("__author__") = "SI Engine Team";
}