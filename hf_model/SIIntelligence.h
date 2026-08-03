#ifndef SI_INTELLIGENCE_H
#define SI_INTELLIGENCE_H

#include "SIBrain.h"
#include "SIConcept.h"
#include "SIReasoning.h"
#include "SICuriosity.h"
#include "SIAttention.h"
#include "SIAlgorithm.h"
#include "SITrainer.h"

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <regex>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ctime>
#include <cmath>
#include <numeric>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>

#include <curl/curl.h>

class CurlHandle {
public:
    CurlHandle() : handle_(curl_easy_init()) {
        if (!handle_) throw std::runtime_error("curl_easy_init failed");
    }
    ~CurlHandle() { if (handle_) curl_easy_cleanup(handle_); }
    CURL* get() const { return handle_; }
    CurlHandle(const CurlHandle&) = delete;
    CurlHandle& operator=(const CurlHandle&) = delete;
private:
    CURL* handle_;
};

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total = size * nmemb;
    std::string* str = static_cast<std::string*>(userp);
    if (str->size() + total > 2 * 1024 * 1024) return 0;
    str->append(static_cast<char*>(contents), total);
    return total;
}

struct SearchResult {
    std::string title;
    std::string snippet;
    std::string url;
};

class SearchProvider {
public:
    virtual ~SearchProvider() = default;
    virtual std::vector<SearchResult> search(const std::string& query) = 0;
    virtual std::string providerName() const = 0;
};

class BraveSearchProvider : public SearchProvider {
public:
    BraveSearchProvider() {
        const char* key = std::getenv("BRAVE_API_KEY");
        apiKey_ = key ? key : "";
    }
    std::string providerName() const override { return "Brave"; }
    std::vector<SearchResult> search(const std::string& query) override {
        if (apiKey_.empty()) return {};
        std::vector<SearchResult> results;
        std::string response;
        if (!curlGet("https://api.search.brave.com/res/v1/web/search?q=" + urlEncode(query), response,
                     {"Accept: application/json", "X-Subscription-Token: " + apiKey_, "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/138.0.0.0 Safari/537.36"}))
            return {};
        std::regex result_re("\"title\":\"([^\"]*)\".*?\"url\":\"([^\"]*)\".*?\"description\":\"([^\"]*)\"");
        auto it = std::sregex_iterator(response.begin(), response.end(), result_re);
        for (; it != std::sregex_iterator() && results.size() < 10; ++it) {
            results.push_back({(*it)[1], (*it)[3], (*it)[2]});
        }
        return results;
    }
private:
    std::string apiKey_;
    bool curlGet(const std::string& url, std::string& out, const std::vector<std::string>& headers) {
        CurlHandle curl;
        curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &out);
        curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 15L);
        curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
        struct curl_slist* hlist = nullptr;
        for (const auto& h : headers) hlist = curl_slist_append(hlist, h.c_str());
        curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, hlist);
        CURLcode res = curl_easy_perform(curl.get());
        curl_slist_free_all(hlist);
        return res == CURLE_OK;
    }
    std::string urlEncode(const std::string& s) {
        CurlHandle curl;
        char* escaped = curl_easy_escape(curl.get(), s.c_str(), (int)s.size());
        std::string result(escaped ? escaped : "");
        curl_free(escaped);
        return result;
    }
};

class BingSearchProvider : public SearchProvider {
public:
    BingSearchProvider() {
        const char* key = std::getenv("BING_API_KEY");
        apiKey_ = key ? key : "";
    }
    std::string providerName() const override { return "Bing"; }
    std::vector<SearchResult> search(const std::string& query) override {
        if (apiKey_.empty()) return {};
        std::vector<SearchResult> results;
        std::string response;
        if (!curlGet("https://api.bing.microsoft.com/v7.0/search?q=" + urlEncode(query), response,
                     {"Ocp-Apim-Subscription-Key: " + apiKey_, "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/138.0.0.0 Safari/537.36"}))
            return {};
        std::regex result_re("\"name\":\"([^\"]*)\".*?\"url\":\"([^\"]*)\".*?\"snippet\":\"([^\"]*)\"");
        auto it = std::sregex_iterator(response.begin(), response.end(), result_re);
        for (; it != std::sregex_iterator() && results.size() < 10; ++it) {
            results.push_back({(*it)[1], (*it)[3], (*it)[2]});
        }
        return results;
    }
private:
    std::string apiKey_;
    bool curlGet(const std::string& url, std::string& out, const std::vector<std::string>& headers) {
        CurlHandle curl;
        curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &out);
        curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 15L);
        curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
        struct curl_slist* hlist = nullptr;
        for (const auto& h : headers) hlist = curl_slist_append(hlist, h.c_str());
        curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, hlist);
        CURLcode res = curl_easy_perform(curl.get());
        curl_slist_free_all(hlist);
        return res == CURLE_OK;
    }
    std::string urlEncode(const std::string& s) {
        CurlHandle curl;
        char* escaped = curl_easy_escape(curl.get(), s.c_str(), (int)s.size());
        std::string result(escaped ? escaped : "");
        curl_free(escaped);
        return result;
    }
};

class WikipediaSearchProvider : public SearchProvider {
public:
    std::string providerName() const override { return "Wikipedia"; }
    std::vector<SearchResult> search(const std::string& query) override {
        std::vector<SearchResult> results;
        std::string searchJson;
        std::string searchPath = "/w/api.php?action=query&list=search&srsearch=" + urlEncode(query) + "&format=json&srlimit=5";
        
        std::cout << "[WIKI] Searching..." << std::endl;
        if (!curlGet("en.wikipedia.org", searchPath, searchJson)) {
            std::cout << "[WIKI] Search failed" << std::endl;
            return results;
        }
        
        std::regex title_re("\"title\":\"([^\"]+)\"");
        std::regex snippet_re("\"snippet\":\"([^\"]*)\"");
        auto title_it = std::sregex_iterator(searchJson.begin(), searchJson.end(), title_re);
        auto snippet_it = std::sregex_iterator(searchJson.begin(), searchJson.end(), snippet_re);
        
        for (; title_it != std::sregex_iterator() && snippet_it != std::sregex_iterator() && results.size() < 5; ++title_it, ++snippet_it) {
            SearchResult r;
            r.title = (*title_it)[1].str();
            r.snippet = std::regex_replace((*snippet_it)[1].str(), std::regex("<[^>]*>"), "");
            std::string pageTitle = r.title;
            std::replace(pageTitle.begin(), pageTitle.end(), ' ', '_');
            r.url = "https://en.wikipedia.org/wiki/" + pageTitle;
            results.push_back(r);
        }
        
        if (!results.empty()) {
            std::string extractJson;
            std::string extractPath = "/w/api.php?action=query&prop=extracts&exlimit=1&explaintext=1&exsectionformat=plain&titles=" 
                                      + urlEncode(results[0].title) + "&format=json";
            if (curlGet("en.wikipedia.org", extractPath, extractJson)) {
                std::regex extract_re("\"extract\":\"([^\"]*)\"");
                std::smatch m;
                if (std::regex_search(extractJson, m, extract_re)) {
                    std::string extract = m[1].str();
                    std::string clean;
                    bool esc = false;
                    for (char c : extract) {
                        if (esc) { clean += (c == 'n') ? ' ' : (c == 't' ? ' ' : c); esc = false; }
                        else if (c == '\\') esc = true;
                        else clean += c;
                    }
                    results[0].snippet = clean;
                    std::cout << "[WIKI] Extract: " << clean.size() << " chars" << std::endl;
                }
            }
        }
        std::cout << "[WIKI] Found " << results.size() << " results" << std::endl;
        return results;
    }
private:
    bool curlGet(const std::string& host, const std::string& path, std::string& out) {
        CurlHandle curl;
        std::string fullUrl = "https://" + host + path;
        curl_easy_setopt(curl.get(), CURLOPT_URL, fullUrl.c_str());
        curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &out);
        curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 15L);
        curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "User-Agent: SIIntelligence/3.0");
        curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers);
        CURLcode res = curl_easy_perform(curl.get());
        curl_slist_free_all(headers);
        return res == CURLE_OK;
    }
    std::string urlEncode(const std::string& s) {
        CurlHandle curl;
        char* escaped = curl_easy_escape(curl.get(), s.c_str(), (int)s.size());
        std::string result(escaped ? escaped : "");
        curl_free(escaped);
        return result;
    }
};

class SIIntelligence {
private:
    std::unique_ptr<SIBrain> brain_;
    std::unique_ptr<SIConcept> concepts_;
    std::unique_ptr<SIReasoning> reasoning_;
    std::unique_ptr<SICuriosity> curiosity_;
    std::unique_ptr<SIAttention> attention_;
    std::unique_ptr<SIAlgorithm> algorithm_;

    std::vector<std::string> conversationHistory_;
    std::vector<std::string> knowledgeBase_;
    bool webEnabled_;
    int webSearchCount_;
    mutable std::mutex webMutex_;
    std::unordered_map<std::string, std::string> searchCache_;
    std::unordered_map<std::string, std::string> learnedAnswers_;
    std::vector<std::unique_ptr<SearchProvider>> providers_;

    std::string summarizeText(const std::string& text) {
        if (text.size() <= 1000) return text;
        std::istringstream iss(text);
        std::vector<std::string> words;
        std::string w;
        while (iss >> w) words.push_back(w);
        if (words.size() <= 500) return text;
        std::unordered_map<std::string, int> freq;
        for (const auto& wd : words) freq[wd]++;
        auto scoreSentence = [&](const std::string& s) -> double {
            std::istringstream sis(s);
            std::string sw;
            double sc = 0; int c = 0;
            while (sis >> sw) { sc += freq[sw]; c++; }
            return c ? sc / c : 0;
        };
        std::vector<std::string> sentences;
        std::regex sent_re("[^.!?]+[.!?]");
        auto it = std::sregex_iterator(text.begin(), text.end(), sent_re);
        for (; it != std::sregex_iterator(); ++it) sentences.push_back(it->str());
        std::vector<std::pair<double, std::string>> ranked;
        for (const auto& s : sentences) ranked.emplace_back(scoreSentence(s), s);
        std::sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) { return a.first > b.first; });
        std::string summary;
        int wc = 0;
        for (const auto& [sc, s] : ranked) {
            if (wc >= 500) break;
            summary += s + " ";
            wc += (int)std::count(s.begin(), s.end(), ' ') + 1;
        }
        return summary.empty() ? text.substr(0, 1000) : summary;
    }

    std::string searchAndLearn(const std::string& query) {
        std::cout << "[WEB] Searching for: " << query << std::endl;
        {
            std::lock_guard<std::mutex> lock(webMutex_);
            if (!webEnabled_) return "";
            if (webSearchCount_ > 50) return "";
            auto it = searchCache_.find(query);
            if (it != searchCache_.end()) {
                std::cout << "[WEB] Cache hit!" << std::endl;
                return it->second;
            }
        }

        std::string combinedText;
        for (auto& provider : providers_) {
            std::cout << "[WEB] Trying: " << provider->providerName() << std::endl;
            auto results = provider->search(query);
            std::cout << "[WEB] Got " << results.size() << " results" << std::endl;
            if (results.empty()) continue;

            // Use Wikipedia extracts directly (they're already plain text)
            for (auto& res : results) {
                if (res.snippet.size() > 200) {
                    combinedText += res.snippet + "\n\n";
                    std::cout << "[WEB] Using extract: " << res.snippet.size() << " chars" << std::endl;
                }
            }
            if (!combinedText.empty()) break;
        }

        if (combinedText.empty()) {
            std::cout << "[WEB] No results." << std::endl;
            return "";
        }

        std::string summary = summarizeText(combinedText);
        learn(summary);

        // STORE IN LEARNED ANSWERS
        std::string key = query;
        std::transform(key.begin(), key.end(), key.begin(), ::tolower);
        learnedAnswers_[key] = summary;

        {
            std::lock_guard<std::mutex> lock(webMutex_);
            searchCache_[query] = summary;
            webSearchCount_++;
            std::cout << "[WEB] Learned! Total: " << webSearchCount_ << std::endl;
        }
        return summary;
    }

public:
    SIIntelligence() 
        : brain_(std::make_unique<SIBrain>())
        , concepts_(std::make_unique<SIConcept>())
        , reasoning_(std::make_unique<SIReasoning>(*concepts_))
        , curiosity_(std::make_unique<SICuriosity>())
        , attention_(std::make_unique<SIAttention>())
        , algorithm_(std::make_unique<SIAlgorithm>(brain_.get(), concepts_.get()))
        , webEnabled_(true)
        , webSearchCount_(0) {
        providers_.push_back(std::make_unique<WikipediaSearchProvider>());
        providers_.push_back(std::make_unique<BraveSearchProvider>());
        providers_.push_back(std::make_unique<BingSearchProvider>());
    }

    void enableWeb(bool enable) { webEnabled_ = enable; }

    void learn(const std::string& text) {
        auto tokens = SITrainer::tokenize(text);
        if (tokens.empty()) return;
        knowledgeBase_.push_back(text);
        if (knowledgeBase_.size() > 50000) knowledgeBase_.erase(knowledgeBase_.begin());
        brain_->learnSequence(tokens);
        concepts_->learnFromContext(tokens);
        algorithm_->index(text);
        for (const auto& token : tokens) attention_->addToContext(token);
    }

    std::string think(const std::string& input) {
        auto tokens = SITrainer::tokenize(input);
        if (tokens.empty()) return "Please say something meaningful.";

        conversationHistory_.push_back(input);
        learn(input);

        // 1. Internal knowledge
        std::string algoResponse = algorithm_->getBestResponse(input);
        if (!algoResponse.empty() && algoResponse.find("don't know") == std::string::npos && algoResponse.size() > 20) {
            return algoResponse;
        }

        // 2. CHECK LEARNED ANSWERS (instant recall from web)
        std::string lookup = input;
        std::transform(lookup.begin(), lookup.end(), lookup.begin(), ::tolower);
        auto memory = learnedAnswers_.find(lookup);
        if (memory != learnedAnswers_.end()) {
            return "I know this: " + memory->second;
        }

        // 3. WEB SEARCH
        std::string webSummary = searchAndLearn(input);
        if (!webSummary.empty()) {
            return "I found: " + webSummary;
        }

        // 4. Brain generation as last resort
        for (const auto& token : tokens) {
            if (token.size() < 2) continue;
            auto generated = brain_->generate({token});
            if (generated.size() > 3) {
                std::string response;
                for (const auto& t : generated) response += t + " ";
                if (!response.empty()) response.pop_back();
                return response;
            }
        }

        return "I don't know about that yet. Teach me with ::teach <word> <meaning>";
    }

    void answerQuestion(const std::string& word, const std::string& explanation) {
        curiosity_->learnAnswer(word, explanation);
        learn(explanation);
    }

    void save(const std::string& path) {
        brain_->save(path + ".brain");
        algorithm_->save(path + ".algo");
    }

    void load(const std::string& path) {
        brain_->load(path + ".brain");
        algorithm_->load(path + ".algo");
    }

    void setCuriosity(double level) { curiosity_->setCuriosity(level); }
    size_t vocabularySize() const { return brain_->totalNodes(); }
    size_t knowledgeBaseSize() const { return knowledgeBase_.size(); }
    int getWebSearchCount() const {
        std::lock_guard<std::mutex> lock(webMutex_);
        return webSearchCount_;
    }
    std::vector<std::string> getSuggestedTopics() { return algorithm_->getSuggestedTopics(); }
};

#endif