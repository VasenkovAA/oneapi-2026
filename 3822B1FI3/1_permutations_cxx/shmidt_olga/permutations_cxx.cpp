#include "permutations_cxx.h"
#include <algorithm>
#include <unordered_map>

void Permutations(dictionary_t& dictionary) 
{
    std::unordered_map<std::string, std::vector<std::string>> groups;

    for (const auto& pair : dictionary) 
    {
        std::string sorted = pair.first;
        std::sort(sorted.begin(), sorted.end());
        groups[sorted].push_back(pair.first);
    }

    for (auto& group : groups) 
    {
        auto& words = group.second;
        std::sort(words.begin(), words.end(), std::greater<std::string>());

        for (size_t i = 0; i < words.size(); ++i) 
        {
            auto& vec = dictionary[words[i]];
            vec.reserve(words.size() - 1);
            for (size_t j = 0; j < words.size(); ++j) 
            {
                if (i != j) 
                {
                    vec.push_back(words[j]);
                }
            }
        }
    }
}