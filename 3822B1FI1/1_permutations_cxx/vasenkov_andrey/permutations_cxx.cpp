
#include "permutations_cxx.h"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include <map>

void Permutations(dictionary_t& dictionary) {
    std::map<std::string, std::vector<std::string>> groups;

    for (const auto& entry : dictionary) {
        std::string sorted = entry.first;
        std::sort(sorted.begin(), sorted.end());
        groups[sorted].push_back(entry.first);
    }

    for (auto& entry : dictionary) {
        const std::string& key = entry.first;
        std::string sorted = key;
        std::sort(sorted.begin(), sorted.end());

        const auto& group = groups[sorted];
        auto& perm_vec = entry.second;

        perm_vec.clear();
        perm_vec.reserve(group.size());

        for (const std::string& candidate : group) {
            if (candidate != key) {
                perm_vec.push_back(candidate);
            }
        }

        std::sort(perm_vec.begin(), perm_vec.end(), std::greater<std::string>());
    }
}
