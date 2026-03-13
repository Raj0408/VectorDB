#pragma once
#include <vector>
#include <cstdint>
#include <unordered_map>

class VectorStore
{
    public:
        VectorStore(size_t dim);
        void insert(uint64_t id,const std::vector<float>& vector);
        std::vector<uint64_t> search_brute_force(
            const std::vector<float>& query,
            size_t k
        );
        size_t size() const;
        float distance_to_node(const std::vector<float>& query, int node_index);
        int greedy_search(const std::vector<float>& query);

    private:
        size_t dim_;
        std::vector<uint64_t> ids;
        std::vector<float> vectors;
        std::vector<std::vector<int>> graph;
        int entry_point = -1;

        float l2_distance(const float* a, const float* b) const;
        size_t max_neighbors = 8;
};