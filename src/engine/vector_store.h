#pragma once
#include <vector>
#include <cstdint>

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
    private:
        size_t dim_;
        std::vector<uint64_t> ids;
        std::vector<float> vectors;
        float l2_distance(const float* a, const float* b) const;
};