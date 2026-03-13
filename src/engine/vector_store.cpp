#include "vector_store.h"

#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <cmath>

VectorStore::VectorStore(size_t const dim) : dim_(dim) {};

void VectorStore::insert(uint64_t id, const std::vector<float>& vector)
{
   if (vector.size() != dim_)
   {
      throw std::runtime_error("Dimension Mismatch");
   }
   int updated_index = ids.size();
   ids.push_back(id);
   vectors.insert(vectors.end(), vector.begin(), vector.end());

   graph.push_back({});
   if(entry_point == -1){
      entry_point = 0; // Entry Point
      return;
   }

   using Pair = std::pair<float, int>;
   auto cmp = [](const Pair& a, const Pair& b) {
        return a.first < b.first;
    };
   std::priority_queue<Pair,std::vector<Pair>,decltype(cmp)> heap(cmp);

   for(int i = 0;i<updated_index;i++){
      float distance = distance_to_node(vector,i);
      if(heap.size() < max_neighbors){
         heap.emplace(distance,i);
      } else if (distance < heap.top().first) {
            heap.pop();
            heap.emplace(distance, i);
      }
   }

   while(!heap.empty()){
      int n = heap.top().second;
      heap.pop();

      graph[updated_index].push_back(n);
      graph[n].push_back(updated_index);
   }
}

float VectorStore::l2_distance(const float* a, const float* b) const
{
   float l2 = 0;
   for (size_t i = 0; i < dim_; ++i)
   {
      l2 += (a[i] - b[i]) * (a[i] - b[i]);
   }
   return sqrt(l2);
}

float VectorStore::distance_to_node(const std::vector<float>& query, int node_index) {
    return l2_distance(
        query.data(),
        &vectors[node_index * dim_]
    );
}

size_t VectorStore::size() const
{
   return VectorStore::ids.size();
}

std::vector<uint64_t> VectorStore::search_brute_force(const std::vector<float>& query, size_t k)
{
   if (query.size() != dim_) {
      throw std::runtime_error("Query dimension mismatch");
   }
   using Pair = std::pair<uint64_t, float>;
   auto cmp = [](const Pair& a, const Pair& b)
   {
      return a.second > b.second;
   };
   std::priority_queue<Pair, std::vector<Pair>, decltype(cmp)> max_heap(cmp);
   for (size_t i = 0; i < ids.size(); ++i)
   {
      const float* vctr_ptr = &vectors[i*dim_];
      float l2 = l2_distance(vctr_ptr,query.data());
      max_heap.emplace(ids[i], l2);
   }
   size_t i = 0;
   std::vector<uint64_t> result;
   while (i < k)
   {
      result.push_back(max_heap.top().first);
      max_heap.pop();
      i++;
   }
   return result;
}


int VectorStore::greedy_search(const std::vector<float>& query) {
    if (entry_point == -1) {
        return -1;
    }

    int current = entry_point;
    float current_dist = l2_distance(
        query.data(),
        &vectors[current * dim_]
    );

    bool improved = true;

    while (improved) {
        improved = false;

        for (int neighbor : graph[current]) {

            float neighbor_dist = l2_distance(
                query.data(),
                &vectors[neighbor * dim_]
            );

            if (neighbor_dist < current_dist) {
                current = neighbor;
                current_dist = neighbor_dist;
                improved = true;
            }
        }
    }

    return current;
}