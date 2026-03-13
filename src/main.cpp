#include "engine/vector_store.h"
#include <iostream>
#include <random>
#include <chrono>

int main() {
    const size_t dim = 32;
    const size_t num_vectors = 10000;


    VectorStore store(dim);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  
    for (size_t i = 0; i < num_vectors; ++i) {
        std::vector<float> vec(dim);
        for (size_t j = 0; j < dim; ++j) {
            vec[j] = dist(gen);
        }
        store.insert(i, vec);
    }

    std::vector<float> query(dim);
    for (size_t j = 0; j < dim; ++j) {
        query[j] = dist(gen);
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto results = store.search_brute_force(query, 5);
    auto end = std::chrono::high_resolution_clock::now();

    

    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Search took: " << duration.count() << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    int node = store.greedy_search(query);
    end = std::chrono::high_resolution_clock::now();

    duration = end -start;

    std::cout << "Greedy search result node index: " << node << std::endl;

    std::cout << "Search took: " << duration.count() << " ms\n";
    std::cout << "Top IDs:\n";
    for (auto id : results) {
        std::cout << id << "\n";
    }
    std::cout << "Query \n";
    for (auto id : query){
        std::cout << id << "\n";
    }

    return 0;
}