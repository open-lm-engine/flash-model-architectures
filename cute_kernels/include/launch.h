#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

template <typename T>
struct ChunkedArray {
    T* array;
    size_t num_elements;
};

template <typename T>
std::vector<ChunkedArray<T> > chunk_array(T* x, const size_t& num_elements) {
    const size_t num_elements_per_chunk = std::numeric_limits<uint>::max();
    size_t start = 0;

    std::vector<ChunkedArray<T> > result;

    // We will "yield" chunks of the array, each of size 'chunk_size' or less
    while (start < num_elements) {
        size_t remaining = num_elements - start;
        size_t current_chunk_size = std::min(remaining, num_elements_per_chunk);

        ChunkedArray<T> chunked_x;
        chunked_x.array = (T*)&x[start];
        chunked_x.num_elements = current_chunk_size;

        result.push_back(chunked_x);
        start += current_chunk_size;
    }

    return result;
}
