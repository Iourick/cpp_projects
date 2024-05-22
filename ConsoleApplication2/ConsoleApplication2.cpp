#include <iostream>
#include <chrono>
#include <vector>

// Function without restrict
void copy_array_no_restrict(int* dst, const int* src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
}

// Function with restrict
void copy_array_restrict(int* __restrict dst, const int* __restrict src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
}

int main() {
    const size_t n = 100000000; // Size of the arrays
    std::vector<int> src(n, 1); // Source array initialized to 1
    std::vector<int> dst(n, 0); // Destination array initialized to 0

    // Measure time for function without restrict
    auto start_no_restrict = std::chrono::high_resolution_clock::now();
    copy_array_no_restrict(dst.data(), src.data(), n);
    auto end_no_restrict = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_no_restrict = end_no_restrict - start_no_restrict;

    // Measure time for function with restrict
    auto start_restrict = std::chrono::high_resolution_clock::now();
    copy_array_restrict(dst.data(), src.data(), n);
    auto end_restrict = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_restrict = end_restrict - start_restrict;

    // Print the elapsed times
    std::cout << "Time without restrict: " << elapsed_no_restrict.count() << " seconds\n";
    std::cout << "Time with restrict: " << elapsed_restrict.count() << " seconds\n";

    return 0;
}
