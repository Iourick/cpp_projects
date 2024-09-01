#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <iostream>

// Define the SCand struct
struct SCand {
    int mt;
    int mdt;
    int mwidth;
    float msnr;
};

using Cand = SCand;

// Define the comparator struct
struct CompareCandMember {
    const Cand* d_vctCandHeap;
    int member_offset;

    CompareCandMember(const Cand* _d_vctCandHeap, int _member_offset)
        : d_vctCandHeap(_d_vctCandHeap), member_offset(_member_offset) {}

    __host__ __device__
        bool operator()(const int& idx1, const int& idx2) const {
        const char* base1 = reinterpret_cast<const char*>(&d_vctCandHeap[idx1]);
        const char* base2 = reinterpret_cast<const char*>(&d_vctCandHeap[idx2]);
        int value1 = *reinterpret_cast<const int*>(base1 + member_offset);
        int value2 = *reinterpret_cast<const int*>(base2 + member_offset);

        return value1 < value2;
    }
};

int main() {
    // Initialize host vector with example data
    thrust::host_vector<Cand> h_vctCandHeap = {
        {10, 2, 3, 4.0f}, {30, 5, 6, 7.0f}, {20, 8, 9, 10.0f}, {50, 11, 12, 13.0f}, {40, 14, 15, 16.0f},
        {15, 3, 4, 5.0f}, {35, 6, 7, 8.0f}, {25, 9, 10, 11.0f}, {55, 12, 13, 14.0f}, {45, 15, 16, 17.0f}
    };

    // Copy to device vector
    thrust::device_vector<Cand> d_vctCandHeap = h_vctCandHeap;

    // Initialize d_vctNumDelegates with specific indices
    thrust::device_vector<int> d_vctNumDelegates = { 8, 6,5 };

    // Define the offset for the member 'mt'
    int member_offset = offsetof(Cand, mt);

    // Sort d_vctNumDelegates based on 'mt' member of d_vctCandHeap
    thrust::sort(d_vctNumDelegates.begin(), d_vctNumDelegates.end(),
        CompareCandMember(thrust::raw_pointer_cast(d_vctCandHeap.data()), member_offset));

    // Print the sorted indices
    thrust::host_vector<int> h_vctNumDelegates = d_vctNumDelegates;
    std::cout << "Sorted d_vctNumDelegates indices:" << std::endl;
    for (size_t i = 0; i < h_vctNumDelegates.size(); ++i) {
        std::cout << h_vctNumDelegates[i] << std::endl;
    }

    // Verify the sorting
    thrust::host_vector<Cand> h_vctCandHeap_sorted = d_vctCandHeap;
    std::cout << "Sorted Cand elements by mt:" << std::endl;
    for (size_t i = 0; i < h_vctNumDelegates.size(); ++i) {
        int idx = h_vctNumDelegates[i];
        std::cout << "mt: " << h_vctCandHeap_sorted[idx].mt << std::endl;
    }

    return 0;
}
