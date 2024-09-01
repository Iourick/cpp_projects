#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <cuda_runtime.h>

// Define the Cand structure
typedef struct SCand {
    int mt;
    int mdt;
    int mwidth;
    int msnr;
} Cand;

// Example kernel to fill d_arrCandHeap with values
__global__ void gather_candidates_in_heap_kernel(int* d_digitized, int cols, int valTresh, int WndWidth, int* d_plan, Cand* d_arr, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        d_arr[idx].mt = idx;  // Just for example, you should fill it with real data
        d_arr[idx].mdt = num_elements - idx;  // Use num_elements - idx for diversity
        d_arr[idx].mwidth = idx + 2;
        d_arr[idx].msnr = idx + 3;
    }
}

// Custom comparator to sort indices based on the mt field
struct CompareIndicesByMt {
    Cand* d_arr;

    CompareIndicesByMt(Cand* d_arr) : d_arr(d_arr) {}

    __host__ __device__
        bool operator()(int lhs, int rhs) const {
        return d_arr[lhs].mt < d_arr[rhs].mt;
    }
};

int main() {
    int rows = 10;  // Example row size
    int cols = 20;  // Example column size
    int num_elements = rows * cols + 1;

    // Initialize the device vector and fill it with some positive values
    thrust::device_vector<int> d_plan(num_elements);
    thrust::sequence(d_plan.begin(), d_plan.end(), 1);

    // Retrieve the value of the last element of d_plan
    int last_element = d_plan.back();

    // Allocate device memory for array of Cand structures based on the last element of d_plan
    thrust::device_vector<Cand> d_arrCandHeap(last_element);

    // Example values for the kernel parameters
    thrust::device_vector<int> d_digitized(num_elements);
    int valTresh = 10;
    int WndWidth = 5;

    // Launch a kernel to fill d_arrCandHeap with some values
    int blockSize = 256;
    int numBlocks = (last_element + blockSize - 1) / blockSize;
    gather_candidates_in_heap_kernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(d_digitized.data()), cols, valTresh, WndWidth, thrust::raw_pointer_cast(d_plan.data()), thrust::raw_pointer_cast(d_arrCandHeap.data()), last_element);
    cudaDeviceSynchronize();

    // Create a thrust::device_vector of indices
    thrust::device_vector<int> d_indices(last_element);
    thrust::sequence(d_indices.begin(), d_indices.end());

    // Sort the indices based on the mt field of d_arrCandHeap
    thrust::sort(d_indices.begin(), d_indices.end(), CompareIndicesByMt(thrust::raw_pointer_cast(d_arrCandHeap.data())));

    // Create a new device vector to hold the sorted Cand structures
    thrust::device_vector<Cand> d_sortedCandHeap(last_element);

    // Rearrange the Cand structures based on the sorted indices
    thrust::gather(d_indices.begin(), d_indices.end(), d_arrCandHeap.begin(), d_sortedCandHeap.begin());

    // Copy the sorted device vector to host and print for verification
    thrust::host_vector<Cand> h_sortedCandHeap = d_sortedCandHeap;
    std::cout << "Sorted Cand array on device (copied to host for verification):\n";
    for (int i = 0; i < last_element; ++i) {
        const Cand& cand = h_sortedCandHeap[i];
        std::cout << "h_sortedCandHeap[" << i << "] = { mt: " << cand.mt
            << ", mdt: " << cand.mdt
            << ", mwidth: " << cand.mwidth
            << ", msnr: " << cand.msnr << " }\n";
    }

    return 0;
}
