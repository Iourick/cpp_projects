
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <iostream>

// Assuming SizeType is some type that can be converted to int
using SizeType = size_t;

int main() {
    // Example std::vector<SizeType>
    std::vector<SizeType> state_shape_flattened_s = { 1, 2, 3, 4, 5 };

    // Convert std::vector<SizeType> to thrust::host_vector<int>
    thrust::host_vector<int> state_shape_h(state_shape_flattened_s.begin(), state_shape_flattened_s.end());

    // Allocate thrust::device_vector<int> of the same size
    thrust::device_vector<int> state_shape_d(state_shape_h.size());

    // Copy data from host vector to device vector
    thrust::copy(state_shape_h.begin(), state_shape_h.end(), state_shape_d.begin());

    // Verify the copy (optional, just for demonstration)
    thrust::host_vector<int> state_shape_h_result = state_shape_d; // Copy back to host to print
    for (int val : state_shape_h_result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
