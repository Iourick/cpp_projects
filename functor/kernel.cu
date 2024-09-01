#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <cuda_runtime.h>

// Functor to digitize the values
struct DigitizeFunctor {
    float max_value;

    DigitizeFunctor(float max_value) : max_value(max_value) {}

    __host__ __device__
        int operator()(const float& x) const {
        return static_cast<int>(x * (1 << 16) / max_value);
    }
};

// CUDA kernel to initialize the device vector
__global__ void initialize_vector(float* d_vec, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        d_vec[idx] = static_cast<float>(idx);
    }
}

int main() {
    int NUms = 128;

    // Initialize the device vector
    thrust::device_vector<float> d_normalized_fdmt(NUms);

    // Launch the kernel to fill d_normalized_fdmt with values from 0 to 127
    int blockSize = 256;
    int numBlocks = (NUms + blockSize - 1) / blockSize;
    initialize_vector << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(d_normalized_fdmt.data()), NUms);
    cudaDeviceSynchronize();

    // Calculate the maximum value in d_normalized_fdmt
    auto max_iter = thrust::max_element(d_normalized_fdmt.begin(), d_normalized_fdmt.end());
    float max_value = *max_iter;

    // Print the maximum value for debugging
    std::cout << "Max value: " << max_value << std::endl;

    // Create the d_digitized vector
    thrust::device_vector<int> d_digitized(NUms);

    // Apply the transformation to digitize the values
    thrust::transform(d_normalized_fdmt.begin(), d_normalized_fdmt.end(),
        d_digitized.begin(), DigitizeFunctor(max_value));

    // Copy the result to host and print for verification
    thrust::host_vector<int> h_digitized = d_digitized;
    for (int i = 0; i < NUms; ++i) {
        std::cout << "d_digitized[" << i << "] = " << h_digitized[i] << std::endl;
    }

    return 0;
}
