
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

struct structTemp {
    int ma;
    int md;
};

int main() {
    // Host arrays
    int arra[] = { 1, 2, 3, 4, 5 };
    int arrd[] = { 6, 7, 8, 9, 10 };
    const int size = 5;  // Size of the arrays

    // Create a host vector of structTemp
    thrust::host_vector<structTemp> h_vec_struct(size);

    // Initialize host vector with values from arra and arrd
    for (int i = 0; i < size; ++i) {
        h_vec_struct[i].ma = arra[i];
        h_vec_struct[i].md = arrd[i];
    }

    // Copy the host vector to the device vector
    thrust::device_vector<structTemp> d_vec_struct = h_vec_struct;

    // To verify, copy back to host and print the values
    thrust::copy(d_vec_struct.begin(), d_vec_struct.end(), h_vec_struct.begin());
    for (int i = 0; i < size; ++i) {
        std::cout << "Element at index " << i << ": ma=" << h_vec_struct[i].ma
            << ", md=" << h_vec_struct[i].md << std::endl;
    }

    return 0;
}
