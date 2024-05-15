
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <iostream>
struct structTemp {
    int ma;
    int md;
};
int main()
{
    std::vector<std::vector<int>> coordinates;

    thrust::device_vector<structTemp > d_vec_struct(10);
    
    // Initialize a device vector with 10 elements
    thrust::device_vector<int> d_vec(10);
    thrust::device_vector<int> d_vec1(1);
    d_vec1 = d_vec1;
    // Fill the vector with some values
    for (int i = 0; i < d_vec.size(); i++) {
        d_vec[i] = 10 - i;
    }
    d_vec1 = d_vec;
    int ia = d_vec1[0];
    // Sort the vector in ascending order
    thrust::sort(d_vec.begin(), d_vec.end());

    // Print the sorted values
    std::cout << "Sorted values: ";
    for (int i = 0; i < d_vec.size(); i++) {
        std::cout << d_vec[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
