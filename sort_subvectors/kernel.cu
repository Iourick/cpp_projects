//IN THIS PROJECT IS BEING DONE SORTING OF SUBARRAYS.
//SUBARRAYS ARE STORED ON GPU.
//ALL SUBARRAYS BELONG TO ONE ARRAY.
//THEY ARE SEPARETED BY ARRAY / VECTOR CONTAINING NUMBERS OF THEIR FIRST ELEMENT
//THIS ARRAY WITH NUMBERS OF THEIR FIRST ELEMENT IS STORED ON CPU

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <cstddef> // For offsetof
#include <cstdlib> // For rand()
#include <ctime> // For time()
#include <algorithm> // For std::shuffle
#include <random> // For std::default_random_engine
#include <chrono> // For timing
#include "Segmented_sort.cuh"

 
//-----------------------------------------------------------------------------------------------------------------
/*-----  PROFILING VARIANT   ------------------------------------------------------------------------------------------------------*/
int main() {
    int quant_candidates = 1<<20; // Large dataset size
    int quantityGroups = 1<<10; // Number of chunks

    // Initialize random seed
    //std::srand(std::time(0));
    std::srand(0);

    // Initialize host vector with random example data
    thrust::host_vector<Cand> h_vctCandHeap(quant_candidates);
    for (int i = 0; i < quant_candidates; ++i)
    {
        h_vctCandHeap[i].mt = std::rand() % 100;
        h_vctCandHeap[i].mdt = std::rand() % 100;
        h_vctCandHeap[i].mwidth = std::rand() % 100;
        h_vctCandHeap[i].msnr = static_cast<float>(std::rand()) / RAND_MAX;
    }
    int ihih = h_vctCandHeap[20].mt;
    // Copy to device vector
    thrust::device_vector<Cand> d_vctCandHeap = h_vctCandHeap;

    // Initialize device vector with permuted indices
    thrust::host_vector<int> h_vctIndecies(quant_candidates);
    for (int i = 0; i < quant_candidates; ++i) {
        h_vctIndecies[i] = i;
    }

    // Shuffle indices
    //std::shuffle(h_vctIndecies.begin(), h_vctIndecies.end(), std::default_random_engine(std::time(0)));
    std::shuffle(h_vctIndecies.begin(), h_vctIndecies.end(), std::default_random_engine(0));
    thrust::device_vector<int> d_vctIndecies = h_vctIndecies;

    // Initialize host vector with group begin indices
    thrust::host_vector<int> h_vctOffset(quantityGroups );
    int chunk_size = quant_candidates / quantityGroups;
    for (int i = 0; i < quantityGroups; ++i)
    {
        h_vctOffset[i] = i * chunk_size;
    }
    h_vctOffset.push_back(quant_candidates);

    // Ensure synchronization
    checkCudaErrors(cudaDeviceSynchronize());

        // Initialize host vector with group begin indices
   
  
    thrust::device_vector<int> d_vctOffset = h_vctOffset;

    // Define the member offset (for example, mt)
    int member_offset = offsetof(Cand, mt);

    // Call the sort_subarrays function
    auto start = std::chrono::high_resolution_clock::now();
    //segmented_sort::sort_subarrays(d_vctCandHeap, d_vctOffset, d_vctIndecies, member_offset);

   segmented_sort::sort_subarrays_cub(d_vctCandHeap, d_vctOffset, d_vctIndecies, member_offset);
    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Print the elapsed time
    std::cout << "Time taken to sort subarrays: " << elapsed.count() << " seconds" << std::endl;
    // Ensure synchronization
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy sorted indices back to host for verification
    thrust::host_vector<int> h_sorted_indices = d_vctIndecies;
    std::vector<int> vctOut(h_sorted_indices.begin(), h_sorted_indices.end());

    
  

    return 0;
}



/*DEBUGGING VARIANT WITH PREDICTABLE RESULTS*/
/*--------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------------------------------------*/
//int main()
//{
//    int quant_candidates = 10; // Example size
//    int quantityGroups = 2; // Number of chunks
//
//    // Initialize host vector with example data
//    thrust::host_vector<Cand> h_vctCandHeap = {
//        {10, 2, 3, 4.0f}, {30, 5, 6, 7.0f}, {20, 8, 9, 10.0f}, {50, 11, 12, 13.0f}, {40, 14, 15, 16.0f},
//        {15, 3, 4, 5.0f}, {35, 6, 7, 8.0f}, {25, 9, 10, 11.0f}, {55, 12, 13, 14.0f}, {45, 15, 16, 17.0f}
//    };
//
//    for (size_t i = 0; i < h_vctCandHeap.size(); ++i) {
//        std::cout <<i<< "  {"
//            << h_vctCandHeap[i].mt << ", "
//            << h_vctCandHeap[i].mdt << ", "
//            << h_vctCandHeap[i].mwidth << ", "
//            << h_vctCandHeap[i].msnr << "}"
//            << std::endl;
//    }
//
//    // Copy to device vector
//    thrust::device_vector<Cand> d_vctCandHeap = h_vctCandHeap;
//
//    // Initialize device vector with permuted indices
//    thrust::host_vector<int> h_vctIndecies = { 3, 0, 4, 1, 2, 8, 5, 6, 9, 7 }; // Example permutation
//    for (size_t i = 0; i < h_vctIndecies.size(); ++i)
//    {
//        std::cout << h_vctIndecies[i];
//        if (i < h_vctIndecies.size() - 1) {
//            std::cout << ", ";
//        }
//    }
//    std::cout << std::endl;
//
//    thrust::device_vector<int> d_vctIndecies = h_vctIndecies;
//
//    // Initialize host vector with group begin indices
//    thrust::host_vector<int> h_vctOffset = { 0, 5,10 }; // Example chunk beginnings
//    thrust::device_vector<int>d_vctOffset = h_vctOffset;    
//
//    // Ensure synchronization
//    checkCudaErrors(cudaDeviceSynchronize());
//
//    // Define the member offset (for example, mt)
//    int member_offset = offsetof(Cand, mt);
//
//    // Call the sort_subarrays function
//    segmented_sort::sort_subarrays(d_vctCandHeap, d_vctOffset, d_vctIndecies, member_offset);
//   // segmented_sort::sort_subarrays_cub(d_vctCandHeap, d_vctOffset, d_vctIndecies, member_offset);
//    // Ensure synchronization
//    checkCudaErrors(cudaDeviceSynchronize());
//
//    // Copy sorted indices back to host for verification
//    thrust::host_vector<int> h_sorted_indices = d_vctIndecies;
//
//    // Print the sorted indices for each chunk
//    std::cout << "Sorted indices by chunks:\n";
//    for (int i = 0; i < quantityGroups; ++i) {
//        int start_index = h_vctOffset[i];
//        int end_index =  h_vctOffset[i + 1];
//        std::cout << "Chunk " << i << ": ";
//        for (int j = start_index; j < end_index; ++j) {
//            std::cout << h_sorted_indices[j] << " ";
//        }
//        std::cout << std::endl;
//    }
//
//    return 0;
//}
//






