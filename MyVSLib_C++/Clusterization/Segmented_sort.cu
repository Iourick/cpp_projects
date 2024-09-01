#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <cstddef> // For offsetof
#include <cstdlib> // For rand()
#include <ctime> // For time()
#include <algorithm> // For std::shuffle
#include "Segmented_sort.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cub/cub.cuh>


namespace segmented_sort
{   
    void sort_subarrays(const thrust::device_vector<Cand>& d_vctCandHeap,
        thrust::device_vector<int>& d_vctOffset,
        thrust::device_vector<int>& d_indices,
        int member_offset)
    {
        int* pind = thrust::raw_pointer_cast(d_indices.data());
        thrust::host_vector<int> h_vctOffset = d_vctOffset;
        for (int i = 0; i < (h_vctOffset.size() - 1); ++i)
        {
            int start_index = h_vctOffset[i];

            int end_index = h_vctOffset[i + 1];
            int length = end_index - start_index;
            thrust::device_ptr<int> dev_ptr_start = thrust::device_pointer_cast(pind + start_index);
            thrust::device_ptr<int> dev_ptr_end = dev_ptr_start + length;

            // Reintroduce the sorting step with additional checks
            try {
                thrust::sort(thrust::device, dev_ptr_start, dev_ptr_end,
                    CompareCandMember(thrust::raw_pointer_cast(d_vctCandHeap.data()), member_offset));

            }
            catch (std::runtime_error& e) {
                std::cerr << "Runtime error during sorting: " << e.what() << std::endl;
            }
        }
    }
    //--------------------------------------------------------------------------------------
    void sort_subarrays_cub(const thrust::device_vector<Cand>& d_vctCandHeap,
        thrust::device_vector<int>& d_vctOffset,
        thrust::device_vector<int>& d_indices,
        int member_offset)
    {
        int* d_arrKeys = nullptr;
        const int QUantElements = d_vctCandHeap.size();
        const int num_segments = d_vctOffset.size() - 1;
        cudaMalloc((void**)&d_arrKeys, QUantElements * sizeof(int));
        int threads = 1024;
        int blocks = (QUantElements + threads - 1) / threads;
        extract_keys_kernel << < blocks, threads >> > (thrust::raw_pointer_cast(d_vctCandHeap.data())
            , QUantElements
            , thrust::raw_pointer_cast(d_indices.data())
            , member_offset
            , d_arrKeys);
        int* d_values_in = thrust::raw_pointer_cast(d_indices.data());
        int* d_offsets = thrust::raw_pointer_cast(d_vctOffset.data());
        // Determine temporary device storage requirements
        void* d_temp_storage = nullptr;
        size_t   temp_storage_bytes = 0;
        cub::DeviceSegmentedSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_arrKeys, d_arrKeys, d_values_in, d_values_in,
            QUantElements, num_segments, d_offsets, d_offsets + 1);

        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // Run sorting operation
        cub::DeviceSegmentedSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_arrKeys, d_arrKeys, d_values_in, d_values_in,
            QUantElements, num_segments, d_offsets, d_offsets + 1);

        cudaFree(d_temp_storage);
        cudaFree(d_arrKeys);
    }
    //----------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------
    void sort_subarrays_cub_(const Cand* d_parrCand,
        const int QUantElements,
        thrust::device_vector<int>& d_vctOffset,
        thrust::device_vector<int>& d_indices,
        int member_offset)
    {
        int* d_arrKeys = nullptr;       
        const int num_segments = d_vctOffset.size() - 1;
        cudaMalloc((void**)&d_arrKeys, QUantElements * sizeof(int));
        int threads = 1024;
        int blocks = (QUantElements + threads - 1) / threads;
        extract_keys_kernel << < blocks, threads >> > (d_parrCand
            , QUantElements
            , thrust::raw_pointer_cast(d_indices.data())
            , member_offset
            , d_arrKeys);
        int* d_values_in = thrust::raw_pointer_cast(d_indices.data());
        int* d_offsets = thrust::raw_pointer_cast(d_vctOffset.data());
        // Determine temporary device storage requirements
        void* d_temp_storage = nullptr;
        size_t   temp_storage_bytes = 0;
        cub::DeviceSegmentedSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_arrKeys, d_arrKeys, d_values_in, d_values_in,
            QUantElements, num_segments, d_offsets, d_offsets + 1);

        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // Run sorting operation
        cub::DeviceSegmentedSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_arrKeys, d_arrKeys, d_values_in, d_values_in,
            QUantElements, num_segments, d_offsets, d_offsets + 1);

        cudaFree(d_temp_storage);
        cudaFree(d_arrKeys);
    }

} // !namespace segmented_sort

__global__ void validate_access(int* indices, Cand* d_vctCandHeap, int member_offset, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        const char* base = reinterpret_cast<const char*>(&d_vctCandHeap[indices[idx]]);
        int value = *reinterpret_cast<const int*>(base + member_offset);
        printf("Accessing idx: %d, value: %d\n", indices[idx], value);
    }
}


void checkCudaErrors(cudaError_t err)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(err);
    }
}
void sort_subarrays(const thrust::device_vector<Cand>& d_vctCandHeap,
    thrust::host_vector<int>& h_vctGroupBeginIndices,
    thrust::device_vector<int>& d_indices,
    int member_offset)
{
    //// Print the sizes of the vectors for debugging
    //std::cout << "d_vctCandHeap size: " << d_vctCandHeap.size() << std::endl;
    //std::cout << "h_vctGroupBeginIndices size: " << h_vctGroupBeginIndices.size() << std::endl;
    //std::cout << "d_indices size: " << d_indices.size() << std::endl;

    int* pind = thrust::raw_pointer_cast(d_indices.data());

    // std::cout << "Raw pointers obtained successfully" << std::endl; // Debug print

     // Copy device data to host for validation
    /* thrust::host_vector<int> h_indices = d_indices;
     thrust::host_vector<Cand> h_candHeap = d_vctCandHeap;
     std::cout << "Indices on Device: ";
     for (int i = 0; i < h_indices.size(); ++i) {
         std::cout << h_indices[i] << " ";
     }
     std::cout << std::endl;

     std::cout << "CandHeap on Device: ";
     for (int i = 0; i < h_candHeap.size(); ++i) {
         std::cout << "{" << h_candHeap[i].mt << ", " << h_candHeap[i].mdt << ", " << h_candHeap[i].mwidth << ", " << h_candHeap[i].msnr << "} ";
     }
     std::cout << std::endl;*/

     // Validate access using a kernel
    // int threadsPerBlock = 256;
     //int blocksPerGrid = (d_indices.size() + threadsPerBlock - 1) / threadsPerBlock;
    /* validate_access << <blocksPerGrid, threadsPerBlock >> > (pind, const_cast<Cand*>(thrust::raw_pointer_cast(d_vctCandHeap.data())), member_offset, d_indices.size());
     checkCudaErrors(cudaDeviceSynchronize());*/

    for (int i = 0; i < h_vctGroupBeginIndices.size(); ++i)
    {
        //std::cout << "Processing chunk " << i << std::endl; // Debug print

        // Ensure indices are within bounds
        /*if (i >= h_vctGroupBeginIndices.size()) {
            std::cerr << "Index out of bounds: " << i << std::endl;
            return;
        }*/

        int start_index = h_vctGroupBeginIndices[i];
        int end_index = (i == h_vctGroupBeginIndices.size() - 1) ? d_indices.size() : h_vctGroupBeginIndices[i + 1];
        int length = end_index - start_index;

        // std::cout << "start_index: " << start_index << ", end_index: " << end_index << ", length: " << length << std::endl; // Debug print

        if (start_index < 0 || end_index > d_indices.size() || length <= 0) {
            std::cerr << "Invalid indices or length: start_index = " << start_index << ", end_index = " << end_index << ", length = " << length << std::endl;
            continue;
        }

        // Additional debugging: print the values of indices being sorted
        /*thrust::host_vector<int> h_debug_indices(d_indices.begin() + start_index, d_indices.begin() + end_index);
        std::cout << "Indices to be sorted: ";
        for (int j = 0; j < h_debug_indices.size(); ++j)
        {
            std::cout << h_debug_indices[j] << " ";
        }
        std::cout << std::endl;*/

        thrust::device_ptr<int> dev_ptr_start = thrust::device_pointer_cast(pind + start_index);
        thrust::device_ptr<int> dev_ptr_end = dev_ptr_start + length;

        //std::cout << "Pointers to subvector obtained successfully" << std::endl; // Debug print

        // Print memory addresses
        //std::cout << "dev_ptr_start: " << dev_ptr_start.get() << ", dev_ptr_end: " << dev_ptr_end.get() << std::endl;

        //// Validate the data before sorting
        //thrust::host_vector<int> h_debug_sort_data(dev_ptr_start, dev_ptr_end);
        //for (int j = 0; j < h_debug_sort_data.size(); ++j) {
        //    const char* base = reinterpret_cast<const char*>(&h_candHeap[h_debug_sort_data[j]]);
        //    int value = *reinterpret_cast<const int*>(base + member_offset);
        //    std::cout << "Index " << h_debug_sort_data[j] << " Value: " << value << std::endl;
        //}

        // Reintroduce the sorting step with additional checks
        try {
            thrust::sort(thrust::device, dev_ptr_start, dev_ptr_end,
                CompareCandMember(thrust::raw_pointer_cast(d_vctCandHeap.data()), member_offset));
            checkCudaErrors(cudaGetLastError());
        }
        catch (std::runtime_error& e) {
            std::cerr << "Runtime error during sorting: " << e.what() << std::endl;
        }

        //std::cout << "Subvector sorted successfully" << std::endl; // Debug print
    }

    // Ensure synchronization after sorting
  //  checkCudaErrors(cudaDeviceSynchronize());
}
//--------------------------------------------------------------------------------------
void sort_subarrays_cub(const thrust::device_vector<Cand>& d_vctCandHeap,
    thrust::device_vector<int>& d_vctOffset,
    thrust::device_vector<int>& d_indices,
    int member_offset)
{
    int* d_arrKeys = nullptr;
    const int QUantElements = d_vctCandHeap.size();
    const int num_segments = d_vctOffset.size() - 1;
    cudaMalloc((void**)&d_arrKeys, QUantElements * sizeof(int));
    int threads = 1024;
    int blocks = (QUantElements + threads - 1) / threads;
    extract_keys_kernel << < blocks, threads >> > (thrust::raw_pointer_cast(d_vctCandHeap.data())
        , QUantElements
        , thrust::raw_pointer_cast(d_indices.data())
       ,  member_offset
        , d_arrKeys);
    int* d_values_in = thrust::raw_pointer_cast(d_indices.data());       
    int* d_offsets = thrust::raw_pointer_cast(d_vctOffset.data());
    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;
    cub::DeviceSegmentedSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_arrKeys, d_arrKeys, d_values_in, d_values_in,
        QUantElements, num_segments, d_offsets, d_offsets + 1);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sorting operation
    cub::DeviceSegmentedSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_arrKeys, d_arrKeys, d_values_in, d_values_in,
        QUantElements, num_segments, d_offsets, d_offsets + 1);

    cudaFree(d_temp_storage);
    cudaFree(d_arrKeys);
}
//-----------------------------------------------------------------------------
__global__
void extract_keys_kernel(const Cand* d_arrCand
    , const int QUantElements
    , const int* d_arrIndeces
    , int member_offset
    , int* d_arrKeys)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= QUantElements)
    {
        return;
    }
    const char* base = reinterpret_cast<const char*>(&d_arrCand[d_arrIndeces[index]]);
    // Get the value of the member using the offset   
    d_arrKeys[index] = *reinterpret_cast<const int*>(base + member_offset);
}