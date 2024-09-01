#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "D://My_DMT//src//Clusterization//cumsum_customized.cuh"

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>

#include <thrust/gather.h>

#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cub/cub.cuh> // or equivalently <cub/device/device_scan.cuh>

//---------------------------------------------------------------------------
int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    unsigned long long clock_rate = prop.clockRate;


    int num_elements = 1 << 5 + 3;

    // Initialize device vector with example data
    thrust::device_vector<int> d_arr(num_elements, 1);
    thrust::device_vector<int> d_arr0(num_elements, 1);
    thrust::device_vector<int> d_out(num_elements);
    thrust::device_vector<int> d_out1(num_elements);
    thrust::device_vector<int> d_arr_(num_elements);
    // thrust::sequence(d_arr.begin(), d_arr.end(), 1); // Fill with 0, 1, 2, ..., num_elements-1
    d_arr_ = d_arr;
    d_arr0 = d_arr;

    thrust::device_vector<int> d_arr_cumsum(num_elements);
    thrust::device_vector<int> d_arr_temp(num_elements);
    thrust::host_vector<int> h_arr_cumsum(num_elements);

    // Print original device vector
    thrust::host_vector<int> h_arr = d_arr;
    std::cout << "Original vector: ";
    for (int i = 0; i < num_elements; ++i)
    {
        //  std::cout << h_arr[i] << " ";
    }
    std::cout << std::endl;

    // Compute cumulative sum

    auto start = std::chrono::high_resolution_clock::now();

    thrust::inclusive_scan(d_arr.begin(), d_arr.end(), d_out.begin());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  Time taken by function thrust::inclusive_scan : " << duration.count() << " microseconds" << std::endl;


    // Copy result back to host and print
    h_arr = d_out;
    std::vector<int> std_arr(h_arr.begin(), h_arr.end());
    std::cout << "Cumulative sum vector: ";
    for (int i = 0; i < num_elements; ++i)
    {
        // std::cout << h_arr[i] << " ";
    }
    std::cout << std::endl;

    int LEn = d_arr.size();
    int treads_per_block = 32;
    int blocks_per_grid = (d_arr.size());
    start = std::chrono::high_resolution_clock::now();

    // calc_cumsum_kernel_v0 << < blocks_per_grid, treads_per_block, treads_per_block * sizeof(int) >> > (thrust::raw_pointer_cast(d_arr_.data()), LEn, thrust::raw_pointer_cast(d_out1.data()));

     // cudaDeviceSynchronize();


       //cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //   std::cout << "  Time taken by function fncFdmtU_cu kertnel0 : " << duration.count()  << " microseconds" << std::endl;
      /* d_arr_ = d_arr0;

       thrust::host_vector<int> h_arr_ = d_arr_;
       std::vector<int> std_arr_(h_arr_.begin(), h_arr_.end());*/

    treads_per_block = 1024;
    int* d_iparrInp = thrust::raw_pointer_cast(d_arr_.data());
    start = std::chrono::high_resolution_clock::now();
    /* fnc_cumsum_customized_gpu(thrust::raw_pointer_cast(d_arr_.data())
         , thrust::raw_pointer_cast(d_arr_temp.data())
         , LEn
         ,  thrust::raw_pointer_cast(d_out1.data()));*/
    inclusiveScan_CUB(d_iparrInp, num_elements);
    //void* d_temp_storage = nullptr;
    //size_t   temp_storage_bytes = 0;
    //cub::DeviceScan::InclusiveSum(
    //    d_temp_storage, temp_storage_bytes,
    //    thrust::raw_pointer_cast(d_arr_.data())
    //    , thrust::raw_pointer_cast(d_arr_.data())
    //    , num_elements);

    //// Allocate temporary storage
    //cudaMalloc(&d_temp_storage, temp_storage_bytes);

    //// Run exclusive prefix sum
    //cub::DeviceScan::InclusiveSum(
    //    d_temp_storage, temp_storage_bytes,
    //    thrust::raw_pointer_cast(d_arr_.data())
    //    , thrust::raw_pointer_cast(d_arr_.data())
    //    , num_elements);

    //cudaFree(d_temp_storage);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  Time taken by function cub::DeviceScan::InclusiveSum: " << duration.count() << " microseconds" << std::endl;


    h_arr_cumsum = d_arr_;
    std::vector<int> std_arr_cumsum(h_arr_cumsum.begin(), h_arr_cumsum.end());
    std::cout << "Cumulative sum kernel vector: ";
    for (int i = 0; i < num_elements; ++i)
    {
        //std::cout << h_arr_cumsum[i] << " ";
    }
    std::cout << std::endl;
    int irez = 0;
    for (int i = 0; i < h_arr_cumsum.size(); ++i)
    {
        //int  it = abs(h_arr_cumsum[i] - h_arr[i]);

        int  it = std_arr_cumsum[i] - std_arr[i];
        if (it < 0)
        {
            it = -it;
        }
        if (it > irez)
        {
            irez = it;
            //   std::cout <<i<<"  "<< std_arr_cumsum[i] << "   " << std_arr[i] << std::endl;
        }
    }
    std::cout << "irez = " << irez << std::endl;

    int itemp = h_arr.back();
    int ir = 0;
    for (int i = (h_arr_cumsum.size() - 1 - 1024); i < h_arr_cumsum.size() - 1; ++i)
    {
        ir += h_arr_cumsum[i];
    }
    int ir1 = h_arr.back();
    d_arr_.clear();
    d_arr.clear();
    d_arr_cumsum.clear();
    return 0;
}
