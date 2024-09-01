#include <thrust/device_vector.h>
#include <iostream>
#include <cuda_runtime.h>
#include <iostream>
#include "cumsum_customized.cuh"
#include <cub/cub.cuh>

__global__
void calc_cumsum_kernel_v0(const int* d_arrRegroupingPlan, const int LEnarr, int* d_arr_cumsum)
{
    extern __shared__ int  iarr[];
    if (!(blockIdx.x < LEnarr))
    {
        return;
    }
    int isum = 0;
    int numElem = blockIdx.x;
    int id = threadIdx.x;
    for (int i = id; i <= numElem; i += blockDim.x)
    {
        isum += d_arrRegroupingPlan[i];
    }
    iarr[id] = isum;
    __syncthreads();

    // Parallel reduction 
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (id < s)
        {
            iarr[id] += iarr[id + s];
        }
        __syncthreads();
    }
    if (id == 0)
    {
        d_arr_cumsum[blockIdx.x] = iarr[0];
    }
    __syncthreads();
}


__global__
void calc_part_cumsum_kernel(const int* d_arrRegroupingPlan, const int LEnarr, int* d_arr_cumsum)
{
    if (threadIdx.x >= LEnarr)
    {
        return;
    }
    int indexElem = blockIdx.x * blockDim.x + threadIdx.x;
    int loopEnd = (blockIdx.x + 1) * blockDim.x;
    if (loopEnd > LEnarr)
    {
        loopEnd = LEnarr;
    }
    int isum = 0;

    for (int i = threadIdx.x; i < loopEnd; i += blockDim.x)
    {
        isum += d_arrRegroupingPlan[i];
    }
    d_arr_cumsum[indexElem] = isum;

    __syncthreads();
}

__global__
void calc_part_cumsum_kernel_new(const int* d_arrRegroupingPlan, const int LEnarr, int* d_arr_cumsum, unsigned long long clock_rate)
{
    extern __shared__ int  iarr[];
    if (threadIdx.x >= LEnarr)
    {
        return;
    }
    int indexElem = blockIdx.x * blockDim.x + threadIdx.x;
    int loopEnd = (blockIdx.x + 1) * blockDim.x;
    if (loopEnd > LEnarr)
    {
        loopEnd = LEnarr;
    }
    int isum = 0;

    for (int i = threadIdx.x; i < loopEnd; i += blockDim.x)
    {
        isum += d_arrRegroupingPlan[i];
    }
    iarr[threadIdx.x] = isum;
    //d_arr_cumsum[indexElem] = isum;
    __syncthreads();


    const unsigned int wait_time_us = 100; // 1000 microseconds (1 millisecond)
    const unsigned long long start_clock = clock64();
    unsigned long long clock_offset;
    do
    {
        clock_offset = clock64() - start_clock;
    } while (clock_offset < wait_time_us * (clock_rate / 1000000ULL)); // Adjust for clock rate
    if (indexElem < LEnarr)
    {
        d_arr_cumsum[indexElem] = iarr[threadIdx.x];
    }
}
//--------------------------------------------------------------------------
__global__
void calc_part_cumsum_kernel_v1(const int* d_arrRegroupingPlan, const int LEnarr, int* d_arr_cumsum)
{
    if (threadIdx.x >= LEnarr)
    {
        return;
    }
    int isum = 0;
    for (int i = threadIdx.x; i < LEnarr; i += blockDim.x)
    {
        isum += d_arrRegroupingPlan[i];
        d_arr_cumsum[i] = isum;
    }
}
//---------------------------------------------------------
__global__
void complete_cumsum_kernel(int* d_arrRegroupingPlan, const int LEnarr, int* d_arr_partSum)
{
    extern __shared__ int  iarr[];
    int indexElem = blockIdx.x * blockDim.x + threadIdx.x;
    int iwidth = blockDim.x;
    // printf("threadIdx.x = %i \n", threadIdx.x);
    if (indexElem >= LEnarr)
    {
        return;
    }
    else
    {
        int isum = 0;
        int ind_start = indexElem - iwidth + 1; //max(0, indexElem - iwidth + 1);
        if (ind_start < 0)
        {
            ind_start = 0;
        }
        for (int i = ind_start; i < indexElem + 1; ++i)
        {
            isum += d_arr_partSum[i];
        }
        d_arrRegroupingPlan[indexElem] = isum;
    }
}
//---------------------------------------------------------------------
void calc_cumsum_custom_gpu(int* d_arrInput
    , int* d_arr_buff, const int LEnarr, const int treads_per_block, int* d_arrOutput)
{
    int  blocks_per_grid = (LEnarr + treads_per_block - 1) / treads_per_block;   
    calc_part_cumsum_kernel_v1 << < blocks_per_grid, treads_per_block >> >
        (d_arrInput, LEnarr, d_arr_buff);    
    complete_cumsum_kernel << < blocks_per_grid, treads_per_block, treads_per_block * sizeof(int) >> >
        (d_arrOutput, LEnarr, d_arr_buff);   
}
//------------------------------------------------------------------------

void fnc_cumsum_customized_gpu(int* d_arrInput
    , int* d_arr_buff, const int LEnarr, int* d_arrOutput)
{
    if (LEnarr < 800000)
    {
        int treads_per_block = 1024;
        int blocks_per_grid = LEnarr;

        calc_cumsum_kernel_v0 << < blocks_per_grid, treads_per_block, treads_per_block * sizeof(int) >> > (d_arrInput, LEnarr, d_arrOutput);
        return;
    }

    int  treads_per_block = 1024;
    int  blocks_per_grid = (LEnarr + treads_per_block - 1) / treads_per_block;
    calc_part_cumsum_kernel_v1 << < blocks_per_grid, treads_per_block >> >
        (d_arrInput, LEnarr, d_arr_buff);
   

    complete_cumsum_kernel << < blocks_per_grid, treads_per_block, treads_per_block * sizeof(int) >> >
        (d_arrOutput, LEnarr, d_arr_buff);
    return;
}

//----------------------------------------------------------
void fnc_cumsum_customized_gpu_new(int* d_arrInput
    , int* d_arr_buff, const int LEnarr, int* d_arrOutput,
    unsigned long long clock_rate)
{
    if (LEnarr < 800000)
    {
        int treads_per_block = 1024;
        int blocks_per_grid = LEnarr;

        calc_cumsum_kernel_v0 << < blocks_per_grid, treads_per_block, treads_per_block * sizeof(int) >> > (d_arrInput, LEnarr, d_arrOutput);
        return;
    }

    int  treads_per_block = 1024;
    int  blocks_per_grid = (LEnarr + treads_per_block - 1) / treads_per_block;
    /*calc_part_cumsum_kernel_v1 << < blocks_per_grid, treads_per_block >> >
        (d_arrInput, LEnarr, d_arr_buff);*/
    calc_part_cumsum_kernel_new << < blocks_per_grid, treads_per_block >> >
        (d_arrInput, LEnarr, d_arr_buff, clock_rate);

    complete_cumsum_kernel << < blocks_per_grid, treads_per_block, treads_per_block * sizeof(int) >> >
        (d_arrOutput, LEnarr, d_arr_buff);
    return;
}
//-----------------------------------------------
__global__
void scanKernel(int* d_out, int* d_in, int size) 
{
    extern __shared__ int temp[]; // shared memory for scan operation

    int thid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory
    int ai = thid;
    int bi = thid + blockDim.x;

    if (ai < size) temp[ai] = d_in[ai];
    if (bi < size) temp[bi] = d_in[bi];

    // Up-sweep (reduce) phase
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            if (bi < size) temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the last element for exclusive scan
    if (thid == 0) temp[2 * blockDim.x - 1] = 0;

    // Down-sweep phase
    for (int d = 1; d < 2 * blockDim.x; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            if (bi < size) {
                int t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
    }

    __syncthreads();

    // Write results to device memory
    if (ai < size) d_out[ai] = temp[ai];
    if (bi < size) d_out[bi] = temp[bi];
}
//------------------------------------------------------------------------------------

void inclusiveScan_CUB(float* d_arr, const int LEn)
{
    void* d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_arr
        , d_arr
        , LEn);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run exclusive prefix sum
    cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_arr
        , d_arr
        , LEn);

    cudaFree(d_temp_storage);
}

//------------------------------------------------------------------------------------

void inclusiveScan_CUB(int* d_arr, const int LEn)
{
    void* d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_arr
        , d_arr
        , LEn);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run exclusive prefix sum
    cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_arr
        , d_arr
        , LEn);

    cudaFree(d_temp_storage);
}
