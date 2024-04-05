#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math_functions.h>
#include "Detection.cuh"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include "yr_cart.h"
#include <stdio.h>
void detect_signal_gpu(fdmt_type_* d_arr, fdmt_type_* d_norm, const int Rows
    , const int  Cols, const int WndWidth, const dim3 gridSize, const dim3 blockSize
    , float* d_pAuxArray, int* d_pAuxNumArray, int* d_pWidthArray, structOutDetection* pstructOut)
{ 
    //cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(&deviceProp, 0); // Assuming GPU device 0

    //// Access properties
    //int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    //int maxBlocksPerGrid = deviceProp.maxGridSize[0];
    //int maxSharedMemoryPerBlock = deviceProp.sharedMemPerBlock;
    
    cudaError_t cudaStatus;
    size_t sz = (sizeof(float) + sizeof(int)) * blockSize.x + 2 * sizeof(int) + sizeof(float);
    multi_windowing_kernel << < gridSize, blockSize/*, sz*/ >> > (d_arr, d_norm
        , Cols, WndWidth, d_pAuxArray, d_pAuxNumArray, d_pWidthArray);
    cudaDeviceSynchronize();
    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));       
    }
    int block_size = 1024;
    size_t sz0 = block_size * (sizeof(int) + sizeof(float));
    complete_detection_kernel << < 1, block_size, sz0 >> > (Cols, gridSize.x * gridSize.y, d_pAuxArray, d_pAuxNumArray
        , d_pWidthArray, pstructOut);
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    
}
//----------------------------------------------------
__global__
void multi_windowing_kernel(fdmt_type_* d_arr, fdmt_type_* d_norm, const int  Cols
    , const int WndWidth, float* d_pAuxArray, int* d_pAuxIntArray, int* d_pWidthArray)
{
  //  /*extern*/ __shared__ char buff[1024 *( sizeof(float) + sizeof(int))+ 2 * sizeof(int) + sizeof(float)];
    /*extern*/ __shared__ float arr_buff[1024];
    /*extern*/ __shared__ int arr_nums[1024];
    /*extern */__shared__ float pmax[1];
    /*extern*/ __shared__ int pnum[1];
    /*extern*/ __shared__ int pwidth[1];
    
    int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Cols)
    {
        return;
    }
    int nrow = blockIdx.y;  


    /*float* arr_buff = (float*)buff;
    int* arr_nums = (int*)(arr_buff + blockDim.x);
    float* pmax = (float*)(arr_nums + blockDim.x);
    int* pnum = (int*)(pmax + 1);
    int* pwidth = (pnum + 1);*/

    // initialization
    pmax[0] = -1000000.0;// /*1.0 - FLT_MAX*/;
    pnum[0] = -1;
    pwidth[0] = -1;

    //------------------------------------

    for (int iw = 1; iw < WndWidth + 1; ++iw)
    {
        if (idx + iw - 1 < Cols)
        {
            float sig2 = 0;
            
            for (int j = 0; j < iw ; ++j)
            {               
                sig2 +=  (float)d_arr[nrow * Cols + idx + j] / sqrtf((float)d_norm[nrow * Cols + idx + j] + 0.0000001);
                
            }
            arr_buff[tid] = sig2/sqrtf((float)(iw));
            arr_nums[tid] = nrow * Cols + idx;
        }
        else
        {
            arr_buff[tid] = 1.0 - FLT_MAX;
            arr_nums[tid] = -1;
        }


        __syncthreads();

        for (unsigned int s = (blockDim.x) / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                if (arr_buff[tid] < arr_buff[tid + s])
                {
                    arr_buff[tid] = arr_buff[tid + s];
                    arr_nums[tid] = arr_nums[tid + s];
                }
            }
            __syncthreads();
        }

        // Only thread 0 within each block computes the block's sum
        if (tid == 0)
        {
            if (arr_buff[0] > (pmax[0]))
            {
                //printf("arr_buff[0] = %f ; blockIdx.y * gridDim.x + blockIdx.x = %d \n ", arr_buff[0], blockIdx.y * gridDim.x + blockIdx.x);

                pmax[0] = arr_buff[0];
                pnum[0] = arr_nums[0];
                pwidth[0] = iw;


            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        //printf("*pmax = %f ; blockIdx.y * gridDim.x + blockIdx.x = %d \n ", pmax[0], blockIdx.y * gridDim.x + blockIdx.x);
        d_pAuxArray[blockIdx.y * gridDim.x + blockIdx.x] = pmax[0];
        d_pAuxIntArray[blockIdx.y * gridDim.x + blockIdx.x] =  pnum[0];
        d_pWidthArray[blockIdx.y * gridDim.x + blockIdx.x] =  pwidth[0];
    }
    __syncthreads();
}

//-----------------------------------------------------
__global__
void calcWindowedImage_kernel(fdmt_type_* d_arr, fdmt_type_* d_norm, const int  Cols
    , const int WndWidth, float* d_pOutArray)
{
    extern __shared__ char buff[];

    int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= Cols)
    {
        return;
    }

    if (idx + WndWidth - 1 < Cols)
    {
        float sig2 = 0;

        for (int j = 0; j < WndWidth; ++j)
        {

            sig2 += (float)d_arr[blockIdx.y * Cols + idx + j] / sqrtf((float)d_norm[blockIdx.y * Cols + idx + j] + 0.0000001);

        }
        d_pOutArray[blockIdx.y * Cols +tid] = sig2 / sqrtf((float)(WndWidth));
        
    }
    else
    {
        d_pOutArray[blockIdx.y * Cols + tid] = 0.0;
        
    }
}
//----------------------------------------------
//void detect_signal_gpu(fdmt_type_* d_arr, fdmt_type_* d_norm, const int Rows
//    , const int  Cols, const int WndWidth, const dim3 gridSize, const dim3 blockSize
//    , float* d_pAuxArray, int* d_pAuxNumArray, int* d_pWidthArray, structOutDetection* pstructOut)
//{
//    size_t sz = (sizeof(float) + sizeof(int)) * blockSize.x /*+ 2 * sizeof(int) + sizeof(float)*/;
//    multi_windowing_kernel << < gridSize, blockSize/*, sz*/ >> > (d_arr, d_norm
//        , Cols, WndWidth, d_pAuxArray, d_pAuxNumArray, d_pWidthArray);
//    cudaDeviceSynchronize();
//    float* pAuxArray = (float*)malloc(gridSize.x * gridSize.y * sizeof(float));
//    int* pAuxIntArray = (int*)malloc(gridSize.x * gridSize.y * sizeof(int));
//    int* pWidthArray = (int*)malloc(gridSize.x * gridSize.y * sizeof(int));
//    cudaMemcpy(pAuxArray, d_pAuxArray, gridSize.x * gridSize.y * sizeof(float), cudaMemcpyDeviceToHost);
//    cudaDeviceSynchronize();
//    cudaMemcpy(pWidthArray, d_pWidthArray, gridSize.x * gridSize.y * sizeof(int), cudaMemcpyDeviceToHost);
//    cudaDeviceSynchronize();
//    cudaMemcpy(pAuxIntArray, d_pAuxNumArray, gridSize.x * gridSize.y * sizeof(int), cudaMemcpyDeviceToHost);
//    cudaDeviceSynchronize();
//
//
//    fdmt_type_* arr = (fdmt_type_*)malloc(Rows * Cols * sizeof(fdmt_type_));
//    fdmt_type_* norm = (fdmt_type_*)malloc(Rows * Cols * sizeof(fdmt_type_));
//    float* pAuxArray0 = (float*)malloc(gridSize.x * gridSize.y * sizeof(float));
//    int* pAuxIntArray0 = (int*)malloc(gridSize.x * gridSize.y * sizeof(int));
//    int* pWidthArray0 = (int*)malloc(gridSize.x * gridSize.y * sizeof(int));
//    cudaMemcpy(arr, d_arr, Rows * Cols * sizeof(fdmt_type_), cudaMemcpyDeviceToHost);
//    cudaMemcpy(norm, d_norm, Rows * Cols * sizeof(fdmt_type_), cudaMemcpyDeviceToHost);
//
//    multi_windowing_cpu(arr, norm, Cols, WndWidth, pAuxArray0, pAuxIntArray0, pWidthArray0, gridSize, blockSize);
//
//    float valmax1 = -0., valmin1 = 0.;
//    unsigned int iargmax1 = -1, iargmin1 = -1;
//    findMaxMinOfArray(pAuxArray0, gridSize.x * gridSize.y, &valmax1, &valmin1
//        , &iargmax1, &iargmin1);
//
//    int block_size = min(128, gridSize.x * gridSize.y);
//    size_t sz0 = block_size * (sizeof(int) + sizeof(float));
//    complete_detection_kernel << < 1, block_size, sz0 >> > (Cols, gridSize.x * gridSize.y, d_pAuxArray, d_pAuxNumArray
//        , d_pWidthArray, pstructOut);
//    cudaDeviceSynchronize();
//
//
//    free(arr);
//    free(norm);
//   
//}

void multi_windowing_cpu(fdmt_type_* arr, fdmt_type_* norm, const int  Cols
    , const int WndWidth, float* pAuxArray, int* pAuxIntArray, int* pWidthArray, const dim3 gridSize, const dim3 blockSize)
{
    const int NUmGrigRows = gridSize.y;
    const int NUmGrigCols = gridSize.x;
    float sum2 = 0.;
    float norm2 = 0.;
    for (int i = 0; i < NUmGrigRows; ++i)
    {
        
        for (int j = 0; j < NUmGrigCols; ++j)
        {
            if ((j == (NUmGrigCols - 1))&&(i == 1))
            {
                int yyy = 0;
            }
            pAuxArray[i * NUmGrigCols + j] = 1. - FLT_MAX;
            int num_begin = i * Cols + j * blockSize.x;
            for (int k = 0; k < blockSize.x; ++k)
            {
                
                for (int iw = 1; iw <= WndWidth; ++iw)
                {
                    
                    sum2 = 0.;
                    norm2 = 0.;
                    for (int q = 0; q < iw ; ++q)
                    {
                        if (j * blockSize.x + k + q >= Cols)
                        {
                            sum2 = 1. - FLT_MAX;
                            norm2 = 1.;
                            break;
                        }
                        
                        float t = norm[num_begin + k + q] + 0.00001;// max((float)norm[num_begin + k + q], 1.);
                        sum2 += ((float)arr[num_begin + k + q])/sqrt(t);
                        
                    }
                    sum2 = sum2 / sqrt((float)iw);
                    if (sum2  > pAuxArray[i * NUmGrigCols + j])
                    {
                        pAuxArray[i * NUmGrigCols + j] = sum2 ;
                        pAuxIntArray[i * NUmGrigCols + j] = num_begin + k;
                        pWidthArray[i * NUmGrigCols + j] = iw ;
                    }
                    
                }
                
            }
            
        }
    }
}
//----------------------------------------------

__global__
void complete_detection_kernel(const int Cols, const int LEnGrid, float* d_pAuxArray, int* d_pAuxNumArray
    , int* d_pWidthArray, structOutDetection* pstructOut)
{

    extern __shared__ char buff[];
    float* arr = (float*)buff;
    int* nums = (int*)(arr + blockDim.x);


    float val_loc = -1000.;
    int num_loc = -1;


    for (int i = threadIdx.x; i < LEnGrid; i += blockDim.x)
    {
        if (d_pAuxArray[i] > val_loc)
        {
            val_loc = d_pAuxArray[i];
            num_loc = i;
        }
    }
    arr[threadIdx.x] = val_loc;
    nums[threadIdx.x] = num_loc;
    __syncthreads();


    for (unsigned int s = (blockDim.x) / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            if (arr[threadIdx.x] < arr[threadIdx.x + s])
            {
                arr[threadIdx.x] = arr[threadIdx.x + s];
                nums[threadIdx.x] = nums[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        pstructOut->snr =  arr[0];
        pstructOut->irow = d_pAuxNumArray[nums[0]] / Cols;
        pstructOut->icol = d_pAuxNumArray[nums[0]] % Cols;
        pstructOut->iwidth = d_pWidthArray[nums[0]];
        //printf("!!! %f \n", arr[0]);
        
    }
    __syncthreads();

}