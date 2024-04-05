
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math_functions.h>

#include <stdio.h>
#include <iostream>
#include "Constants.h"
#include <chrono>
#include "Detection.cuh"

#define warpSize 32// CUDA warp size

//struct structOutDetection 
//{
//    int iwidth;
//    int irow;
//    int icol;
//    float snr;
//};
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// Function to perform maximum reduction within a warp along with argmaximum
__device__ void warpMaxArgmax(float value, int idx, float* pmaxVal, int* pmaxIdx)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        int tempVal = __shfl_down_sync(0xffffffff, value, offset);
        int tempIdx = __shfl_down_sync(0xffffffff, idx, offset);
        if (tempVal > value)
        {
            value = tempVal;
            idx = tempIdx;
        }
    }
    if (threadIdx.x % warpSize == 0)
    {
        *pmaxVal = value;
        *pmaxIdx = idx;
    }
}
//--------------------------------------------
__global__
void multi_windowing_kernel_v1(fdmt_type_* d_arr, fdmt_type_* d_norm, const int Rows
    , const int  Cols, const int WndWidth, float* d_pAuxArray, int* d_pAuxIntArray, int* d_pWidthArray)
{
    int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int nrow = blockIdx.y;
    extern __shared__ char buff[];

    float* arr_cum_sum = (float*)buff;
    float* arr_buff = (arr_cum_sum + blockDim.x+ 1);
    
    int* arr_nums = (int*)(arr_buff + blockDim.x );
    float* pmax = (float*)(arr_nums + blockDim.x);
    int* pnum = (int*)(pmax + 1);
    float* pwidth = (float*)(pnum + 1);
   
    
    // 
    arr_cum_sum[0] = 0.;
    for (int i = 0; i < blockDim.x; ++i)
    {
        arr_cum_sum[i+1] = arr_cum_sum[i ] + (float)d_arr[nrow * Cols + blockIdx.x * blockDim.x + i]/sqrtf(0.0000001 + (float)d_norm[nrow * Cols + blockIdx.x * blockDim.x + i]);
        
    }
    for (int i = blockDim.x; i < blockDim.x + WndWidth; ++i)
    {
        arr_cum_sum[i + 1] = 1.0 - FLT_MAX / 25.0;
    }
    // initialization
    *pmax = 1.0 - FLT_MAX / 25.0;
    *pnum = -1;
    *pwidth = -1;

    //------------------------------------

    for (int iw = 1; iw < WndWidth + 1; ++iw)
    {
        if (idx + iw - 1 < Cols)
        {
           

            
            arr_buff[tid] = (arr_cum_sum[tid + iw] - arr_cum_sum[tid])/sqrtf((float)iw);
            arr_nums[tid] = nrow * Cols + idx;
        }
        else
        {
            arr_buff[tid] = 1.0 - FLT_MAX;
            arr_buff[tid] = -1;
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
            if (arr_buff[0] > (*pmax))
            {
                *pmax = arr_buff[0];
                *pnum = arr_nums[0];
                *pwidth = iw;
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        //printf("*pmax = %f ; blockIdx.y * gridDim.x + blockIdx.x = %d \n ", *pmax, blockIdx.y * gridDim.x + blockIdx.x);
        d_pAuxArray[blockIdx.y * gridDim.x + blockIdx.x] = *pmax;
        d_pAuxIntArray[blockIdx.y * gridDim.x + blockIdx.x] = *pnum;
        d_pWidthArray[blockIdx.y * gridDim.x + blockIdx.x] = *pwidth;
    }
    __syncthreads();
}
//----------------------------------------------------------
__global__
void multi_windowing_kernel_v2(fdmt_type_* d_arr, fdmt_type_* d_norm, const int Rows
    , const int  Cols, const int WndWidth, float* d_pAuxArray, int* d_pAuxIntArray, int* d_pWidthArray)
{
    int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int nrow = blockIdx.y;
    extern __shared__ char buff[];

    float* arr_sign_cum_sum = (float*)buff;
    float* arr_norm_cum_sum = (arr_sign_cum_sum + blockDim.x + WndWidth + 1);
    float* arr_buff = (arr_norm_cum_sum + blockDim.x + WndWidth + 1);
    int* arr_nums = (int*)(arr_buff + blockDim.x);
    float* pmax = (float*)(arr_nums + blockDim.x);
    int* pnum = (int*)(pmax + 1);
    float* pwidth = (float*)(pnum + 1);

    // initialization
    *pmax = -1.0;
    *pnum = -1;
    *pwidth = -1;
    arr_sign_cum_sum[0] = 0.;
    arr_norm_cum_sum[0] = 0.000001;
    int N1 = Cols - blockIdx.x * blockDim.x - blockDim.x - WndWidth;
    float valcur_s = 0;
    float valcur_n = 0;
    if (N1 < 0)
    {
        const int Num = Cols - blockIdx.x * blockDim.x;
        for (int i = 0; i < Num; ++i)
        {
            float ts = (float)d_arr[nrow * Cols + blockIdx.x * blockDim.x + i];
            valcur_s += ts * ts;
            arr_sign_cum_sum[i + 1] = valcur_s;


            float tn = (float)d_norm[nrow * Cols + blockIdx.x * blockDim.x + i];
            valcur_n += tn * tn;
            arr_norm_cum_sum[i + 1] = valcur_n;
        }
        for (int i = Num; i < N1 + blockDim.x + WndWidth; ++i)
        {
            arr_sign_cum_sum[i + 1] = -1000.;
            arr_norm_cum_sum[i + 1] = 1000.;
        }
    }
    else
    {
        for (int i = 0; i < blockDim.x + WndWidth; ++i)
        {
            float ts = (float)d_arr[nrow * Cols + blockIdx.x * blockDim.x + i];
            valcur_s += ts * ts;
            arr_sign_cum_sum[i + 1] = valcur_s;


            float tn = (float)d_norm[nrow * Cols + blockIdx.x * blockDim.x + i];
            valcur_n += tn * tn;
            arr_norm_cum_sum[i + 1] = valcur_n;
        }
    }


    //------------------------------------



    for (int iw = 0; iw < WndWidth; ++iw)
    {
        arr_buff[tid] = sqrtf((arr_sign_cum_sum[tid + 1 + iw] - arr_sign_cum_sum[tid]) / (arr_norm_cum_sum[tid + 1 + iw] - arr_norm_cum_sum[tid]));
        arr_nums[tid] = nrow * Cols + blockIdx.x * blockDim.x + tid;
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
            if (arr_buff[0] > (*pmax))
            {
                *pmax = arr_buff[0];
                *pnum = arr_nums[0];
                *pwidth = iw + 1;

            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_pAuxArray[blockIdx.y * gridDim.x + blockIdx.x] = *pmax;
        d_pAuxIntArray[blockIdx.y * gridDim.x + blockIdx.x] = *pnum;
        d_pWidthArray[blockIdx.y * gridDim.x + blockIdx.x] = *pwidth;
    }
    __syncthreads();
}

//----------------------------------------------------------
__global__
void multi_windowing_kernel_v3(fdmt_type_* d_arr, fdmt_type_* d_norm, const int Rows
    , const int  Cols, const int WndWidth, float* d_pAuxArray, int* d_pAuxIntArray
    , int* d_pWidthArray, structOutDetection* pstructOut)
{
    int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int nrow = blockIdx.y;
    extern __shared__ char buff[];

    float* arr_sign_cum_sum = (float*)buff;
    float* arr_norm_cum_sum = (arr_sign_cum_sum + blockDim.x + WndWidth + 1);
    float* arr_buff = (arr_norm_cum_sum + blockDim.x + WndWidth + 1);
    int* arr_nums = (int*)(arr_buff + blockDim.x);
    float* pmax = (float*)(arr_nums + blockDim.x);
    int* pnum = (int*)(pmax + 1);
    float* pwidth = (float*)(pnum + 1);

    // initialization
    *pmax = -1.0;
    *pnum = -1;
    *pwidth = -1;
    arr_sign_cum_sum[0] = 0.;
    arr_norm_cum_sum[0] = 0.;
    int N1 = Cols - blockIdx.x * blockDim.x - blockDim.x - WndWidth;
    if (N1 < 0)
    {
        const int Num = Cols - blockIdx.x * blockDim.x;
        for (int i = 0; i < Num; ++i)
        {
            float ts = (float)d_arr[nrow * Cols + blockIdx.x * blockDim.x + i];
            arr_sign_cum_sum[i + 1] = arr_sign_cum_sum[i] + ts * ts;
            float tn = (float)d_norm[nrow * Cols + blockIdx.x * blockDim.x + i];
            arr_norm_cum_sum[i + 1] = arr_norm_cum_sum[i] + tn * tn;
        }
        for (int i = Num; i < N1 + blockDim.x + WndWidth; ++i)
        {
            arr_sign_cum_sum[i + 1] = -1000.;
            arr_norm_cum_sum[i + 1] = 1000.;
        }
    }
    else
    {
        for (int i = 0; i < blockDim.x + WndWidth; ++i)
        {
            float ts = (float)d_arr[nrow * Cols + blockIdx.x * blockDim.x + i];
            arr_sign_cum_sum[i + 1] = arr_sign_cum_sum[i] + ts * ts;
            float tn = (float)d_norm[nrow * Cols + blockIdx.x * blockDim.x + i];
            arr_norm_cum_sum[i + 1] = arr_norm_cum_sum[i] + tn * tn + 0.000001;
        }
    }

    //------- computations -----------------------------

    for (int iw = 0; iw < WndWidth; ++iw)
    {
        arr_buff[tid] = sqrtf((arr_sign_cum_sum[tid + 1 + iw] - arr_sign_cum_sum[tid]) / (arr_norm_cum_sum[tid + 1 + iw] - arr_norm_cum_sum[tid]));
        arr_nums[tid] = nrow * Cols + blockIdx.x * blockDim.x + tid;
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
            if (arr_buff[0] > (*pmax))
            {
                *pmax = arr_buff[0];
                *pnum = arr_nums[0];
                *pwidth = iw + 1;

            }


        }
        __syncthreads();

    }

    if (tid == 0)
    {
        d_pAuxArray[blockIdx.y * gridDim.x + blockIdx.x] = *pmax;
        d_pAuxIntArray[blockIdx.y * gridDim.x + blockIdx.x] = *pnum;
        d_pWidthArray[blockIdx.y * gridDim.x + blockIdx.x] = *pwidth;

    }
    __syncthreads();
//------- second part -------------------------------------
    if (0 == blockIdx.x)
    {
        float* arr = (float*)buff;
        int* nums = (int*)(arr + blockDim.x);


        float val_loc = -1000.;
        int num_loc = -1;

        const int len = gridDim.x * gridDim.y;
        for (int i = threadIdx.x; i < len; i += blockDim.x)
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
            pstructOut->snr = arr[0];
            pstructOut->irow = d_pAuxIntArray[nums[0]] / Cols;
            pstructOut->icol = d_pAuxIntArray[nums[0]] % Cols;
            pstructOut->iwidth = d_pWidthArray[nums[0]];
        }
        __syncthreads();
    }

}
//-------------------------------------------------
//__global__
//void complite_detection_kernel(const int Cols, const int LEnGrid, float* d_pAuxArray, int* d_pAuxNumArray
//    , int* d_pWidthArray, structOutDetection* pstructOut)
//{
//
//    extern __shared__ char buff[];
//    float* arr = (float*)buff;
//    int* nums = (int*)(arr + blockDim.x);
//
//
//    float val_loc = -1000.;
//    int num_loc = -1;
//
//
//    for (int i = threadIdx.x; i < LEnGrid; i += blockDim.x)
//    {
//        if (d_pAuxArray[i] > val_loc)
//        {
//            val_loc = d_pAuxArray[i];
//            num_loc = i;
//        }
//    }
//    arr[threadIdx.x] = val_loc;
//    nums[threadIdx.x] = num_loc;
//    __syncthreads();
//
//
//    for (unsigned int s = (blockDim.x) / 2; s > 0; s >>= 1)
//    {
//        if (threadIdx.x < s)
//        {
//            if (arr[threadIdx.x] < arr[threadIdx.x + s])
//            {
//                arr[threadIdx.x] = arr[threadIdx.x + s];
//                nums[threadIdx.x] = nums[threadIdx.x + s];
//            }
//        }
//        __syncthreads();
//    }
//    if (threadIdx.x == 0)
//    {
//        pstructOut->snr = arr[0];
//        pstructOut->irow = d_pAuxNumArray[nums[0]] / Cols;
//        pstructOut->icol = d_pAuxNumArray[nums[0]] % Cols;
//        pstructOut->iwidth = d_pWidthArray[nums[0]];
//    }
//    __syncthreads();
//
//}
//-------------------------------------------------
// Function to perform block-level reduction using shared memory
//template <unsigned int blockSize>
__global__ void blockMaxArgMax(int* data, int* result, int* argMax)
{
    extern __shared__ int sharedData[];

    int tid = threadIdx.x;
    int laneId = tid & (warpSize - 1); // Lane ID within the warp
    int warpId = tid / warpSize; // Warp ID


    // Each warp performs a warp-level reduction
    float myValue = data[blockIdx.x * blockDim.x + threadIdx.x];
    int myIdx = blockIdx.x * blockDim.x + tid;
    int  maxIdx;
    float maxVal;
    warpMaxArgmax(myValue, myIdx, &maxVal, &maxIdx);
    // int maxVal = warpMax(myValue);

    // Store warp-level maximum in shared memory
    if (laneId == 0) {
        sharedData[tid / warpSize] = maxVal;
        sharedData[blockDim.x / warpSize + warpId] = maxIdx;
    }
    __syncthreads();

    // Perform block-level reduction using shared memory
    if (tid < warpSize)
    {
        int blockMaxVal = (tid < (blockDim.x / warpSize)) ? sharedData[tid] : INT_MIN;
        int blockMaxIdx = (tid < (blockDim.x / warpSize)) ? sharedData[blockDim.x / warpSize + tid] : -1;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            //blockMaxVal = max(blockMaxVal, __shfl_down_sync(0xffffffff, blockMaxVal, offset));
            int tempVal = __shfl_down_sync(0xffffffff, blockMaxVal, offset);
            int tempIdx = __shfl_down_sync(0xffffffff, blockMaxIdx, offset);
            if (tempVal > blockMaxVal)
            {
                blockMaxVal = tempVal;
                blockMaxIdx = tempIdx;
            }
        }
        if (tid == 0)
        {
            result[blockIdx.x] = blockMaxVal;
            argMax[blockIdx.x] = blockMaxIdx;
            //printf("blockIdx.x = %d ,max = %d\n", blockIdx.x, blockMaxVal);
        }
    }
    __syncthreads();
}
__global__
void reduction_blocks_kernel(const int Cols, const int LEnGrid, float* data, int* d_pIndArrayInp
    , int* d_pWidthArrayInp, float* result, int* d_pIndArrayOut
    , int* d_pWidthArrayOut, structOutDetection* pstructOut)
{
    extern __shared__ float sharedData_[];
     
    float* sharedDataf = sharedData_;
    int* sharedInd = (int*)(sharedDataf + blockDim.x / warpSize);
    int tid = threadIdx.x;
    int laneId = tid & (warpSize - 1); // Lane ID within the warp
    int warpId = tid / warpSize; // Warp ID


    // Each warp performs a warp-level reduction
    float myValue = data[blockIdx.x * blockDim.x + threadIdx.x];
    int myIdx = blockIdx.x * blockDim.x + tid;
    int  maxIdx;
    float maxVal;
    
    warpMaxArgmax(myValue, myIdx, &maxVal, &maxIdx);   
    
    // Store warp-level maximum in shared memory
    if (laneId == 0)
    {
       sharedDataf[tid / warpSize] = maxVal;
       sharedInd[ warpId] = maxIdx;
    }
    __syncthreads();
    
   
    // Perform block-level reduction using shared memory
    if (tid < warpSize)
    {
        float blockMaxVal = (tid < (blockDim.x / warpSize)) ? sharedDataf[tid] : FLT_MIN;
        int blockMaxIdx = (tid < (blockDim.x / warpSize)) ? sharedInd[tid] : -1;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            //blockMaxVal = max(blockMaxVal, __shfl_down_sync(0xffffffff, blockMaxVal, offset));
            float tempVal = __shfl_down_sync(0xffffffff, blockMaxVal, offset);
            int tempIdx = __shfl_down_sync(0xffffffff, blockMaxIdx, offset);
            if (tempVal > blockMaxVal)
            {
                blockMaxVal = tempVal;
                blockMaxIdx = tempIdx;
            }
        }
        if (tid == 0)
        {
            result[blockIdx.x] = blockMaxVal;
            d_pIndArrayOut[blockIdx.x] = d_pIndArrayInp[blockMaxIdx];
            d_pWidthArrayOut[blockIdx.x] = d_pIndArrayInp[blockMaxIdx];
            if (gridDim.x == 1)
            {
                pstructOut->snr = blockMaxVal;
                pstructOut->irow = d_pIndArrayOut[blockIdx.x] / Cols;
                pstructOut->icol = d_pIndArrayOut[blockIdx.x] % Cols;
                pstructOut->iwidth = d_pWidthArrayOut[blockIdx.x];
            }
            /*blockMaxVal = 10.;
            printf("blockIdx.x = %d ,max = %f\n", blockIdx.x, blockMaxVal);*/
        }
    }
    __syncthreads();
   

}
 //----------------------------------------------------
 bool doDetectSignal_v1(fdmt_type_* d_arr, fdmt_type_* d_norm, const int Rows
     , const int  Cols, const int WndWidth, const float valTresh, const dim3 gridSize, const dim3 blockSize, size_t sz
     , float* d_pAuxArray, int* d_pAuxNumArray, int* d_pWidthArray, structOutDetection* pstructOut)
 {

     multi_windowing_kernel_v1 << < gridSize, blockSize, sz >> > (d_arr, d_norm, Rows
         , Cols, WndWidth, d_pAuxArray, d_pAuxNumArray, d_pWidthArray);
     cudaDeviceSynchronize();

     int block_size = 1024;
     size_t sz0 = block_size * (sizeof(int) + sizeof(float));
     complete_detection_kernel << < 1, block_size, sz0 >> > (Cols, gridSize.x * gridSize.y, d_pAuxArray, d_pAuxNumArray
         , d_pWidthArray, pstructOut);
     cudaDeviceSynchronize();

     if (pstructOut->snr >= valTresh)
     {
         return true;
     }

     return false;
 }
 //----------------------------------------------------
 bool doDetectSignal_v2(fdmt_type_* d_arr, fdmt_type_* d_norm, const int Rows
     , const int  Cols, const int WndWidth, const float valTresh, const dim3 gridSize, const dim3 blockSize, size_t sz
     , float* d_pAuxArray, int* d_pAuxNumArray, int* d_pWidthArray, structOutDetection* pstructOut)
 {
     size_t sz1 = (blockSize.x + WndWidth + 1) * 2 * sizeof(float) + (sizeof(float) + sizeof(int)) * blockSize.x + 2 * sizeof(float) + sizeof(int);
     multi_windowing_kernel_v2 << < gridSize, blockSize, sz1 >> > (d_arr, d_norm, Rows
         , Cols, WndWidth, d_pAuxArray, d_pAuxNumArray, d_pWidthArray);
     cudaDeviceSynchronize();

     int block_size = 1024;
     size_t sz0 = block_size * (sizeof(int) + sizeof(float));
     complete_detection_kernel << < 1, block_size, sz0 >> > (Cols, gridSize.x * gridSize.y, d_pAuxArray, d_pAuxNumArray
         , d_pWidthArray, pstructOut);
     cudaDeviceSynchronize();

     if (pstructOut->snr >= valTresh)
     {
         return true;
     }

     return false;
 }

 
 //----------------------------------------------------
 bool doDetectSignal_v2_2(fdmt_type_* d_arr, fdmt_type_* d_norm, const int Rows
     , const int  Cols, const int WndWidth, const float valTresh, const dim3 gridSize, const dim3 blockSize, size_t sz
     , float* d_pAuxArray, int* d_pAuxNumArray, int* d_pWidthArray, structOutDetection* pstructOut,char *arrBuff)
 {
     size_t sz1 = (sizeof(float) + sizeof(int)) * blockSize.x + 2 * sizeof(int) + sizeof(float);
     multi_windowing_kernel << < gridSize, blockSize, sz1 >> > (d_arr, d_norm
         , Cols, WndWidth, d_pAuxArray, d_pAuxNumArray, d_pWidthArray);
     
     cudaDeviceSynchronize();

     /*int block_size = 256;
     size_t sz0 = block_size * (sizeof(int) + sizeof(float));
     complite_detection_kernel << < 1, block_size, sz0 >> > (Cols, gridSize.x * gridSize.y, d_pAuxArray, d_pAuxNumArray
         , d_pWidthArray, pstructOut);
     cudaDeviceSynchronize();*/
     int blocksize = gridSize.x * gridSize.y;
     int gridsize = 1;
     float* result = (float*)arrBuff;
     int* d_pIndArrayOut = (int*)(result + gridsize);
     int* d_pWidthArrayOut = d_pIndArrayOut + gridsize;

     reduction_blocks_kernel<<< gridsize, blocksize, blocksize / warpSize * (sizeof(int) +sizeof(float)) >> >
         (Cols, gridSize.x * gridSize.y, d_pAuxArray, d_pAuxNumArray, d_pWidthArray
         , result, d_pIndArrayOut, d_pWidthArrayOut, pstructOut);
     cudaDeviceSynchronize();
     if (pstructOut->snr >= valTresh)
     {
         return true;
     }

     return false;
 }

int main()
{
    // 1 Create arrays  and stuff them on GPU
    const int rows =  256;
    const int cols =  (1 << 20)/rows;
    fdmt_type_* d_arr = 0;
    fdmt_type_* d_norm = 0;
    cudaMalloc((void**)&d_arr, rows * cols * sizeof(fdmt_type_));
    cudaMalloc((void**)&d_norm, rows * cols * sizeof(fdmt_type_));

    float* arr = (float*)malloc(rows * cols * sizeof(fdmt_type_));
    float* norm = (float*)malloc(rows * cols * sizeof(fdmt_type_));
    for (int i = 0; i < rows * cols; ++i)
    {
        arr[i] = 1;
        norm[i] = 1.;
    }
    arr[3] = 20;

    arr[12] = 40;
    arr[8] = 0;
    arr[9] = 0;
    arr[11] = 0;
    arr[13] = 0;
    arr[14] = 0;
    arr[15] = 0;
    arr[16] = 0;
    arr[17] = 0;
    for (int i = 3; i < 5; ++i)
    {
        norm[cols + i] = 0.001;
    }

    cudaMemcpy(d_arr, arr, rows * cols * sizeof(fdmt_type_), cudaMemcpyHostToDevice);
    cudaMemcpy(d_norm, norm, rows * cols * sizeof(fdmt_type_), cudaMemcpyHostToDevice);
    // !1

    // 2.max length of window
    const int WndWidth = 10;
    //!2
    
    // 3. treshold
    float valTresh = 5.;
    // !3

    // 4. output struct
    structOutDetection* pstructOut = NULL;// (structOutDetection*)malloc(sizeof(structOutDetection));
    cudaMallocManaged((void**)&pstructOut, sizeof(structOutDetection));

    const dim3 blockSize = dim3(256, 1, 1);
    const dim3 gridSize = dim3((cols + blockSize.x - 1) / blockSize.x, rows, 1);
    size_t sz = blockSize.x * (sizeof(int) + 2* sizeof(float)) + sizeof(float) + 2*sizeof(int);

    float* d_pAuxArray = 0;
    cudaMalloc((void**)&d_pAuxArray, rows * gridSize.x  * sizeof(float));
    int* d_pAuxIntArray = 0;
    cudaMalloc((void**)&d_pAuxIntArray, rows * gridSize.x * sizeof(int));
    int* d_pAuxWidthArray = 0;
    cudaMalloc((void**)&d_pAuxWidthArray, rows * gridSize.x * sizeof(int));

    int n_repeat = 5;

    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    ///////////  1 var  //////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_repeat; ++i)
    {
        if (doDetectSignal_v1(d_arr, d_norm, rows
            , cols, WndWidth, valTresh, gridSize, blockSize, sz
            , d_pAuxArray, d_pAuxIntArray, d_pAuxWidthArray, pstructOut))

        {
            /*std::cout << "detection" << std::endl;
            std::cout << "row = " << pstructOut->irow << std::endl;
            std::cout << "col = " << pstructOut->icol << std::endl;
            std::cout << "width = " << pstructOut->iwidth << std::endl;*/
            //std::cout << "snr = " << pstructOut->snr << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "doDetectSignal_v1 time:                     " << duration.count() / ((float)n_repeat) << " microseconds" << std::endl;
    std::cout << "detection" << std::endl;
            std::cout << "row = " << pstructOut->irow << std::endl;
            std::cout << "col = " << pstructOut->icol << std::endl;
            std::cout << "width = " << pstructOut->iwidth << std::endl;
            std::cout << "snr = " << pstructOut->snr << std::endl;

            /////////////////////////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////////////
        ///////////  2 var   //////////////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////////////

            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < n_repeat; ++i)
            {
                if (doDetectSignal_v2(d_arr, d_norm, rows
                    , cols, WndWidth, valTresh, gridSize, blockSize, 2 * sz
                    , d_pAuxArray, d_pAuxIntArray, d_pAuxWidthArray, pstructOut))

                {
                    /*std::cout << "detection" << std::endl;
                    std::cout << "row = " << pstructOut->irow << std::endl;
                    std::cout << "col = " << pstructOut->icol << std::endl;
                    std::cout << "width = " << pstructOut->iwidth << std::endl;
                    std::cout << "snr = " << pstructOut->snr << std::endl;*/
                }
            }
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            std::cout << "doDetectSignal_v2 time:                     " << duration.count() / ((float)n_repeat) << " microseconds" << std::endl;
            std::cout << "detection" << std::endl;
            std::cout << "row = " << pstructOut->irow << std::endl;
            std::cout << "col = " << pstructOut->icol << std::endl;
            std::cout << "width = " << pstructOut->iwidth << std::endl;
            std::cout << "snr = " << pstructOut->snr << std::endl;
            
            /////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
///////////  2_2 var   //////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
            char* d_pBuff = 0;
            cudaMalloc((void**)&d_pBuff, gridSize.x * sizeof(float));
            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < n_repeat; ++i)
            {
                if (doDetectSignal_v2_2(d_arr, d_norm, rows
                    , cols, WndWidth, valTresh, gridSize, blockSize, 2 * sz
                    , d_pAuxArray, d_pAuxIntArray, d_pAuxWidthArray, pstructOut, d_pBuff))

                {
                    /*std::cout << "detection" << std::endl;
                    std::cout << "row = " << pstructOut->irow << std::endl;
                    std::cout << "col = " << pstructOut->icol << std::endl;
                    std::cout << "width = " << pstructOut->iwidth << std::endl;
                    std::cout << "snr = " << pstructOut->snr << std::endl;*/
                }
            }
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            std::cout << "doDetectSignal_v2_2 time:                     " << duration.count() / ((float)n_repeat) << " microseconds" << std::endl;
            std::cout << "detection" << std::endl;
            std::cout << "row = " << pstructOut->irow << std::endl;
            std::cout << "col = " << pstructOut->icol << std::endl;
            std::cout << "width = " << pstructOut->iwidth << std::endl;
            std::cout << "snr = " << pstructOut->snr << std::endl;

            /////////////////////////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////////////
        ///////////  3 var   //////////////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////////////

            start = std::chrono::high_resolution_clock::now();
            /*for (int i = 0; i < n_repeat; ++i)
            {
                multi_windowing_kernel_v3 << < gridSize, blockSize, sz >> > (d_arr, d_norm, rows
                    , cols, WndWidth, d_pAuxArray, d_pAuxIntArray
                    , d_pAuxWidthArray, pstructOut);
                cudaDeviceSynchronize();
            }*/
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            std::cout << "multi_windowing_kernel_v3 time:                     " << duration.count() / ((float)n_repeat) << " microseconds" << std::endl;
            std::cout << "detection" << std::endl;
            std::cout << "row = " << pstructOut->irow << std::endl;
            std::cout << "col = " << pstructOut->icol << std::endl;
            std::cout << "width = " << pstructOut->iwidth << std::endl;
            std::cout << "snr = " << pstructOut->snr << std::endl;
            int iii = 0;
            if (pstructOut->snr >= valTresh)

            {
                /*std::cout << "detection" << std::endl;
                std::cout << "row = " << pstructOut->irow << std::endl;
                std::cout << "col = " << pstructOut->icol << std::endl;
                std::cout << "width = " << pstructOut->iwidth << std::endl;
                std::cout << "snr = " << pstructOut->snr << std::endl;*/
            }
            /////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
///////////  detect_signal_gpu var   //////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
            const dim3 blockSize0(256, 1, 1);
            const dim3 gridSize0((cols + blockSize0.x - 1) / blockSize0.x, rows, 1);
            start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < n_repeat; ++i)
            {
                detect_signal_gpu( d_arr,  d_norm, rows
                    , cols, WndWidth, gridSize0, blockSize0, d_pAuxArray, d_pAuxIntArray
                    , d_pAuxWidthArray, pstructOut);
                
                cudaDeviceSynchronize();
            }
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            std::cout << "detect_signal_gpu time:                     " << duration.count() / ((float)n_repeat) << " microseconds" << std::endl;
            std::cout << "detection" << std::endl;
            std::cout << "row = " << pstructOut->irow << std::endl;
            std::cout << "col = " << pstructOut->icol << std::endl;
            std::cout << "width = " << pstructOut->iwidth << std::endl;
            std::cout << "snr = " << pstructOut->snr << std::endl;
            
            if (pstructOut->snr >= valTresh)

            {
                /*std::cout << "detection" << std::endl;
                std::cout << "row = " << pstructOut->irow << std::endl;
                std::cout << "col = " << pstructOut->icol << std::endl;
                std::cout << "width = " << pstructOut->iwidth << std::endl;
                std::cout << "snr = " << pstructOut->snr << std::endl;*/
            }
    //cudaDeviceSynchronize();

    cudaFree(pstructOut);
    cudaFree(d_arr);
    cudaFree(d_norm);
    cudaFree(d_pAuxArray);
    cudaFree(d_pAuxIntArray);
    cudaFree(d_pAuxWidthArray);
    cudaFree(pstructOut);
    free(arr);
    free(norm);
    cudaFree(d_pBuff);
   
    return 0;
}

