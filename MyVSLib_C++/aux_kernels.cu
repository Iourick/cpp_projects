#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <math_functions.h>
#include "aux_kernels.cuh"

//-----------------------------------------------------------------
__global__ void calcRowMeanAndDisp(float* d_arrIm, int nRows, int nCols
    , float* arrSumMean, float* arrDisps)
{
    extern __shared__ float sbuff[];
    float* sdata = sbuff;  

    float* d_arr = d_arrIm + nCols * blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;
    if (tid >= nCols)
    {
        return;
    }

    float localSum = 0.0f;
    float localSquaredSum = 0.0f;
    
    // Calculate partial sums within each block   
    while (i < nCols)
    {
        localSum += d_arr[i];
        localSquaredSum += d_arr[i] * d_arr[i];
        i += blockDim.x;
        //++numLocal;
    }

    // Store partial sums in shared memory
    //numLocal = len / blockDim.x;
    //sNums[tid] = numLocal;
    sdata[tid] = localSum;// / numLocal;
    sdata[blockDim.x + tid] = localSquaredSum;// / numLocal;

    __syncthreads();

    // Parallel reduction within the block to sum partial sums
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if ((tid < s) && (tid < nCols))
        {
            sdata[tid] = sdata[tid] + sdata[tid + s];
            sdata[blockDim.x + tid] = sdata[blockDim.x + tid] + sdata[blockDim.x + tid + s];
        }
        __syncthreads();
    }

    // Only thread 0 within each block computes the block's sum
    if (tid == 0)
    {
        arrSumMean[blockIdx.x] = sdata[0] / ((float)nCols);
        arrDisps[blockIdx.x] = sdata[blockDim.x] / ((float)nCols) - sdata[0] / ((float)nCols) * sdata[0] / ((float)nCols);
        
    }
    __syncthreads();

}

//-----------------------------------------------------------------
__global__ void kernel_OneSM_Mean_and_Std(float* d_arrMeans, float* d_arrDisps, int len
    , float* pmean0, float *pstd)
{
    extern __shared__ float sbuff[];
    float* sdata = sbuff;
    //int* sNums = (int*)((char*)sbuff + 2 * blockDim.x * sizeof(float));


    unsigned int tid = threadIdx.x;
    unsigned int i = tid;
    if (tid >= len)
    {
        return;
    }

    float localSum0 = 0.0f;
    float localSum1 = 0.0f;
    //int numLocal = 0;
    // Calculate partial sums within each block   
    while (i < len)
    {
        localSum0 += d_arrMeans[i];
        localSum1 += d_arrDisps[i] + d_arrMeans[i] * d_arrMeans[i];
        i += blockDim.x;
        //++numLocal;
    }

    // Store partial sums in shared memory
    //numLocal = len / blockDim.x;
    //sNums[tid] = numLocal;
    sdata[tid] = localSum0;// / numLocal;
    sdata[blockDim.x + tid] = localSum1;// / numLocal;

    __syncthreads();

    // Parallel reduction within the block to sum partial sums
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if ((tid < s) && (tid < len))
        {
            sdata[tid] = sdata[tid] + sdata[tid + s];// (sdata[tid] * sNums[tid] + sdata[tid + s] * sNums[tid + s]) / (sNums[tid] + sNums[tid + s]);
            sdata[blockDim.x + tid] = sdata[blockDim.x + tid] + sdata[blockDim.x + tid + s];// (sdata[blockDim.x + tid] * sNums[tid] + sdata[blockDim.x + tid + s] * sNums[tid + s])
               // / (sNums[tid] + sNums[tid + s]);
            //sNums[tid] = sNums[tid] + sNums[tid + s];
        }
        __syncthreads();
    }

    // Only thread 0 within each block computes the block's sum
    if (tid == 0)
    {
        *pmean0 = sdata[0]/ ((float)len);
        *pstd = sqrtf(sdata[blockDim.x] / ((float)len) - sdata[0] / ((float)len) * sdata[0] / ((float)len));

    }
    __syncthreads();
}
//--------------------------------------------------------------
__global__ void kernel_normalize_array(fdmt_type_* pAuxBuff, const unsigned int len
    , float* pmean, float* pdev, float* parrInp)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len)
    {
        return;
    }
    pAuxBuff[i] = (fdmt_type_)((parrInp[i] - (*pmean)) / ((*pdev) * 0.25));

}
//-----------------------------------------------------------------
__global__ void kernel_calcDispersion_for_Short_Onedimentional_Array(float* d_poutDisp, const float* d_arrInp, const float* d_mean, const int len)
    
{
    extern __shared__ float sbuff[];
    float* sdata = sbuff;
    int* sNums = (int*)((char*)sbuff +  blockDim.x * sizeof(float));


    unsigned int tid = threadIdx.x;
    unsigned int i = tid;// blockIdx.x* blockDim.x + threadIdx.x;
    if (tid >= len)
    {
        return;
    }

    float localSum0 = 0.0f;    
    int numLocal = 0;
    // Calculate partial sums within each block   
    while (i < len)
    {
        localSum0 += (d_arrInp[i] - (*d_mean)) * (d_arrInp[i] - (*d_mean));        
        i += blockDim.x;
        ++numLocal;
    }

    // Store partial sums in shared memory
    //numLocal = len / blockDim.x;
    sNums[tid] = numLocal;
    sdata[tid] = localSum0 / numLocal;
    

    __syncthreads();

    // Parallel reduction within the block to sum partial sums
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if ((tid < s) && (tid < len))
        {
            sdata[tid] = (sdata[tid] * sNums[tid] + sdata[tid + s] * sNums[tid + s]) / (sNums[tid] + sNums[tid + s]);            
            sNums[tid] = sNums[tid] + sNums[tid + s];
        }
        __syncthreads();
    }

    // Only thread 0 within each block computes the block's sum
    if (tid == 0)
    {
        *d_poutDisp = sdata[0];        

    }
    __syncthreads();
}
//------------------------------------------------------------------------------
__global__
void calculateMeanAndSTD_for_oneDimArray_kernel(float* d_arr, const unsigned int len, float* pmean, float* pstd)
{

    extern __shared__ float sbuff[];
    float* sdata = sbuff;

    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x;
    if (i >= len)
    {
        return;
    }

    float localSum = 0.0f;
    float localSquaredSum = 0.0f;

    // Calculate partial sums within each block

    while (i < len)
    {
        localSum += d_arr[i];
        localSquaredSum += d_arr[i] * d_arr[i];
        i += blockDim.x;

    }

    // Store partial sums in shared memory    
    sdata[tid] = localSum;
    sdata[blockDim.x + tid] = localSquaredSum;

    __syncthreads();

    // Parallel reduction within the block to sum partial sums
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if ((tid < s) && (tid < len))
        {
            sdata[tid] = sdata[tid] + sdata[tid + s];// (sdata[tid] * sNums[tid] + sdata[tid + s] * sNums[tid + s]) / (sNums[tid] + sNums[tid + s]);
            sdata[blockDim.x + tid] = sdata[blockDim.x + tid] + sdata[blockDim.x + tid + s];// (sdata[blockDim.x + tid] * sNums[tid] + sdata[blockDim.x + tid + s] * sNums[tid + s])
        }
        __syncthreads();
    }

    // Only thread 0 within each block computes the block's sum
    if (tid == 0)
    {
        *pmean = sdata[0] / ((float)len);
        *pstd = sqrtf(sdata[blockDim.x] / ((float)len) - (*pmean) * (*pmean));

    }
    __syncthreads();

}


