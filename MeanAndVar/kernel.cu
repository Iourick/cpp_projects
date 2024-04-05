
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <cmath>
#include <random>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846 // Define M_PI if not already defined
#endif

#define BLOCK_SIZE 1024 //256

void generateGaussian(float* arr, int len, float mean, float variance) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f); // Gaussian distribution with mean 0 and variance 1

    for (int i = 0; i < len; ++i) {
        //float u1 = dist(gen); // First random number
        //float u2 = dist(gen); // Second random number

        //// Box-Muller transform to generate Gaussian numbers
        //float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2 * M_PI * u2);
        float u1 = dist(gen); // First random number
        // Adjust generated number to desired mean and variance
        arr[i] = u1 * sqrtf(variance) + mean;
    }
}
// This function was taken in openGL and doesn't work properly.
__global__ void calculateMeanAndVariance(float* d_arr, float* mean, float* variance, unsigned int len)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float localSum = 0.0f;
    float localSquaredDiff = 0.0f;

    // Calculate partial sums within each block
    while (i < len)
    {
        localSum += d_arr[i];
        i += blockDim.x /** gridDim.x*/;
    }

    // Store partial sums in shared memory
    sdata[tid] = localSum;
    __syncthreads();

    // Parallel reduction within the block to sum partial sums
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Only thread 0 within each block computes the block's sum
    if (tid == 0) {
        //atomicAdd(mean, sdata[0]);
        *mean = sdata[0];
    }
    __syncthreads();

    // Calculate variance (dispersion)
    while (i < len) 
    {
        float diff = d_arr[i] - *mean / len;  // Calculate difference from mean
        localSquaredDiff += diff * diff;
        i += blockDim.x /** gridDim.x*/;
    }

    // Store partial squared differences in shared memory
    sdata[tid] = localSquaredDiff;
    __syncthreads();

    // Parallel reduction within the block to sum squared differences
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Only thread 0 within each block computes the block's sum of squared differences
    if (tid == 0)
    {
        //atomicAdd(variance, sdata[0]);
        *variance = sdata[0];
    }
}

//--------------------------------------------------
__global__ void calculateMeanAndVariance_(float* d_arr, float* sum, float* sumSquared, unsigned int len)
{
    extern __shared__ float sdata[];
    //extern __shared__ float sSquareddata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float localSum = 0.0f;
    float localSquaredSum = 0.0f;

    // Calculate partial sums within each block
    while (i < len)
    {
        localSum += d_arr[i];
        localSquaredSum += d_arr[i] * d_arr[i];
        i += blockDim.x * gridDim.x;
    }

    // Store partial sums in shared memory
    sdata[tid] = localSum;
    sdata[blockDim.x + tid] = localSquaredSum;
    __syncthreads();

    // Parallel reduction within the block to sum partial sums
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
        }
        __syncthreads();
    }

    // Only thread 0 within each block computes the block's sum
    if (tid == 0)
    {
        *sum = sdata[0];
        *sumSquared = sdata[blockDim.x];
        //atomicAdd(sum, sdata[0]);
       // atomicAdd(sumSquared, sdata[blockDim.x]);
    }
    __syncthreads();
    
}

//--------------------------------------------------
__global__ void calculateMeanAndVariance__(float* d_arr, float* mean, float* meanSquared, unsigned int len)
{
    /*extern __shared__ float sdata[];
    extern __shared__ int sNums[];*/
    extern __shared__ float sbuff[];
    float* sdata = sbuff;
    int* sNums = (int*)((char*)sbuff + 2 * blockDim.x * sizeof(float));
    //((char*)sbuff + 2 * blockDim.x * sizeof(float))

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len)
    {
        return;
    }

    float localSum = 0.0f;
    float localSquaredSum = 0.0f;

    // Calculate partial sums within each block
    int numLocal = 0;
    while (i < len)
    {
        localSum += d_arr[i];
        localSquaredSum += d_arr[i] * d_arr[i];
        i += blockDim.x * gridDim.x;
        ++numLocal;
    }

    // Store partial sums in shared memory
    //numLocal = len / blockDim.x;
    sNums[tid] = numLocal;
    sdata[tid] = localSum / numLocal;
    sdata[blockDim.x + tid] = localSquaredSum / numLocal;

    __syncthreads();

    // Parallel reduction within the block to sum partial sums
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if ((tid < s)&& (tid < len))
        {
            sdata[tid] = (sdata[tid] * sNums[tid] + sdata[tid + s] * sNums[tid + s]) / (sNums[tid] + sNums[tid + s]);
            sdata[blockDim.x + tid] = (sdata[blockDim.x + tid] * sNums[tid] + sdata[blockDim.x + tid + s] * sNums[tid + s])
                / (sNums[tid] + sNums[tid + s]);
            sNums[tid] = sNums[tid] + sNums[tid + s];

            /*sdata[tid] = (sdata[tid]  + sdata[tid + s])/2.;
            sdata[blockDim.x + tid] = (sdata[blockDim.x + tid] + sdata[blockDim.x + tid + s]) / 2.;*/



        }
        __syncthreads();
    }

    // Only thread 0 within each block computes the block's sum
    if (tid == 0) 
    {
        *mean = sdata[0];
        *meanSquared = sdata[blockDim.x];
        /**mean = 0.;
        atomicAdd();
        *meanSquared = 0.;
        atomicAdd(meanSquared, sdata[blockDim.x]);*/
    }
    __syncthreads();

}

__global__ void calcPartialSums(float* input, float* output, float* output2, int size)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    

    // Each thread loads one element from global memory to shared memory
    if (i < size) {
        sdata[tid] = input[i];
        sdata[blockDim.x + tid] = input[i] * input[i];
    }
    else {
        sdata[tid] = 0.0f;
        sdata[blockDim.x + tid] = 0.0f;
    }

    __syncthreads();

    // Perform parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0)
    {
        int itemp = blockDim.x;
        if (blockIdx.x == (gridDim.x - 1))
        {
            int j = size % blockDim.x;
            if (j != 0)
            {
                itemp = j;
            }

        }
        atomicAdd(&output[blockIdx.x], sdata[0] / itemp);
        atomicAdd(&output2[blockIdx.x], sdata[blockDim.x] / itemp);
         
    }
    __syncthreads();
}


__global__ void utilizePartialSums(float* input, float* input2,  int len, float weightLast, float*mean, float * meanSquared)
{
    extern __shared__ float sbuff[];
    float* sdata = sbuff;
    float* sWeights = (float*)((char*)sbuff + 2 * blockDim.x * sizeof(float));

    //((char*)sbuff + 2 * blockDim.x * sizeof(float))

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float localSum = 0.0f;
    float localSum2 = 0.0f;

    // Calculate partial sums within each block
    float weightLocal = 0.;
    while (i < len)
    {
        localSum += input[i];
        localSum2 += input2[i];
        i += blockDim.x * gridDim.x;
        if (i == (len - 1))
        {
            weightLocal += weightLast;
        }
        else
        {
            weightLocal += 1.;
        }
        
    }

    // Store partial sums in shared memory
    //numLocal = len / blockDim.x;
    sWeights[tid] = weightLocal;
    sdata[tid] = localSum / weightLocal;
    sdata[blockDim.x + tid] = localSum2 / weightLocal;

    __syncthreads();

    // Parallel reduction within the block to sum partial sums
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = (sdata[tid] * sWeights[tid] + sdata[tid + s] * sWeights[tid + s]) / (sWeights[tid] + sWeights[tid + s]);
            sdata[blockDim.x + tid] = (sdata[blockDim.x + tid] * sWeights[tid] + sdata[blockDim.x + tid + s] * sWeights[tid + s])
                / (sWeights[tid] + sWeights[tid + s]);
            sWeights[tid] = sWeights[tid] + sWeights[tid + s];

            /*sdata[tid] = (sdata[tid]  + sdata[tid + s])/2.;
            sdata[blockDim.x + tid] = (sdata[blockDim.x + tid] + sdata[blockDim.x + tid + s]) / 2.;*/



        }
        __syncthreads();
    }

    // Only thread 0 within each block computes the block's sum
    if (tid == 0) {
        atomicAdd(mean, sdata[0]);
        atomicAdd(meanSquared, sdata[blockDim.x]);
    }
    __syncthreads();

}
//-----------------------------------------------------------
void calculateMeanAndMeanSquared_by2sqans(float* d_arr, float* mean, float* meanSquared, unsigned int len)
{
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
    float* output = NULL;
    float* output2 = NULL;
    cudaMallocManaged(&output, blocksPerGrid * sizeof(float));
    cudaMallocManaged(&output2, blocksPerGrid * sizeof(float));

    calcPartialSums << < blocksPerGrid, threadsPerBlock, 2 * sizeof(float)* threadsPerBlock >> > (d_arr, output, output2, len);
    //float arr[1000] = { 0. };
    //cudaMemcpy(arr, output, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    float weightLast = 1.;
    int itemp = len % threadsPerBlock;
    if (itemp != 0)
    {
        weightLast = ((float)itemp) / ((float)threadsPerBlock);
    }
    const int threads = 256;
    utilizePartialSums<<<1, threadsPerBlock, 3 * sizeof(float) * threads>>>(output, output2, blocksPerGrid, weightLast, mean, meanSquared);

    cudaFree(output);
    cudaFree(output2);
}
//-----------------------------------------------------------------

//-----------------------------------------------------------------
__global__ void calcRowMeanAndDisp(float* d_arrIm, int nRows, int nCols
    , float* arrSumMean, float* arrSumMeanSquared)
{
    extern __shared__ float sbuff[];
    float* sdata = sbuff;
    int* sNums = (int*)((char*)sbuff + 2 * blockDim.x * sizeof(float));

    float* d_arr = d_arrIm + nCols * blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;// blockIdx.x* blockDim.x + threadIdx.x;
    if (tid >= nCols)
    {
        return;
    }

    float localSum = 0.0f;
    float localSquaredSum = 0.0f;
    int numLocal = 0;
    // Calculate partial sums within each block   
    while (i < nCols)
    {
        localSum += d_arr[i];
        localSquaredSum += d_arr[i] * d_arr[i];
        i += blockDim.x ;
        ++numLocal;
    }

    // Store partial sums in shared memory
    //numLocal = len / blockDim.x;
    sNums[tid] = numLocal;
    sdata[tid] = localSum / numLocal;
    sdata[blockDim.x + tid] = localSquaredSum / numLocal;

    __syncthreads();

    // Parallel reduction within the block to sum partial sums
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if ((tid < s) && (tid < nCols))
        {
            sdata[tid] = (sdata[tid] * sNums[tid] + sdata[tid + s] * sNums[tid + s]) / (sNums[tid] + sNums[tid + s]);
            sdata[blockDim.x + tid] = (sdata[blockDim.x + tid] * sNums[tid] + sdata[blockDim.x + tid + s] * sNums[tid + s])
                / (sNums[tid] + sNums[tid + s]);
            sNums[tid] = sNums[tid] + sNums[tid + s];
        }
        __syncthreads();
    }

    // Only thread 0 within each block computes the block's sum
    if (tid == 0)
    {
        arrSumMean[blockIdx.x] = sdata[0];
        arrSumMeanSquared[blockIdx.x] = sdata[blockDim.x] -sdata[0] * sdata[0];
        /*atomicAdd(&arrSumMean[blockIdx.x], sdata[0]);
        atomicAdd(&arrSumMeanSquared[blockIdx.x], sdata[blockDim.x]);*/
    }
    __syncthreads();    
    
}

//-----------------------------------------------------------------
__global__ void kernel_OneSM_Mean_and_Disp(float* d_arrMeans, float* d_arrDisps, int len
    , float* arrSumMean0, float* arrSumMean1)
{
    extern __shared__ float sbuff[];
    float* sdata = sbuff;
    int* sNums = (int*)((char*)sbuff + 2 * blockDim.x * sizeof(float));

    
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;// blockIdx.x* blockDim.x + threadIdx.x;
    if (tid >= len)
    {
        return;
    }

    float localSum0 = 0.0f;
    float localSum1 = 0.0f;
    int numLocal = 0;
    // Calculate partial sums within each block   
    while (i < len)
    {
        localSum0 += d_arrMeans[i];
        localSum1 += d_arrDisps[i] + d_arrMeans[i] * d_arrMeans[i];
        i += blockDim.x ;
        ++numLocal;
    }

    // Store partial sums in shared memory
    //numLocal = len / blockDim.x;
    sNums[tid] = numLocal;
    sdata[tid] = localSum0 / numLocal;
    sdata[blockDim.x + tid] = localSum1 / numLocal;

    __syncthreads();

    // Parallel reduction within the block to sum partial sums
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if ((tid < s) && (tid < len))
        {
            sdata[tid] = (sdata[tid] * sNums[tid] + sdata[tid + s] * sNums[tid + s]) / (sNums[tid] + sNums[tid + s]);
            sdata[blockDim.x + tid] = (sdata[blockDim.x + tid] * sNums[tid] + sdata[blockDim.x + tid + s] * sNums[tid + s])
                / (sNums[tid] + sNums[tid + s]);
            sNums[tid] = sNums[tid] + sNums[tid + s];         
        }
        __syncthreads();
    }

    // Only thread 0 within each block computes the block's sum
    if (tid == 0)
    {
        *arrSumMean0 = sdata[0];
        *arrSumMean1 = sdata[blockDim.x] -sdata[0] * sdata[0];
        
    }
    __syncthreads();
}


const int REDUCE_THREADS = 32;
//------------------------------------------------------
__global__ void meanAndDispersion(float* array,int ROWS, int COLS, float* result)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;

    // Calculate sum within each row
    for (int i = tid; i < ROWS * COLS; i += stride) {
        sum += array[i];
    }

    // Reduce sum across threads within a block
    extern __shared__ float blockSums[];
    blockSums[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            blockSums[threadIdx.x] += blockSums[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Store row-wise sums in result array
    if (threadIdx.x == 0) {
        result[blockIdx.x] = blockSums[0] / COLS;
    }

    // Calculate overall mean and dispersion
    if (tid == 0) {
        float overall_sum = 0.0f;
        for (int j = 0; j < gridDim.x; ++j) {
            overall_sum += result[j];
        }
        float overall_mean = overall_sum / ROWS;
        result[gridDim.x] = overall_mean;

        // Calculate dispersion (variance)
        float dispersion = 0.0f;
        for (int k = 0; k < ROWS * COLS; ++k) {
            float diff = array[k] - overall_mean;
            dispersion += diff * diff;
        }
        result[gridDim.x + 1] = dispersion / (ROWS * COLS);
    }
}

int main() 
{
    
   // const int arraySize = (1 << 20) - 331; // Set your array size
    const int nRows = 256;
    const int nCols = (1 << 19) / nRows;
    const int arraySize = nRows * nCols; // Set your array size
    const int threadsPerBlock = 512;// BLOCK_SIZE;
    const int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate and populate input array on the device
    float* d_arr;
    cudaMalloc((void**)&d_arr, arraySize * sizeof(float));

    // ... Copy data to d_arr ...
    float *arr  = (float*)malloc(arraySize * sizeof(float));
    //for (int i = 0; i < arraySize; ++i)
    //{  
    //    arr[i] = 1.;
    //    /*if (i < arraySize / 2)
    //    {
    //        arr[i] = 1.;
    //    }
    //    else
    //    {
    //        arr[i] = -1;
    //    }*/
    //}
    const float VAlMean = 1.;
    const float VAlDisp = 16.;
    generateGaussian(arr, arraySize, VAlMean, VAlDisp);
    cudaMemcpy(d_arr, arr, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    /////////////////////////////////////////////////////////
    ////////////   1   /////////////////////////////////////////////
    /////////////////////////////////////////////////////////
    // Allocate memory for mean and variance on the device
    float* d_mean, * d_variance;
    cudaMalloc((void**)&d_mean, sizeof(float));
    cudaMalloc((void**)&d_variance, sizeof(float));

    // Initialize mean and variance to zero on the device
    cudaMemset(d_mean, 0, sizeof(float));
    cudaMemset(d_variance, 0, sizeof(float));
    int num = 100;
    // Calculate mean and variance
    cudaMemset(d_mean, 0, sizeof(float));
    cudaMemset(d_variance, 0, sizeof(float));
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num; ++i)
    {
        
        calculateMeanAndVariance_ << <1, threadsPerBlock, 2 *threadsPerBlock * sizeof(float) >> > (d_arr, d_mean, d_variance, arraySize);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "1 Time : " << duration.count() / ((double)num) << " microseconds" << std::endl;
    // Copy mean and variance back to host
    float h_mean, h_variance;
    cudaMemcpy(&h_mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_variance, d_variance, sizeof(float), cudaMemcpyDeviceToHost);

    // Finalize mean and variance calculation
    h_mean /= arraySize;
    h_variance = h_variance/ arraySize - h_mean * h_mean;

    // Output the mean and variance
    std::cout << "Mean: " << h_mean << std::endl;
    std::cout << "Variance: " << h_variance << std::endl;
    /////////////////////////////////////////////////////////
    ////////////   2   /////////////////////////////////////////////
    /////////////////////////////////////////////////////////
     
    // Allocate memory for mean and variance on the device
    float* d_sum, * d_SquaredSum;
    cudaMalloc((void**)&d_sum, sizeof(float));
    cudaMalloc((void**)&d_SquaredSum, sizeof(float));
    // Initialize mean and variance to zero on the device
    cudaMemset(d_sum, 0, sizeof(float));
    cudaMemset(d_SquaredSum, 0, sizeof(float));
    

    start = std::chrono::high_resolution_clock::now();
    // Calculate mean and variance
    for (int i = 0; i < num; ++i)
    {
        
        calculateMeanAndVariance__ << <1, threadsPerBlock, threadsPerBlock* (2 * sizeof(float) + sizeof(int)) >> > (d_arr, d_sum, d_SquaredSum, arraySize);
        //cudaDeviceSynchronize();
    }

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "2 Time : " << duration.count() / ((double)num) << " microseconds" << std::endl;
    // Copy mean and variance back to host
    
    cudaMemcpy(&h_mean, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_variance, d_SquaredSum, sizeof(float), cudaMemcpyDeviceToHost);

    // Finalize mean and variance calculation
   // h_mean /= arraySize;
   // h_variance = h_variance/ arraySize - h_mean * h_mean;
    h_variance = h_variance  - h_mean * h_mean;

    // Output the mean and variance
    std::cout << "Mean: " << h_mean << std::endl;
    std::cout << "Variance: " << h_variance << std::endl;

    /////////////////////////////////////////////////////////
    ///////////   3   //// BAD CASE //////////////////////////////////////////
    /////////////////////////////////////////////////////////
   

   // // Initialize mean and variance to zero on the device
   // cudaMemset(d_sum, 0, sizeof(float));
   // cudaMemset(d_SquaredSum, 0, sizeof(float));

   // start = std::chrono::high_resolution_clock::now();
   // // Calculate mean and variance
   // for (int i = 0; i < num; ++i)
   // {
   //     cudaMemset(d_sum, 0, sizeof(float));
   //     cudaMemset(d_SquaredSum, 0, sizeof(float));
   //     // Calculate mean and variance
   //     calculateMeanAndMeanSquared_by2sqans(d_arr, d_sum, d_SquaredSum, arraySize);
   //     // Copy mean and variance back to host
   // }

   // end = std::chrono::high_resolution_clock::now();
   // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
   // std::cout << "3 Time : " << duration.count() / ((double)num) << " microseconds" << std::endl;
   // cudaMemcpy(&h_mean, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
   // cudaMemcpy(&h_variance, d_SquaredSum, sizeof(float), cudaMemcpyDeviceToHost);

   // // Finalize mean and variance calculation
   //// h_mean /= arraySize;
   //// h_variance = h_variance/ arraySize - h_mean * h_mean;
   // h_variance = h_variance - h_mean * h_mean;

   // // Output the mean and variance
   // std::cout << "Mean: " << h_mean << std::endl;
   // std::cout << "Variance: " << h_variance << std::endl;

    /////////////////////////////////////////////////////////
    ///////////   4   //////////////////////////////////////////////
    /////////////////////////////////////////////////////////

    // Initialize mean and variance to zero on the device
    int blocks = nRows;
    int ithreads = 1024;

    float* d_arrSumMean = NULL;
    float* d_arrSumMeanSquared = NULL;
    cudaMallocManaged(&d_arrSumMean, ithreads * sizeof(float));
    cudaMallocManaged(&d_arrSumMeanSquared, ithreads * sizeof(float));

    start = std::chrono::high_resolution_clock::now();
    // Calculate mean and variance
    for (int i = 0; i < num; ++i)
    {
        // Calculate mean and variance
        calcRowMeanAndDisp << < blocks, ithreads, 3 * ithreads * sizeof(float) >> >
            (d_arr, nRows, nCols, d_arrSumMean, d_arrSumMeanSquared);
        cudaDeviceSynchronize();

    }

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "4 Time : " << duration.count() / ((double)num) << " microseconds" << std::endl;
    /*for (int i = 0; i < 100; ++i)
    {
        std::cout << "mean: " << d_arrSumMean[i] << ";  disp: " << d_arrSumMeanSquared[i] << std::endl;
    }*/

    // Output the mean and variance
    std::cout << "Mean: " << h_mean << std::endl;
    std::cout << "Variance: " << h_variance << std::endl;

    //for (int i = 0; i < nRows; ++i)
    //{
    //    if (fabs(d_arrSumMeanSquared[i] - /*VAlMean * VAlMean -*/ VAlDisp) > 1.)
    //    {
    //        int ggg = 0;
    //    }
    //}



    /////////////////////////////////////////////////////////
    ///////////   5    //////////////////////////////////////////////
    /////////////////////////////////////////////////////////

    // Initialize mean and variance to zero on the device
    blocks = nRows;
    ithreads = 256;   
    

    start = std::chrono::high_resolution_clock::now();
    // Calculate mean and variance
    for (int i = 0; i < num; ++i)
    {
        // Calculate mean and variance
        calcRowMeanAndDisp << < blocks, ithreads, 3 * ithreads * sizeof(float) >> >
            (d_arr, nRows, nCols, d_arrSumMean, d_arrSumMeanSquared);
        cudaDeviceSynchronize();
        kernel_OneSM_Mean_and_Disp<<<1, 256, 256 *3 * sizeof(float) >> >(d_arrSumMean, d_arrSumMeanSquared, nRows
            , d_sum, d_SquaredSum);
        //cudaDeviceSynchronize();


    }

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "5 Time : " << duration.count() / ((double)num) << " microseconds" << std::endl;
    cudaMemcpy(&h_mean, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_variance, d_SquaredSum, sizeof(float), cudaMemcpyDeviceToHost);

    // Finalize mean and variance calculation
   // h_mean /= arraySize;
   // h_variance = h_variance/ arraySize - h_mean * h_mean;
    //h_variance = h_variance - h_mean * h_mean;

    // Output the mean and variance
    std::cout << "Mean: " << h_mean << std::endl;
    std::cout << "Variance: " << h_variance << std::endl;
    std::cout << "------- " << std::endl;
    /*for (int i = 0; i < 100; ++i)
    {
        std::cout << "mean: " << d_arrSumMean[i] << ";  disp: " << d_arrSumMeanSquared[i] << std::endl;
    }*/
///////////////////////////////////////////////////////////////
     /////////////////////////////////////////////////////////
    ///////////   6   BAD CASE //////////////////////////////////////////////
    /////////////////////////////////////////////////////////
   // int ROWS = nRows;
   // int COLS = nCols;
   // 
   // const int BLOCK_SIZE_ = 256; // Assuming a 4x4 thread block
   // float* cudaResult;
   // cudaMalloc((void**)&cudaResult, (ROWS * COLS + 2) * sizeof(float)); // Mean, overall mean, dispersion

   // int THREADS_PER_BLOCK = 256;
   // blocks = (ROWS * COLS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
   // start = std::chrono::high_resolution_clock::now();
   // // Calculate mean and variance
   // for (int i = 0; i < num; ++i)
   //{
   //     meanAndDispersion << <blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float) >> > (d_arr, ROWS, COLS, cudaResult);
   //     cudaDeviceSynchronize();
   // }
   // end = std::chrono::high_resolution_clock::now();
   // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
   // std::cout << "6 Time : " << duration.count() / ((double)num) << " microseconds" << std::endl;

   // 

   // 
   // float* result = (float*)malloc((blocks + 2) * sizeof(float)); // Mean, overall mean, dispersion
   // cudaMemcpy(result, cudaResult, (blocks + 2) * sizeof(float), cudaMemcpyDeviceToHost);

   // printf("Mean value: %f\n", result[blocks]);
   // printf("Dispersion (Variance): %f\n", result[blocks + 1]);    
   // cudaFree(cudaResult);



    // Free memory on the device
    cudaFree(d_arr);
    cudaFree(d_mean);
    cudaFree(d_variance);
    cudaFree(d_arrSumMeanSquared);
    free(arr);

    return 0;
}

