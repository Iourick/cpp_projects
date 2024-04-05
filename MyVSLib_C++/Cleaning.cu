#include "Cleaning.cuh"



//----------------------------------------------
// INPUT:
// d_arr - fdmt input array, dimentions NRows and NCols, allocated on GPU
// d_buff - auxillary allocated memory on GPU, sizeof(d_buff) = (NCols
// OUTPUT:
// d_arr - cleaned array
// Algorithm:
//    1. compute dispersion for each row and kepp in array. Lets denote array as d_arrd, length of array  d_arrd = NRows
//    2. compute mean and dispersion for array d_arrd,lets denote them as d_valm and d_valdisp
//    3. if fabs(d_arrd[i]- d_valm)>4. *SQRT(d_valdisp)-->assign to 0 all elements of matrix d_arr with row with number i 
//         d_arr[i][j] = 0, j =0,..,NCols-1  
void cleanInpFDMT_v0(float* d_arr, const int NRows, const int NCols, float* d_buff)
{
    int threads = 128;
    calcRowDisps_kernel << < NRows, threads, threads * 2 * sizeof(float) >> > (d_arr, NRows, NCols, d_buff);
    //cudaDeviceSynchronize();
    float* pmean = d_buff + NRows;
    float* pstd = pmean + 1;
    threads = 128;
    calculateMeanAndSTD_kernel << <1, threads, threads * 2 * sizeof(float) >> > (d_buff, NRows, pmean, pstd);


    cudaDeviceSynchronize();
    const dim3 blockSize(256, 1, 1);

    const dim3 gridSize(1, NRows, 1);
    clean_out_the_trash_kernel_v2 << < gridSize, blockSize, sizeof(int) >> > (d_arr, NRows, NCols, d_buff, *pmean, *pstd);
    cudaDeviceSynchronize();
}
//-----------------------------------------------------------------
__global__ void calcRowDisps_kernel(float* d_arrIm, const int nRows, const int nCols, float* arrDisps)
{
    extern __shared__ float sbuff[];
    float* sdata = sbuff;
    //int* sNums = (int*)((char*)sbuff + 2 * blockDim.x * sizeof(float));

    float* d_arr = d_arrIm + nCols * blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;
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
            sdata[tid] = sdata[tid] + sdata[tid + s];// (sdata[tid] * sNums[tid] + sdata[tid + s] * sNums[tid + s]) / (sNums[tid] + sNums[tid + s]);
            sdata[blockDim.x + tid] = sdata[blockDim.x + tid] + sdata[blockDim.x + tid + s];
        }
        __syncthreads();
    }

    // Only thread 0 within each block computes the block's sum
    if (tid == 0)
    {
        arrDisps[blockIdx.x] = sdata[blockDim.x] / ((float)nCols) - sdata[0] / ((float)nCols) * sdata[0] / ((float)nCols);
    }
    __syncthreads();

}
//--------------------------------------------------
//--------------------------------------------------
__global__
void calculateMeanAndSTD_kernel(float* d_arr, const unsigned int len, float* pmean, float* pstd)
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
//--------------------------------------------------
__global__ void clean_out_the_trash_kernel_v2(float* d_arr, const int NRows, const int NCols, float* d_buff, const float mean, const float std)
{
    extern __shared__ int sbad[];
    unsigned int i = threadIdx.x;
    unsigned int irow = blockIdx.y;
    if (i >= NCols)
    {
        return;
    }
    if (fabs(d_buff[irow] - mean) > 4. * std)
    {
        sbad[0] = 1;
    }
    else
    {
        sbad[0] = 0;
    }
    if (sbad[0] == 1)
    {
        while (i < NCols)
        {
            d_arr[irow * NCols + i] = 0.;
            i += blockDim.x;
        }
    }


}

