#include "cuda_runtime.h"
#include "Statistical_preprocessing.cuh"
#include <chrono>
#include <vector>


	//--------------------------------------------------------------------
//INPUT:
//1.  d_parr_inp -  intensivities matrix with total length  = NRows x NCols 
// NRows, NCols - rows and columns quantity of d_parr_inp
// 2.d_buff - auxillary buffer to compute mean and dispersions for each row of d_parr_inp
//OUTPUT:
//d_parr_inp - cleaned matrix
// d_parrNormInput - input matrix to calculate "normalization fdmt" matrix  with total length  = NRows x NCols 
void statistical_preprocessing::calc_fdmt_inputs(float* d_parr_inp, const int NRows, const int NCols
    , float* d_buff, float* d_parrNormInput)
	{	

    // 1. memory preparations, installing pointers
    // in "d_arrRowMean[i]" we will store mean value for row with number "i"
    // in "d_arrRowDisp[i]" we will store dispersion value for row with number "i"
		float* d_arrRowMean = (float*)d_buff;
		float* d_arrRowDisp = d_arrRowMean + NRows;	

		
        // *pval_mean - mean value for total array "d_parr_inp"
        // *pval_stdDev -dispertion for total array "d_parr_inp"
		float* pval_mean = d_arrRowDisp + NRows;
		float* pval_stdDev = pval_mean + 1;


		float* pval_dispMean = pval_stdDev + 1;
		float* pval_dispStd = pval_dispMean + 1;
        // !1

        // 2. calculations mean values and dispersions for each row of matrix "d_parr_inp"              
		int blocksPerGrid = NRows;
		int treadsPerChunk = calcThreadsForMean_and_Disp(NCols);
		size_t sz1 = (2 * sizeof(float) + sizeof(int)) * treadsPerChunk;
		calcRowMeanAndDisp << < blocksPerGrid, treadsPerChunk, sz1 >> > (d_parr_inp, NRows, NCols, d_arrRowMean, d_arrRowDisp);
		// !2		
       
		// 3. calculations mean value and standart deviation for full "d_parr_inp"		
        // Here, basing on already calculated arrays "d_arrRowMean" and  "d_arrRowDisp", we calculate "*pval_mean" and  "*pval_stdDev"
        // S0 = d_arrRowMean[0] + ...+d_arrRowMean[NRows-1]
        // S1 = (d_arrRowDisp[0] +  d_arrRowMean[0] * d_arrRowMean[0])+..+(d_arrRowDisp[NRows-1] +  d_arrRowMean[NRows-1] * d_arrRowMean[NRows-1])
        // (*pval_mean) = S0/NRows
        // (*pval_stdDev) =sqrt( S1 - (*pval_mean) *(*pval_mean) )
        blocksPerGrid = 1;
		treadsPerChunk = calcThreadsForMean_and_Disp(NRows);
        size_t sz = treadsPerChunk * (2 * sizeof(float) + sizeof(int));
		kernel_OneSM_Mean_and_Std << <blocksPerGrid, treadsPerChunk, sz >> > (d_arrRowMean, d_arrRowDisp, NRows
			, pval_mean, pval_stdDev);		
		// !3

		
		// 4. calculations mean value and standart deviation for array d_arrRowDisp	
        // INPUT:
        // d_arrRowDisp array with size = NRows
        // OUTPUT:
        // *pval_dispMean, *pval_dispStd - mean value and std (standard deviation = sqrt(dispersion)) of the array d_arrRowDisp
		int threads = 128;
		calculateMeanAndSTD_for_oneDimArray_kernel << <1, threads, threads * 2 * sizeof(float) >> > (d_arrRowDisp, NRows, pval_dispMean, pval_dispStd);
        // !4
		
		// 5. Clean and fill input normalization array
		//Input:
        // d_arrRowDisp - array with dispersion for each row of matrix "d_parr_inp", size = "NRows"
        // *pval_dispMean - mean value of array "d_arrRowDisp"
        // *pval_dispStd - STD of array "d_arrRowDisp"
        // *pval_mean - mean value for total array  "d_parr_inp"
        // *pval_stdDev - STD for total array "d_parr_inp"
        // OUTPUT:
        // d_parr_inp
        // d_parrNormInput
        // functionality:
        // for each row with number "i" 
        // if (fabs(d_arrRowDisp[i] - (*pval_dispMean)) > 4.0 * (*pval_dispStd))
        // then set all values of row number "i" of arrays "d_parr_inp" and "d_parrNormInput" with 0
        // else:
        // set row "d_parrNormInput" with number "i" with 1: d_parrNormInput[i][j] = 1;
        // normalize row "d_parr_inp" with number "i": d_parr_inp[i][j] = (d_parr_inp[i][j] - (*pval_mean))/(*pval_stdDev)
		normalize_clean_and_fill_NormInput << < NRows, 256>> >
			( d_parr_inp, NCols
				, pval_mean, pval_stdDev, d_arrRowDisp, pval_dispMean, pval_dispStd, d_parrNormInput);
        // !5
		
	}
    //----------------------------------------------
    // INPUT:
    // d_arr - fdmt input array, dimentions NRows and NCols, allocated on GPU
    // d_buff - auxillary allocated memory on GPU, sizeof(d_buff) = (NCols
    // OUTPUT:
    // d_arr - cleaned array
    // d_pbarrNorm - =false, if corresponding element of d_arr was cleaned up
    // Algorithm:
    //    1. compute dispersion for each row and keep in array. Lets denote array as d_arrd, length of array  d_arrd = NRows
    //    2. compute mean and dispersion for array d_arrd,lets denote them as d_valm and d_valdisp
    //    3. if fabs(d_arrd[i]- d_valm)>4. *SQRT(d_valdisp)-->assign to 0 all elements of matrix d_arr with row with number i 
    //         d_arr[i][j] = 0, j =0,..,NCols-1  
    // 
    void statistical_preprocessing::clean_fdmtInput_and_create_normInput(float* d_arr, const int NRows, const int NCols
        ,bool* d_pbarrNorm, float* d_buff)
    {
        int threads = 128;
        calcRowDisps_kernel << < NRows, threads, threads * 2 * sizeof(float) >> > (d_arr, NRows, NCols, d_buff);
        //cudaDeviceSynchronize();
        float* pmean = d_buff + NRows;
        float* pstd = pmean + 1;
        threads = 128;
        calculateMeanAndSTD_kernel << <1, threads, threads * 2 * sizeof(float) >> > (d_buff, NRows, pmean, pstd);


        //cudaDeviceSynchronize();
        const dim3 blockSize(256, 1, 1);

        const dim3 gridSize(1, NRows, 1);
        clean_out_the_trash_kernel_v2 << < gridSize, blockSize, sizeof(int) >> > (d_arr, NRows, NCols, d_buff, *pmean, *pstd);
        //cudaDeviceSynchronize();
    }


void cleanInpFDMT_v0(float* d_arr, const int NRows, const int NCols, float* d_buff)
{
    int threads = 128;
    calcRowDisps_kernel << < NRows, threads, threads * 2 * sizeof(float) >> > (d_arr, NRows, NCols, d_buff);
    //cudaDeviceSynchronize();
    float* pmean = d_buff + NRows;
    float* pstd = pmean + 1;
    threads = 128;
    calculateMeanAndSTD_kernel << <1, threads, threads * 2 * sizeof(float) >> > (d_buff, NRows, pmean, pstd);


   // cudaDeviceSynchronize();
    const dim3 blockSize(256, 1, 1);

    const dim3 gridSize(1, NRows, 1);
    clean_out_the_trash_kernel_v2 << < gridSize, blockSize, sizeof(int) >> > (d_arr, NRows, NCols, d_buff, *pmean, *pstd);
   //cudaDeviceSynchronize();
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
__global__ void clean_out_the_trash_kernel_v2(float* d_arr, const int NRows, const int NCols, float* d_buff
    , const float mean, const float std)
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
    , float* pmean0, float* pstd)
{
    extern __shared__ float sbuff[];
    float* sdata = sbuff;   

    unsigned int tid = threadIdx.x;
    unsigned int i = tid;
    if (tid >= len)
    {
        return;
    }

    float localSum0 = 0.0f;
    float localSum1 = 0.0f;
   
    // Calculate partial sums within each block   
    while (i < len)
    {
        localSum0 += d_arrMeans[i];
        localSum1 += d_arrDisps[i] + d_arrMeans[i] * d_arrMeans[i];
        i += blockDim.x;
        
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
            sdata[tid] = sdata[tid] + sdata[tid + s];
            sdata[blockDim.x + tid] = sdata[blockDim.x + tid] + sdata[blockDim.x + tid + s];          
        }
        __syncthreads();
    }

    // Only thread 0 within each block computes the block's sum
    if (tid == 0)
    {
        *pmean0 = sdata[0] / ((float)len);
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
    int* sNums = (int*)((char*)sbuff + blockDim.x * sizeof(float));
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
        if ((tid < s) && ((tid + s) < len))
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
//---------------------------------------------------------------
        //Input:
        // d_arrRowDisp - array with dispersion for each row of matrix "d_parr_inp", size = "NRows"
        // *pval_dispMean - mean value of array "d_arrRowDisp"
        // *pstdDisp - STD of array "d_arrRowDisp"
        // *pmean - mean value for total array  "d_parr_inp"
        // *pstd - STD for total array "d_parr_inp"
        // NCols - column's quantity of matrix "d_arr"
        // OUTPUT:
        //d_arr - intesivity matrix, NRows x NCols
        // d_parrNormInput - input matrix to calculate normalization
        // functionality:
        // for each row with number "i" 
        // if (fabs(d_arrRowDisp[i] - (*pval_dispMean)) > 4.0 * (*pstdDisp))
        // then set all values of row number "i" of arrays "d_arr" and "d_parrNormInput" with 0
        // else:
        // set row "d_parrNormInput" with number "i" with 1: d_parrNormInput[i][j] = 1;
        // normalize row "d_arr" with number "i": d_arr[i][j] = (d_arr[i][j] - (*pmean))/(*pstd)
        // sample for call:
        //normalize_clean_and_fill_NormInput << < NRows, 256 >> >
        //(d_parr_inp, NCols    , pval_mean, pval_stdDev, d_arrRowDisp, pval_dispMean, pval_dispStd, d_parrNormInput);
       // NRows - rows quantity of matrixes "d_arr" and "d_parrNormInput", and size of array "d_arrRowDisp"
__global__
void normalize_clean_and_fill_NormInput(float* d_arr, const int NCols
    , float* pmean, float* pstd, float* d_arrRowDisp, float* pmeanDisp, float* pstdDisp, float* d_parrNormInput)
{    
    unsigned int i = threadIdx.x;
    unsigned int irow = blockIdx.x;
    if (i >= NCols)
    {
        return;
    }  
    
    if (fabsf(d_arrRowDisp[irow] - (*pmeanDisp) )> 4. * (*pstdDisp))
    {
        while (i < NCols)
        {              
            d_arr[irow * NCols + i] = 0.0f;
            d_parrNormInput  [irow * NCols + i] = 0.0f;        
            i += blockDim.x;
        }
    }
    else
    {      
        while (i < NCols)
        {          
            d_arr[irow * NCols + i] = ((d_arr[irow * NCols + i] - (*pmean)) / ((*pstd)));         
            d_parrNormInput[irow * NCols + i] = 1.0f;
            i += blockDim.x;
        }
    }
}

