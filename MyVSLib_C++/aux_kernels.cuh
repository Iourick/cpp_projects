#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Constants.h"

__global__ void calcRowMeanAndDisp(float* d_arrIm, int nRows, int nCols
    , float* arrSumMean, float* arrSumMeanSquared);

__global__ void kernel_OneSM_Mean_and_Std(float* d_arrMeans, float* d_arrDisps, int len
    , float* pmean0, float* pstd);

__global__ void kernel_normalize_array(fdmt_type_* pAuxBuff, const unsigned int len
    , float* pmean, float* pdev, float* parrInp);

__global__ void kernel_calcDispersion_for_Short_Onedimentional_Array(float* d_poutDisp, const float* d_arrInp, const float* d_mean, const int len);

__global__
void calculateMeanAndSTD_for_oneDimArray_kernel(float* d_arr, const unsigned int len, float* pmean, float* pstd);
